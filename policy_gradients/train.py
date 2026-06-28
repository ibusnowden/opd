# Policy Gradient Training — helper functions
#
# Original implementation by Zafir Stojanovski (@zafstojano) — https://github.com/zafstojano/policy-gradients (Apache 2.0)
# Adapted for RLHF Book (https://rlhfbook.com) by Nathan Lambert.
#
# Vendored into research/ from /project/inniang/vibe/code/policy_gradients/train.py and trimmed so the
# whole stack lives under research/:  the standalone `main()` / `main_cli()` (and their `import mlrunx`
# logging) were removed — the training entry point is `research/harness/unified_trainer.py`, which calls
# the helper functions below (rollout, compute_log_probs/values/rewards/advantages, load_model,
# setup_distributed, get_loss_objective, create_dataset, iter_training_batches, ...). No logging here.

import os
import platform
import random
import re
import time
from dataclasses import dataclass
from datetime import timedelta
from itertools import batched  # Requires Python 3.12+

import numpy as np
import reasoning_gym as rg
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from reasoning_gym.composite import DatasetSpec
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.utils import SYSTEM_PROMPTS, extract_answer
from rich.console import Console
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.utils import is_flash_attn_2_available, is_flash_attn_3_available

from .buffer import Experience, ReplayBuffer, join_experiences_batch
from .config import Config, load_config
from .loss import CISPOLoss, GRPOLoss, GSPOLoss, PPOLoss, ReinforceLoss, approx_kl, masked_mean


@dataclass(frozen=True)
class DistEnv:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    enabled: bool = False

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistEnv:
    """Initialize process group from torchrun environment (if enabled)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1

    if enabled and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=30),
        )

    if enabled and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return DistEnv(rank=rank, local_rank=local_rank, world_size=world_size, enabled=enabled)


def cleanup_distributed(dist_env: DistEnv) -> None:
    """Tear down process group after training."""
    if dist_env.enabled and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    """Return the underlying model when wrapped in DDP."""
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def get_model_device(model) -> torch.device:
    """Infer model device from parameters (works for plain and DDP models)."""
    return next(unwrap_model(model).parameters()).device


def _to_scalar(value: float | torch.Tensor) -> float:
    """Convert scalar-like values to Python float."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(torch.float32).item()
    return float(value)


def distributed_sum(value: float | torch.Tensor, device: torch.device, dist_env: DistEnv) -> float:
    """Compute scalar sum across all ranks."""
    reduced = torch.tensor(_to_scalar(value), dtype=torch.float32, device=device)
    if dist_env.enabled:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced.item()


def distributed_mean(value: float | torch.Tensor, device: torch.device, dist_env: DistEnv) -> float:
    """Compute mean scalar across all ranks."""
    total = distributed_sum(value, device, dist_env)
    if dist_env.enabled:
        return total / dist_env.world_size
    return total


def distributed_max(value: float | torch.Tensor, device: torch.device, dist_env: DistEnv) -> float:
    """Compute scalar max across all ranks."""
    reduced = torch.tensor(_to_scalar(value), dtype=torch.float32, device=device)
    if dist_env.enabled:
        dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
    return reduced.item()


def filter_numeric_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Keep only finite numeric values for logging."""
    return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and np.isfinite(v)}


def get_gpu_metrics() -> dict[str, float]:
    """Collect GPU metrics for logging (best-effort, no hard dependency on NVML)."""
    metrics: dict[str, float] = {}
    if not torch.cuda.is_available():
        return metrics

    device_count = torch.cuda.device_count()
    metrics["gpu/device_count"] = float(device_count)

    nvml = None
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        nvml = pynvml
    except Exception:
        nvml = None

    for idx in range(device_count):
        prefix = f"gpu/{idx}" if device_count > 1 else "gpu"
        total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        allocated_gb = torch.cuda.memory_allocated(idx) / (1024**3)
        reserved_gb = torch.cuda.memory_reserved(idx) / (1024**3)
        max_allocated_gb = torch.cuda.max_memory_allocated(idx) / (1024**3)
        max_reserved_gb = torch.cuda.max_memory_reserved(idx) / (1024**3)

        metrics[f"{prefix}/memory_total_gb"] = total_gb
        metrics[f"{prefix}/memory_allocated_gb"] = allocated_gb
        metrics[f"{prefix}/memory_reserved_gb"] = reserved_gb
        metrics[f"{prefix}/memory_max_allocated_gb"] = max_allocated_gb
        metrics[f"{prefix}/memory_max_reserved_gb"] = max_reserved_gb
        metrics[f"{prefix}/memory_used_percent"] = (allocated_gb / total_gb) * 100.0 if total_gb > 0 else 0.0

        if nvml is not None:
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(idx)
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                metrics[f"{prefix}/utilization_percent"] = float(util.gpu)
                metrics[f"{prefix}/memory_utilization_percent"] = float(util.memory)
                metrics[f"{prefix}/temperature_c"] = float(
                    nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                )
                try:
                    metrics[f"{prefix}/power_w"] = float(nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
                except Exception:
                    pass
            except Exception:
                pass

    if nvml is not None:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass

    return metrics


def get_attn_implementation() -> str:
    """Determine the best attention implementation for this platform.

    Priority order:
    1) flash_attention_3 (if available),
    2) flash_attention_2 (if available),
    3) sdpa fallback.
    """
    if platform.machine() != "x86_64":
        return "sdpa"  # aarch64 / DGX Spark - use SDPA with cuDNN

    if is_flash_attn_3_available():
        return "flash_attention_3"
    if is_flash_attn_2_available():
        return "flash_attention_2"
    return "sdpa"


def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model_name: str, device: torch.device, gradient_checkpointing: bool = True):
    """Load model and tokenizer with automatic attention implementation selection."""
    attn_impl = get_attn_implementation()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
    if hasattr(model_config, "tie_word_embeddings"):
        model_config.tie_word_embeddings = False
    # Many decoder-only models (LLaMA, GPT-2) don't define pad_token
    # Set it to eos_token to enable batch padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=model_config,
        trust_remote_code=False,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
    )
    model = model.to(device)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, tokenizer


def get_ref_model(model_name: str, device: torch.device, beta: float):
    """Load reference model for KL penalty (only if beta > 0)."""
    if not beta:
        return None
    ref_model, _ = load_model(model_name, device, gradient_checkpointing=False)
    ref_model.eval()
    return ref_model


def get_val_model(model_name: str, device: torch.device, loss: str, gradient_checkpointing: bool = True):
    """Load value model for PPO (only if loss == 'ppo')."""
    if loss not in ["ppo"]:
        return None
    val_model, _ = load_model(model_name, device, gradient_checkpointing)
    val_device = next(val_model.parameters()).device
    val_model.lm_head = nn.Linear(
        val_model.lm_head.in_features, 1, bias=False, device=val_device, dtype=torch.bfloat16
    )
    return val_model


def get_loss_objective(loss: str, **kwargs) -> nn.Module:
    """Get the loss function module for the specified algorithm."""
    if loss in ["grpo", "drgrpo"]:
        return GRPOLoss(**kwargs)
    elif loss == "gspo":
        return GSPOLoss(**kwargs)
    elif loss in ["rloo", "reinforce"]:
        return ReinforceLoss(**kwargs)
    elif loss == "cispo":
        return CISPOLoss(**kwargs)
    elif loss == "ppo":
        return PPOLoss(**kwargs)
    raise ValueError(f"Unsupported loss type: {loss}")


def _accuracy_reward(dataset: ProceduralDataset, completions: str, entries: list[dict]) -> float:
    """Compute accuracy reward based on extracted answers."""

    def score_answer(completion: str, entry: dict) -> float:
        answer = extract_answer(completion)
        return dataset.score_answer(answer, entry)

    return [score_answer(c, e) for c, e in zip(completions, entries, strict=True)]


def _format_reward(completions: list[str], **kwargs) -> list[float]:
    """Compute format reward based on presence of thinking/answer tags."""

    def count_tags(text: str) -> float:
        count = 0.0
        if re.search(r"\s*<think>\s*", text):
            count += 0.25
        if re.search(r"\s*</think>\s*", text):
            count += 0.25
        if re.search(r"\s*<answer>\s*", text):
            count += 0.25
        if re.search(r"\s*</answer>\s*", text):
            count += 0.25
        return count

    return [count_tags(c) for c in completions]


def compute_rewards(
    dataset: ProceduralDataset, completions: list[str], entries: list[dict], format_weight: float = 0.5
) -> dict[str, list[float]]:
    """Compute accuracy + format + combined rewards.

    Returns a dict with three keys: ``accuracy``, ``format``, ``combined`` (each a list of
    per-completion floats). ``combined = accuracy + format_weight * format`` is what the trainer
    optimizes (`rewards` tensor); ``accuracy`` and ``format`` are logged separately so we can tell
    "the model learned the answer" from "the model learned the format tags".
    """
    accuracy_rewards = _accuracy_reward(dataset, completions, entries)
    format_rewards = _format_reward(completions)
    combined_rewards = [acc + format_weight * fmt for acc, fmt in zip(accuracy_rewards, format_rewards, strict=True)]
    return {"accuracy": accuracy_rewards, "format": format_rewards, "combined": combined_rewards}


def apply_reward_kl(
    rewards: torch.Tensor,
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: torch.Tensor,
    beta: float,
    loss: str,
) -> torch.Tensor:
    """Apply KL penalty to rewards (for REINFORCE/RLOO/PPO)."""
    if not beta or loss not in ["ppo", "rloo", "reinforce"]:
        return rewards
    kl_div = approx_kl(log_probs, log_probs_ref, action_mask)
    kl_div = masked_mean(kl_div, mask=action_mask, dim=-1, keepdim=True)
    rewards = rewards - beta * kl_div
    return rewards


def compute_standardized_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute standardized advantages (GRPO, GSPO, CISPO)."""
    return (rewards - rewards.mean(dim=0, keepdim=True)) / (rewards.std(dim=0, keepdim=True) + eps)


def compute_nonstandardized_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Compute non-standardized advantages (Dr. GRPO)."""
    return rewards - rewards.mean(dim=0, keepdim=True)


def compute_loo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Compute leave-one-out advantages (RLOO)."""
    K = rewards.shape[0]
    return (K / (K - 1)) * (rewards - rewards.mean(dim=0, keepdim=True))


def compute_gae(
    rewards: torch.Tensor, action_mask: torch.Tensor, values: torch.Tensor, gamma: float, lam: float
) -> torch.Tensor:
    """Compute Generalized Advantage Estimation (PPO)."""
    B, S = action_mask.size()
    device = action_mask.device
    last_action_indices = action_mask.long().cumsum(dim=-1).argmax(dim=-1, keepdim=True)
    indices = torch.arange(S, device=device).unsqueeze(0)
    done = (indices >= last_action_indices).float()

    rewards = torch.zeros_like(action_mask, device=device, dtype=torch.float32).scatter_(
        dim=-1, index=last_action_indices, src=rewards
    )

    values = values.to(device)
    advantages = torch.zeros_like(action_mask, dtype=torch.float32, device=device)
    next_values = torch.zeros(B, device=device, dtype=torch.float32)
    running = torch.zeros(B, device=device, dtype=torch.float32)

    for t in reversed(range(S)):
        not_done = 1.0 - done[:, t]
        delta = rewards[:, t] + not_done * gamma * next_values - values[:, t]
        running = delta + not_done * gamma * lam * running
        advantages[:, t] = running
        next_values = values[:, t]

    advantages = advantages * action_mask
    return advantages


def compute_advantages(
    rewards: torch.Tensor,
    loss: str,
    action_mask: torch.Tensor | None = None,
    values: torch.Tensor | None = None,
    gamma: float | None = None,
    lam: float | None = None,
) -> torch.Tensor:
    """Compute advantages using the appropriate method for the loss function."""
    if loss in ["grpo", "gspo", "cispo"]:
        return compute_standardized_advantages(rewards)
    elif loss in ["drgrpo"]:
        return compute_nonstandardized_advantages(rewards)
    elif loss in ["rloo"]:
        return compute_loo_advantages(rewards)
    elif loss in ["ppo"]:
        return compute_gae(rewards, action_mask, values, gamma, lam)
    else:
        return rewards


def compute_log_probs(model, sequence_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute log probabilities for each token in the sequence."""
    if not model:
        return None
    model_device = get_model_device(model)
    sequence_ids, attention_mask = sequence_ids.to(model_device), attention_mask.to(model_device)
    output = model(input_ids=sequence_ids, attention_mask=attention_mask, use_cache=False)
    logits = output.logits[:, :-1, :].to(torch.float32)
    log_probs = F.log_softmax(logits, dim=-1)
    targets = sequence_ids[:, 1:].unsqueeze(-1)
    target_log_probs = torch.gather(log_probs, dim=-1, index=targets).squeeze(-1)
    return target_log_probs


def compute_values(model, sequence_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute value estimates for each position (PPO)."""
    if not model:
        return None
    model_device = get_model_device(model)
    sequence_ids, attention_mask = sequence_ids.to(model_device), attention_mask.to(model_device)
    output = model(input_ids=sequence_ids, attention_mask=attention_mask, use_cache=False)
    values = output.logits[:, :-1, :].squeeze(-1).to(torch.float32)
    return values


def rollout(
    model,
    entries: list[dict],
    dataset: ProceduralDataset,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
):
    """Generate completions and compute rewards."""
    model_device = get_model_device(model)
    generation_model = unwrap_model(model)

    # 1. Format prompts
    message_templates = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPTS["DeepSeekZero"]},
                {"role": "user", "content": entry["question"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for entry in entries
    ]
    model_inputs = tokenizer(
        message_templates,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to(model_device)

    # 2. Generate responses
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_token_id,
    )
    sequence_ids = generation_model.generate(**model_inputs, generation_config=generation_config)
    completion_ids = sequence_ids[:, model_inputs["input_ids"].shape[1] :]
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    # 3. Obtain the generated tokens only
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, model_inputs["input_ids"].shape[1] :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]

    # 4. Compute rewards (split components: accuracy / format / combined)
    reward_components = compute_rewards(dataset, completions, entries)
    rewards = torch.tensor(reward_components["combined"], dtype=torch.float32, device=model_device).unsqueeze(-1)
    accuracy = torch.tensor(reward_components["accuracy"], dtype=torch.float32, device=model_device).unsqueeze(-1)
    format_score = torch.tensor(reward_components["format"], dtype=torch.float32, device=model_device).unsqueeze(-1)

    # 5. Compute attention mask
    attention_mask = sequence_ids != tokenizer.pad_token_id

    return sequence_ids, action_mask, attention_mask, rewards, completions, accuracy, format_score


def create_dataset(cfg: Config) -> ProceduralDataset:
    """Create the training dataset from config."""
    specs = [DatasetSpec(name=s.name, weight=s.weight, config=s.config) for s in cfg.data.specs]
    return rg.create_dataset("composite", size=cfg.data.size, seed=cfg.seed, datasets=specs)


def iter_training_batches(
    dataloader: DataLoader,
    sampler: DistributedSampler | None,
    num_steps: int | None,
):
    """Yield (step, batch) pairs, optionally cycling to reach an exact step budget."""
    if num_steps is None:
        yield from enumerate(dataloader)
        return

    if num_steps < 1:
        return

    epoch = 0
    step = 0
    data_iter = iter(dataloader)
    while step < num_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(dataloader)
            try:
                batch = next(data_iter)
            except StopIteration:
                break

        yield step, batch
        step += 1


