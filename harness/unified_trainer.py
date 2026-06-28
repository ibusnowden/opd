"""unified_trainer — the (alpha, lambda, pi_T) training loop.

Generalizes the reference `policy_gradients` training loop: a teacher term (reverse-KL toward pi_T)
and an outcome-reward term, mixed by `lam`, sampled with on-policy fraction `alpha`. The four
corners (SFT / RL / OPD / OPSD) are config points (see ./configs/, ./README.md). The reference
policy-gradient code is vendored under research/policy_gradients/ and surfaced via `harness._pg`
(plain eager imports — the whole stack lives under research/).

IMPLEMENTED:
  * RL corner — (alpha=1, lam=0): a full on-policy policy-gradient loop (GRPO/RLOO/GSPO/CISPO/PPO via
    cfg.outcome_loss), ported from the reference `policy_gradients.train.main`, using the `_pg.*` helpers
    + W&B logging. `_run_rl_loop`. **Multi-GPU DDP** when launched under torchrun (WORLD_SIZE>1):
    DDP-wrapped student, per-rank `DistributedSampler` on the prompt loader, grad all-reduce, metric
    reduction, rank-0-only logging (group-relative + REINFORCE objectives; PPO + DDP raises). See
    `run_h100_ddp.sh` / `run_research.sh` (NPROC>1).
  * lam > 0 path — (alpha=1, lam>0): the per-token teacher reverse-KL term — OPD (teacher=same_family),
    OPSD-style per-token KL clip, expert-RL+OPD (0<lam<1). `_run_distill_loop`: same rollout/experience
    machinery as `_run_rl_loop` plus a frozen-teacher forward per training batch, blended with the
    configured outcome objective according to the effective lambda schedule.
    Single GPU (raises under torchrun); teacher on `cfg.teacher.device_id`. (Plain OPD `teacher=same_family`
    is end-to-end; `teacher=self` / `hint_writer` conditioning is still stubbed in `harness.teachers`.)
STUBBED (TODO / NotImplementedError):
  * SFT corner (alpha=0, lam=1, teacher=dataset); off-policy mixing (0 < alpha < 1) + IS correction;
    `teacher=self`/`hint_writer` privileged-info conditioning (harness.teachers._build_inputs);
    LoRA / quantization wiring (cfg.fit.*); DDP for the lam>0 path; PPO + DDP; FSDP (7B+ student); eval hooks.

Run (from research/):
    python -m harness.unified_trainer --config harness/configs/rl_grpo.yaml
    # or:  sbatch harness/run_research.sh harness/configs/rl_grpo.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import batched  # py3.12+

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import _pg  # vendored policy_gradients: rollout, compute_*, load_model, setup_distributed, DistEnv, distributed_*, ...
from .config import ResearchConfig, load_config
from .distill_losses import prm_importance_weights, reverse_kl_distill_advantage, sft_ce_loss  # noqa: F401  (sft_ce_loss for train_sft once built; reverse_kl_distill_advantage + prm_importance_weights used by the distill path)
from .teachers import Teacher, PrivilegedInfoTeacher, HintWriterTeacher, PRMTeacher, build_teacher  # noqa: F401


def _teacher_needs_entries(t: Teacher) -> bool:
    """True if `t.token_logprobs` needs `entries` to do its work (so we must call it at rollout
    time and cache the result in Experience). PrivilegedInfoTeacher (OPSD / PRM-as-teacher) and
    HintWriterTeacher both build their conditioning context from `entries`; plain SameFamily /
    Dataset / NoTeacher are stateless w.r.t. entries."""
    return isinstance(t, (PrivilegedInfoTeacher, HintWriterTeacher))
from .wandb_logging import Logger


# --- logger ------------------------------------------------------------------

def _build_logger(cfg: ResearchConfig, is_main: bool) -> Logger:
    return Logger.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name or f"{cfg.recipe}_seed{cfg.seed}",
        group=cfg.wandb_group,
        tags=cfg.wandb_tags or [cfg.recipe],
        config=cfg.model_dump(),
        is_main=is_main,
    )


# --- model / optimizer (cfg.fit-aware) --------------------------------------

def _load_student(cfg: ResearchConfig, device: torch.device):
    """Load (model, tokenizer) for the student, honouring cfg.fit.

    Like policy_gradients.train.load_model (bf16, auto attn impl, gradient checkpointing) BUT it
    does NOT force `tie_word_embeddings = False`: upstream unties unconditionally (only PPO's value
    model needs a separate lm_head, and that's handled by get_val_model), which leaves a tied
    checkpoint's lm_head randomly initialized. We respect the checkpoint's tying here. LoRA/4-bit
    loading is still TODO (cfg.fit.student_lora / student_load_in_4bit) — at OLMo-2-1B(<-7B teacher)
    scale on the 6000 Ada (48 GB) you don't need it (full FT + gradient checkpointing fits).
    """
    if cfg.fit.student_lora or cfg.fit.student_load_in_4bit:
        raise NotImplementedError("LoRA / 4-bit student loading not wired yet (cfg.fit.student_*).")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, trust_remote_code=False,
        attn_implementation=_pg.get_attn_implementation(), dtype=torch.bfloat16,
    ).to(device)
    if cfg.fit.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if cfg.fit.optimizer_8bit:
        # Keep embedding (and untied lm_head) optimizer state in 32-bit when using bnb 8-bit Adam:
        # 8-bit moments on the large vocab embedding are the documented instability source. Must run
        # BEFORE _build_optimizer (GlobalOptimManager is a global singleton keyed by module/param).
        import bitsandbytes as bnb
        mng = bnb.optim.GlobalOptimManager.get_instance()
        n_override = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                mng.register_module_override(module, "weight", {"optim_bits": 32})
                n_override += 1
        print(f"[unified_trainer] 8-bit Adam: registered 32-bit optim override on {n_override} embedding module(s)")
    return model, tokenizer


def _build_optimizer(cfg: ResearchConfig, params):
    if cfg.fit.cpu_offload_optimizer:
        # TODO: DeepSpeed ZeRO-Offload / FSDP CPU offload (not needed once 8-bit Adam fits one card).
        raise NotImplementedError("CPU-offload optimizer not wired yet (cfg.fit.cpu_offload_optimizer).")
    if cfg.fit.optimizer_8bit:
        # bitsandbytes 8-bit Adam: keeps the m/v moments in 8-bit (~4x smaller optimizer state) so a
        # 7B student full-FT fits one 80 GB H100 (vs ~84 GB for fp32-state Adam). Embedding params are
        # kept in 32-bit optim state (registered in _load_student via GlobalOptimManager) — bnb's
        # documented mitigation for the 8-bit-embedding instability. Plain Adam8bit (no weight decay)
        # matches the reference loop's torch.optim.Adam semantics.
        import bitsandbytes as bnb
        return bnb.optim.Adam8bit(params, lr=cfg.lr)
    # Adam (matches the reference loop); switch to AdamW if you want decoupled weight decay.
    return torch.optim.Adam(params, lr=cfg.lr)


# --- eval hook + checkpoint saving (shared by both training loops) ----------

def _eval_k_values(n_samples: int) -> list[int]:
    return [k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256) if k <= n_samples]


def _make_eval_dataset(cfg: ResearchConfig):
    """Held-out reasoning_gym set for the in-loop pass@k eval — first task spec, seed shifted off training."""
    if cfg.eval_every <= 0:
        return None
    from . import eval_passk  # lazy: pulls reasoning_gym; only needed when the eval hook is on
    task = cfg.data.specs[0].name
    return eval_passk._eval_dataset(task, cfg.eval_n_prompts, cfg.seed + cfg.eval_seed_offset)


def _maybe_eval(cfg: ResearchConfig, model, tokenizer, eval_dataset, log: Logger,
                step: int, total_steps: int, is_main: bool, ddp: bool) -> None:
    """Run harness.eval_passk on the held-out set at periodic / final steps; log eval/* (rank 0 only).

    Skipped under DDP for now (eval-during-DDP would desync the ranks' gradient buckets — eval the
    saved checkpoints post-hoc with `python -m harness.eval_passk` instead)."""
    if cfg.eval_every <= 0 or eval_dataset is None or ddp or not is_main:
        return
    if not ((step + 1) % cfg.eval_every == 0 or step + 1 == total_steps):
        return
    from . import eval_passk
    for T in cfg.eval_temps:
        m = eval_passk.evaluate_passk(
            model, tokenizer, eval_dataset,
            n_prompts=cfg.eval_n_prompts, n_samples=cfg.eval_n_samples,
            k_values=_eval_k_values(cfg.eval_n_samples), temperature=T,
            top_p=cfg.top_p, top_k=cfg.top_k, min_p=cfg.min_p, max_new_tokens=cfg.max_new_tokens,
            gen_batch_size=max(cfg.rollout_batch_size, 16), compute_self_bleu=False,
            tag=(f"T{T}" if len(cfg.eval_temps) > 1 else ""),
        )
        log.log(_pg.filter_numeric_metrics(m), step=step)
        pk = ", ".join(f"{kk.split('/')[-1]}={vv:.3f}" for kk, vv in sorted(m.items())
                       if kk.startswith("eval/") and ("pass@" in kk or "accuracy" in kk or "entropy" in kk))
        print(f"[harness] eval @ step {step + 1}/{total_steps} (T={T}): {pk}")


def _save_checkpoint(cfg: ResearchConfig, model, tokenizer, is_main: bool) -> None:
    """save_pretrained the final student (rank 0) — for post-hoc pass@k eval and #5 sparse-vs-dense."""
    if not cfg.save_ckpt or not is_main:
        return
    import os
    run = cfg.wandb_run_name or f"{cfg.recipe}_seed{cfg.seed}"
    out = cfg.ckpt_dir or os.path.join("harness", "checkpoints", run)
    os.makedirs(out, exist_ok=True)
    _pg.unwrap_model(model).save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    print(f"[harness] saved final checkpoint -> {out}")


# --- the loops ---------------------------------------------------------------

def train_sft(cfg: ResearchConfig, log: Logger) -> None:
    """SFT corner: (alpha=0, lam=1, teacher=dataset). Plain NLL on the demonstration tokens."""
    # TODO: DataLoader over (prompt, demonstration) pairs (teacher completions), tokenize with the
    #       student tokenizer, mask the prompt, minimize harness.distill_losses.sft_ce_loss, log to W&B.
    raise NotImplementedError(
        "SFT data path not implemented yet. Needs a teacher-completions dataset + prompt-masking + "
        "the standard NLL loop; the loss itself is `harness.distill_losses.sft_ce_loss`."
    )


def _run_rl_loop(cfg: ResearchConfig, log: Logger, dist_env=None) -> None:
    """On-policy policy gradient, RL corner (alpha=1, lam=0). Port of policy_gradients.train.main.

    Single GPU by default; **multi-GPU DDP** when launched under torchrun (WORLD_SIZE>1) — the student
    is wrapped in `DistributedDataParallel`, each rank rolls out its own shard of prompts (via a
    `DistributedSampler` on the prompt dataloader), gradients all-reduce on `.backward()`, and the
    logged scalars are reduced across ranks (only rank 0 prints / logs to W&B). DDP supports the
    group-relative + REINFORCE objectives; **PPO + DDP is not wired** (the value model would also need
    wrapping) — run NPROC=1 for PPO. (The lam>0 / distill path is single-GPU only — see `_run_distill_loop`.)

    cfg.outcome_loss selects the objective: grpo / drgrpo / gspo / cispo / rloo / reinforce / ppo.
    Optional KL-to-base penalty via cfg.beta (loads a frozen reference model). Logs reward / loss /
    grad_norm / throughput / off-policy-drift / GPU stats to W&B each step.
    """
    if dist_env is None:
        dist_env = _pg.DistEnv()  # single-process default
    is_main = dist_env.is_main_process
    ddp = dist_env.enabled

    cpu = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{dist_env.local_rank}") if ddp else torch.device(f"cuda:{cfg.model_device_id}")
    else:
        device = torch.device("cpu")

    # ---- model, tokenizer, dataset, optimizer ----
    model, tokenizer = _load_student(cfg, device)
    if getattr(tokenizer, "chat_template", None) is None:
        raise RuntimeError(
            f"Tokenizer for {cfg.model_name!r} has no chat_template — the reasoning_gym rollout uses "
            "tokenizer.apply_chat_template(...). Use an instruction-tuned checkpoint (e.g. "
            "allenai/OLMo-2-0425-1B-Instruct) or set a chat_template on the tokenizer."
        )
    dataset = _pg.create_dataset(cfg)
    if ddp:
        train_sampler = DistributedSampler(dataset, num_replicas=dist_env.world_size, rank=dist_env.rank,
                                           shuffle=True, drop_last=True)
        dataloader = DataLoader(dataset=dataset, batch_size=cfg.prompts_per_step, sampler=train_sampler,
                                pin_memory=False, drop_last=True, collate_fn=lambda x: x)
    else:
        train_sampler = None
        dataloader = DataLoader(dataset=dataset, batch_size=cfg.prompts_per_step, shuffle=True,
                                pin_memory=False, drop_last=True, collate_fn=lambda x: x)
    steps_per_epoch = len(dataloader)
    if steps_per_epoch == 0:
        raise ValueError("Dataloader has 0 batches — increase data.size or lower prompts_per_step.")
    total_steps = cfg.num_steps if cfg.num_steps is not None else steps_per_epoch

    ref_model = _pg.get_ref_model(cfg.model_name, device, cfg.beta)          # None unless beta > 0
    val_model = _pg.get_val_model(cfg.model_name, device, cfg.outcome_loss)  # None unless outcome_loss == "ppo"
    if ddp and val_model is not None:
        raise NotImplementedError(
            "PPO + DDP is not wired (the value model would also need DDP-wrapping and the PPO loss "
            "touches both). Run NPROC=1 for PPO, or use a group-relative objective (grpo/drgrpo/gspo/cispo/rloo)."
        )
    if ddp:
        # find_unused_parameters=False (default): a plain causal-LM full forward+backward has no unused
        # params. gradient checkpointing uses use_reentrant=False (set in _load_student) — DDP-friendly.
        # NOTE: grad-accumulation micro-batches all-reduce on every backward (no model.no_sync()) — correct,
        # just batch_acc× the comm; a TODO if it ever matters at this scale.
        model = DistributedDataParallel(model, device_ids=[dist_env.local_rank])
    objective = _pg.get_loss_objective(
        loss=cfg.outcome_loss,
        clip_eps_lo=cfg.clip_eps_lo, clip_eps_hi=cfg.clip_eps_hi,
        clip_eps_val=cfg.clip_eps_val, vf_coef=cfg.vf_coef, beta=cfg.beta,
    ).to(device)
    params = list(model.parameters()) + (list(val_model.parameters()) if val_model else [])
    optimizer = _build_optimizer(cfg, params)
    replay_buffer = _pg.ReplayBuffer()
    eval_dataset = _make_eval_dataset(cfg)   # None unless cfg.eval_every > 0

    if is_main:
        print(f"[harness] RL corner: outcome_loss={cfg.outcome_loss} model={cfg.model_name} "
              f"attn={_pg.get_attn_implementation()} device={device} ddp={ddp}(world_size={dist_env.world_size}) "
              f"steps={total_steps} prompts/step/rank={cfg.prompts_per_step} rollouts/prompt={cfg.num_rollouts} beta={cfg.beta} "
              f"eval_every={cfg.eval_every} save_ckpt={cfg.save_ckpt}")
    log.log_params({
        "recipe": cfg.recipe, "alpha": cfg.alpha, "lam": cfg.lam, "outcome_loss": cfg.outcome_loss,
        "model_name": cfg.model_name, "lr": cfg.lr, "beta": cfg.beta,
        "clip_eps_lo": cfg.clip_eps_lo, "clip_eps_hi": cfg.clip_eps_hi,
        "temperature": cfg.temperature, "top_p": cfg.top_p, "top_k": cfg.top_k,
        "max_new_tokens": cfg.max_new_tokens, "prompts_per_step": cfg.prompts_per_step,
        "num_rollouts": cfg.num_rollouts, "train_batch_size": cfg.train_batch_size,
        "batch_acc": cfg.batch_acc, "data_size": cfg.data.size,
        "steps_per_epoch": steps_per_epoch, "num_steps": total_steps, "world_size": dist_env.world_size,
        "gradient_checkpointing": cfg.fit.gradient_checkpointing,
    })

    start_time = time.time()
    for step, batch in _pg.iter_training_batches(dataloader, sampler=train_sampler, num_steps=cfg.num_steps):
        step_start = time.time()

        # ---------- 1) rollouts (on-policy; each rank does its own prompt shard) ----------
        model.eval()
        if val_model:
            val_model.eval()
        replay_buffer.clear()
        rollout_rewards: list[torch.Tensor] = []
        rollout_tokens = 0.0
        rollout_samples = 0.0
        rollout_start = time.time()

        entries = [entry for entry in batch for _ in range(cfg.num_rollouts)]
        rollout_accuracy: list[torch.Tensor] = []
        rollout_format: list[torch.Tensor] = []
        with torch.no_grad():
            for rollout_batch in batched(entries, cfg.rollout_batch_size):
                sequence_ids, action_mask, attention_mask, rewards, _completions, accuracy, format_score = _pg.rollout(
                    model=model, entries=list(rollout_batch), dataset=dataset, tokenizer=tokenizer,
                    max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature,
                    top_p=cfg.top_p, top_k=cfg.top_k, min_p=cfg.min_p,
                )  # _pg.rollout unwraps DDP for .generate()
                rollout_rewards.append(rewards.detach().cpu())
                rollout_accuracy.append(accuracy.detach().cpu())
                rollout_format.append(format_score.detach().cpu())
                rollout_tokens += float(action_mask.sum().item())
                rollout_samples += float(action_mask.size(0))

                log_probs_old = _pg.compute_log_probs(model, sequence_ids, attention_mask)
                log_probs_ref = _pg.compute_log_probs(ref_model, sequence_ids, attention_mask)  # None if no ref
                values_old = _pg.compute_values(val_model, sequence_ids, attention_mask)         # None unless PPO
                rewards = _pg.apply_reward_kl(rewards, log_probs_old, log_probs_ref, action_mask,
                                              cfg.beta, cfg.outcome_loss)
                advantages = _pg.compute_advantages(rewards, cfg.outcome_loss, action_mask,
                                                    values_old, cfg.gamma, cfg.gae_lambda)  # GAE λ (PPO only)
                replay_buffer.add(
                    _pg.Experience(
                        sequence_ids=sequence_ids, attention_mask=attention_mask, action_mask=action_mask,
                        advantages=advantages, log_probs_old=log_probs_old,
                        log_probs_ref=log_probs_ref, values_old=values_old,
                    ).to(cpu)
                )

        avg_reward = float(torch.cat(rollout_rewards, dim=0).mean().item())
        avg_accuracy = float(torch.cat(rollout_accuracy, dim=0).mean().item())
        avg_format = float(torch.cat(rollout_format, dim=0).mean().item())
        rollout_time = max(time.time() - rollout_start, 1e-6)
        seq_len = rollout_tokens / max(rollout_samples, 1.0)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---------- 2) training updates on the replay buffer (DDP all-reduces grads on .backward()) ----------
        model.train()
        if val_model:
            val_model.train()
        experience_sampler = DataLoader(
            dataset=replay_buffer.buffer, batch_size=cfg.train_batch_size, shuffle=True,
            pin_memory=False, drop_last=True, collate_fn=_pg.join_experiences_batch,
        )
        loss_sum, grad_sum, n_updates, max_off_policy = 0.0, 0.0, 0, 0.0
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss, n_in_acc = 0.0, 0

        for batch_idx, experience in enumerate(experience_sampler):
            experience = experience.to(device)
            log_probs = _pg.compute_log_probs(model, experience.sequence_ids, experience.attention_mask)
            values = _pg.compute_values(val_model, experience.sequence_ids, experience.attention_mask)
            loss = objective(log_probs=log_probs, experience=experience, values=values)

            # how far the current policy has drifted from the rollout policy
            off_pol = _pg.approx_kl(log_probs, experience.log_probs_old, experience.action_mask)
            off_pol = _pg.masked_mean(off_pol, mask=experience.action_mask, dim=-1).max().item()
            max_off_policy = max(max_off_policy, off_pol)

            if not torch.isfinite(loss):
                continue
            (loss / cfg.batch_acc).backward()
            accumulated_loss += float(loss.item())
            n_in_acc += 1

            is_step = (batch_idx + 1) % cfg.batch_acc == 0 or (batch_idx + 1) == len(experience_sampler)
            if is_step:
                grad_norm = clip_grad_norm_(params, max_norm=cfg.max_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                loss_sum += accumulated_loss / max(n_in_acc, 1)
                grad_sum += float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
                n_updates += 1
                accumulated_loss, n_in_acc = 0.0, 0

        step_loss = loss_sum / n_updates if n_updates else float("nan")
        grad_norm_mean = grad_sum / n_updates if n_updates else float("nan")
        step_time = max(time.time() - step_start, 1e-6)
        hours = (time.time() - start_time) / 3600.0

        # ---------- 3) reduce across ranks + log (rank 0) ----------
        # (always call the collectives on every rank — conditionally skipping one would deadlock; a NaN
        #  loss/grad on any rank propagates through the all-reduce, which is what we want it to surface.)
        avg_reward = _pg.distributed_mean(avg_reward, device, dist_env)
        avg_accuracy = _pg.distributed_mean(avg_accuracy, device, dist_env)
        avg_format = _pg.distributed_mean(avg_format, device, dist_env)
        step_loss = _pg.distributed_mean(step_loss, device, dist_env)
        grad_norm_mean = _pg.distributed_mean(grad_norm_mean, device, dist_env)
        max_off_policy = _pg.distributed_max(max_off_policy, device, dist_env)
        global_rollout_tokens = _pg.distributed_sum(rollout_tokens, device, dist_env)
        toks_per_s = global_rollout_tokens / rollout_time
        metrics = {
            "reward": avg_reward,                   # = accuracy + 0.5·format  (what the optimizer sees)
            "reward/accuracy": avg_accuracy,        # mean reasoning_gym verifier score (0/1 per sample on gsm_symbolic)
            "reward/format": avg_format,            # mean count of {<think>,</think>,<answer>,</answer>} tags / 4
            "loss": step_loss,
            "grad_norm": grad_norm_mean,
            "off_policy/max_level": max_off_policy,
            "updates/optimizer_steps": float(n_updates),
            "throughput/tokens_per_sec": toks_per_s,
            "throughput/tokens_per_step": global_rollout_tokens,
            "seq/length_tokens_per_sample": seq_len,
            "time/step_seconds": step_time,
            "time/hours_elapsed": hours,
        }
        metrics.update(_pg.get_gpu_metrics())
        log.log(_pg.filter_numeric_metrics(metrics), step=step)
        if is_main:
            print(f"[harness] step {step + 1}/{total_steps}  reward={avg_reward:+.4f} (acc={avg_accuracy:.3f} fmt={avg_format:.3f})  "
                  f"loss={step_loss:.4f}  grad_norm={grad_norm_mean:.3f}  off_pol={max_off_policy:.4f}  "
                  f"{toks_per_s:.0f} tok/s  {step_time:.1f}s")
        _maybe_eval(cfg, model, tokenizer, eval_dataset, log, step, total_steps, is_main, ddp)

    if is_main:
        print(f"[harness] RL training complete ({total_steps} steps, {(time.time() - start_time) / 60:.1f} min).")
    _save_checkpoint(cfg, model, tokenizer, is_main)


def current_lam(cfg: ResearchConfig, step: int) -> float:
    """Effective λ at `step` (1-indexed) under `cfg.lam_schedule`.

    Schedules (default "const"):
      * "const"          — returns cfg.lam.
      * "step_off"       — cfg.lam if step <  cfg.lam_step, else 0.
      * "step_on"        — 0       if step <  cfg.lam_step, else cfg.lam.
      * "linear_anneal"  — cfg.lam at step≤cfg.lam_step; linearly to 0 at step≥cfg.lam_step_end.

    Designed for the collapse-recovery intervention (RESULTS.md §7.5 bullet 2):
    OPD-interior runs hit a sharp accuracy collapse at step ~100-150 from which only some seeds
    recover. step_off / step_on / linear_anneal each remove or delay the teacher's reverse-KL
    signal across the collapse window to test whether the signal is helpful as warm-up, harmful
    during collapse, or helpful as refinement.
    """
    sched = cfg.lam_schedule
    if sched == "const":
        return float(cfg.lam)
    if sched == "step_off":
        return float(cfg.lam) if step < cfg.lam_step else 0.0
    if sched == "step_on":
        return 0.0 if step < cfg.lam_step else float(cfg.lam)
    if sched == "linear_anneal":
        if step <= cfg.lam_step:
            return float(cfg.lam)
        if step >= cfg.lam_step_end:
            return 0.0
        frac = (step - cfg.lam_step) / float(cfg.lam_step_end - cfg.lam_step)
        return float(cfg.lam) * (1.0 - frac)
    raise ValueError(f"unknown lam_schedule: {sched!r}")


# --- L3 off-policy reverse-KD: build the experience buffer from TEACHER-sampled states ----------
# (§8.1 follow-up). The on-policy distill loop fills its buffer with student.generate() rollouts;
# these helpers instead fill it with the teacher's OWN rollouts (a harness.rft_generate JSONL), so
# the per-token reverse-KL loss lands on teacher-visited states. Everything downstream (teacher
# forward, student forward, reverse_kl_distill_advantage, optimizer) is byte-identical to OPD — only
# the source of the sequences differs. This isolates on-policy-ness alone (the L2 confound §8.1).

def _load_teacher_rollouts(path: str) -> list[dict]:
    """Load a harness.rft_generate JSONL of teacher-sampled rows (templated_prompt / completion /
    answer / accuracy / format). One dict per accepted (or, with --keep_all, every) completion."""
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no teacher rollouts found in {path!r}")
    return rows


def _offpolicy_step_chunks(rows: list[dict], seqs_per_step: int, num_steps: int, seed: int):
    """Yield one list of `seqs_per_step` teacher-rollout rows per training step, drawn by cycling a
    seeded shuffle of `rows` (reshuffle on wrap-around). Deterministic given (rows, seqs_per_step, seed)
    — matches the on-policy loop's sequences-per-step so #optimizer-steps and #sequences/step are equal."""
    import random
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    pos = 0
    for _ in range(num_steps):
        idx: list[int] = []
        while len(idx) < seqs_per_step:
            if pos >= len(order):
                rng.shuffle(order)
                pos = 0
            idx.append(order[pos])
            pos += 1
        yield [rows[i] for i in idx]


def _left_pad(seqs: list[torch.Tensor], pad_value, dtype) -> torch.Tensor:
    """Left-pad a list of 1-D tensors to a common length (matches buffer.pad_sequences how='start')."""
    maxlen = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), maxlen), pad_value, dtype=dtype)
    for i, s in enumerate(seqs):
        out[i, maxlen - s.size(0):] = s.to(dtype)
    return out


def _fill_offpolicy_buffer(cfg, rows, tokenizer, model, ref_model, device, cpu, replay_buffer):
    """Tokenize a step's worth of teacher rollouts, build (sequence_ids, attention_mask, action_mask)
    with the EXACT convention `_pg.rollout` produces (action_mask[:, j] flags predicting token j+1;
    completion targets are positions [P-1, S-2]), run the student (and optional ref) forward for
    log_probs_old, and add Experiences to `replay_buffer`. advantages are zeroed (unused at lam=1);
    teacher_logprobs left None so the same_family teacher recomputes per micro-batch — identical to OPD.

    Returns (avg_reward, avg_accuracy, avg_format, total_action_tokens, n_samples). Reward/accuracy/
    format are the teacher's own verifier scores from the JSONL (rollout-time diagnostics; not optimized)."""
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    max_length = cfg.max_new_tokens + 1024  # prompt + completion guard (gsm prompts are short)

    seqs: list[torch.Tensor] = []
    ams: list[torch.Tensor] = []
    acc_sum = fmt_sum = 0.0
    for row in rows:
        prompt_ids = tokenizer(row["templated_prompt"], add_special_tokens=False)["input_ids"]
        comp_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"] + [eos_id]
        if len(prompt_ids) + len(comp_ids) > max_length:
            # keep the completion intact; truncate the prompt from the left (mirror train_sft_rft)
            if len(comp_ids) >= max_length:
                comp_ids = comp_ids[:max_length]
                prompt_ids = []
            else:
                prompt_ids = prompt_ids[len(prompt_ids) + len(comp_ids) - max_length:]
        input_ids = prompt_ids + comp_ids
        P, S = len(prompt_ids), len(prompt_ids) + len(comp_ids)
        am = torch.zeros(S - 1, dtype=torch.bool)
        am[max(P - 1, 0):] = True  # targets predicting the completion tokens
        seqs.append(torch.tensor(input_ids, dtype=torch.long))
        ams.append(am)
        acc_sum += float(row.get("accuracy", 0.0) or 0.0)
        fmt_sum += float(row.get("format", 0.0) or 0.0)

    total_action_tokens = 0.0
    n_samples = float(len(seqs))
    # batch in chunks of rollout_batch_size for the student/ref forward (memory); grouping is irrelevant
    # at lam=1 (no outcome branch), so chunks need not align to prompts.
    for start in range(0, len(seqs), cfg.rollout_batch_size):
        s_chunk = seqs[start:start + cfg.rollout_batch_size]
        a_chunk = ams[start:start + cfg.rollout_batch_size]
        seq_b = _left_pad(s_chunk, pad_id, torch.long).to(device)
        attn_b = _left_pad([torch.ones(s.size(0), dtype=torch.long) for s in s_chunk], 0, torch.long).to(device)
        am_b = _left_pad(a_chunk, 0, torch.bool).to(device)
        with torch.no_grad():
            log_probs_old = _pg.compute_log_probs(model, seq_b, attn_b)           # (b, maxS-1)
            log_probs_ref = _pg.compute_log_probs(ref_model, seq_b, attn_b)       # None if beta == 0
        advantages = torch.zeros_like(log_probs_old)                              # unused at lam=1; schema parity
        total_action_tokens += float(am_b.sum().item())
        replay_buffer.add(
            _pg.Experience(
                sequence_ids=seq_b, attention_mask=attn_b, action_mask=am_b,
                advantages=advantages, log_probs_old=log_probs_old,
                log_probs_ref=log_probs_ref, values_old=None, teacher_logprobs=None,
            ).to(cpu)
        )

    n = max(len(rows), 1)
    avg_accuracy = acc_sum / n
    avg_format = fmt_sum / n
    avg_reward = avg_accuracy + 0.5 * avg_format
    return avg_reward, avg_accuracy, avg_format, total_action_tokens, n_samples


def _run_distill_loop(cfg: ResearchConfig, log: Logger, dist_env=None) -> None:
    """The lam > 0 path: per-token teacher reverse-KL term. Single GPU. (alpha == 1.)

    Same rollout / experience / training-update machinery as `_run_rl_loop`, but each training batch
    also runs a *frozen teacher* forward when `lam_eff > 0` to get
    log pi_T(y_hat_t | y_hat_<t), then blends that teacher loss with the configured
    `policy_gradients.loss.*` outcome objective according to the effective lambda schedule:

        A_t  = lam * (log pi_T - log pi_theta) + (1 - lam) * A^outcome_t
        loss = - mean_t  A_t * log pi_theta(y_hat_t | y_hat_<t)

      * lam == 1, teacher=`same_family`           -> plain OPD          (Lu & Thinking Machines 2025)
      * lam == 1, teacher=`self` + answer + clip  -> OPSD              (Zhao et al. 2026)  [conditioning stubbed]
      * 0 < lam < 1                                -> expert RL + OPD   (../expert-rl-plus-opd.md)
        — the outcome branch goes through the proper clipped objective (e.g. GRPO) and is blended
        at the loss level with the teacher REINFORCE term.

    The teacher lives on `cfg.teacher.device_id` (default 0 = the student's card): 1B student full-FT
    + 7B bf16 frozen teacher ~30 GB, fits an 80 GB H100. Teacher log-probs are *recomputed* per
    training batch for stateless teachers (the teacher is frozen -> deterministic; a 7B forward over
    ~prompt+completion tokens is cheap). Entries-aware teachers cache log-probs at rollout time.

    NOTE: this duplicates a fair amount of `_run_rl_loop`'s setup / rollout / metrics scaffolding —
    a future refactor should factor the shared parts into helpers parametrized by (teacher, loss_fn).
    Multi-GPU/DDP is not wired here yet (the frozen teacher would be replicated per rank — fine on
    memory — but the DataLoader sharding / grad sync / metric reduction need the same treatment as
    `_run_rl_loop` got); run NPROC=1 for the lam>0 path.
    """
    if dist_env is None:
        dist_env = _pg.DistEnv()
    if dist_env.enabled:
        raise NotImplementedError(
            "Multi-GPU/DDP for the lam>0 (OPD / OPSD / expert-RL+OPD) path is not wired yet — run it "
            "single-GPU (NPROC=1). DDP works for the RL corner (lam=0) — see `_run_rl_loop` / `run_h100_ddp.sh`."
        )
    cpu = torch.device("cpu")
    device = torch.device(f"cuda:{cfg.model_device_id}" if torch.cuda.is_available() else "cpu")

    # ---- student, tokenizer, dataset, optimizer ----
    model, tokenizer = _load_student(cfg, device)
    if getattr(tokenizer, "chat_template", None) is None:
        raise RuntimeError(
            f"Tokenizer for {cfg.model_name!r} has no chat_template — the reasoning_gym rollout uses "
            "tokenizer.apply_chat_template(...). Use an instruction-tuned checkpoint (e.g. "
            "allenai/OLMo-2-0425-1B-Instruct)."
        )
    dataset = _pg.create_dataset(cfg)
    dataloader = DataLoader(
        dataset=dataset, batch_size=cfg.prompts_per_step, shuffle=True,
        pin_memory=False, drop_last=True, collate_fn=lambda x: x,
    )
    steps_per_epoch = len(dataloader)
    if steps_per_epoch == 0:
        raise ValueError("Dataloader has 0 batches — increase data.size or lower prompts_per_step.")
    total_steps = cfg.num_steps if cfg.num_steps is not None else steps_per_epoch

    ref_model = _pg.get_ref_model(cfg.model_name, device, cfg.beta)   # None unless beta > 0 (optional KL-to-base on the student)
    params = list(model.parameters())
    optimizer = _build_optimizer(cfg, params)
    replay_buffer = _pg.ReplayBuffer()
    eval_dataset = _make_eval_dataset(cfg)   # None unless cfg.eval_every > 0

    # ---- teacher pi_T + outcome objective ----
    teacher = build_teacher(cfg.teacher, student_model_name=cfg.model_name, student_model=model)
    if hasattr(teacher, "_ensure_model"):
        teacher._ensure_model()  # load eagerly so a missing snapshot / OOM surfaces now, not mid-step
    # variant (b): a separately-trained step-level PRM provides the per-token importance signal
    # (replacing variant (c)'s self-referential answer_info_gain). The PRM is a frozen 1B-SFT + scalar
    # head; the teacher's reverse-KL logits still come from the 7B-SFT answer-conditioned teacher.
    prm_teacher = None
    if cfg.prm_reweight and cfg.prm_source == "trained":
        from .config import TeacherSpec as _TS
        prm_spec = _TS(kind="prm_trained", model_name=cfg.prm_model_path, device_id=cfg.teacher.device_id)
        prm_teacher = PRMTeacher(prm_spec, student_model_name=cfg.model_name)
        prm_teacher._ensure_model()  # load eagerly so a missing checkpoint / OOM surfaces now
        print(f"[harness] PRM teacher (variant b): loaded trained PRM from {cfg.prm_model_path}")
    # Build the proper clipped outcome objective (e.g. GRPO) whenever the configured run can have
    # lam_t < 1. This includes schedules such as step_off/step_on starting from cfg.lam == 1.0.
    needs_outcome_objective = cfg.lam < 1.0 or cfg.lam_schedule != "const"
    if needs_outcome_objective:
        if cfg.outcome_loss == "ppo":
            raise NotImplementedError(
                "outcome_loss='ppo' is not supported in the distill path (no value model in _run_distill_loop). "
                "Use a group-relative objective (grpo/drgrpo/gspo/cispo/rloo), or use const lam=1."
            )
        outcome_objective = _pg.get_loss_objective(
            loss=cfg.outcome_loss,
            clip_eps_lo=cfg.clip_eps_lo, clip_eps_hi=cfg.clip_eps_hi,
            clip_eps_val=cfg.clip_eps_val, vf_coef=cfg.vf_coef, beta=cfg.beta,
        ).to(device)
    else:
        outcome_objective = None

    teacher_name = cfg.teacher.model_name or cfg.model_name
    print(f"[harness] distill corner: recipe={cfg.recipe} lam={cfg.lam} teacher={cfg.teacher.kind}({teacher_name}) "
          f"per_token_kl_clip={cfg.per_token_kl_clip} model={cfg.model_name} attn={_pg.get_attn_implementation()} "
          f"device={device} steps={total_steps} prompts/step={cfg.prompts_per_step} rollouts/prompt={cfg.num_rollouts} "
          f"beta={cfg.beta} outcome_loss={cfg.outcome_loss}"
          f"{' (unused while lam_eff=1)' if not needs_outcome_objective else ' (clipped, blended at loss level)'}")
    log.log_params({
        "recipe": cfg.recipe, "alpha": cfg.alpha, "lam": cfg.lam,
        "teacher_kind": cfg.teacher.kind, "teacher_model": teacher_name,
        "teacher_condition_on": cfg.teacher.condition_on, "per_token_kl_clip": cfg.per_token_kl_clip,
        "model_name": cfg.model_name, "lr": cfg.lr, "beta": cfg.beta, "outcome_loss": cfg.outcome_loss,
        "temperature": cfg.temperature, "top_p": cfg.top_p, "top_k": cfg.top_k,
        "max_new_tokens": cfg.max_new_tokens, "prompts_per_step": cfg.prompts_per_step,
        "num_rollouts": cfg.num_rollouts, "train_batch_size": cfg.train_batch_size,
        "batch_acc": cfg.batch_acc, "data_size": cfg.data.size,
        "steps_per_epoch": steps_per_epoch, "num_steps": total_steps,
        "gradient_checkpointing": cfg.fit.gradient_checkpointing,
        "offpolicy_teacher_states": cfg.offpolicy_teacher_states,
        "prm_reweight": cfg.prm_reweight, "prm_source": cfg.prm_source,
        "prm_weight_fn": cfg.prm_weight_fn, "prm_temperature": cfg.prm_temperature,
        "prm_weight_ceiling": cfg.prm_weight_ceiling,
    })
    if cfg.prm_reweight:
        variant_label = "variant (b) trained-PRM" if cfg.prm_source == "trained" else "variant (c) self-referential"
        print(f"[harness] PRM-reweighted OPSD ({variant_label}): source={cfg.prm_source} fn={cfg.prm_weight_fn} "
              f"temp={cfg.prm_temperature} ceil={cfg.prm_weight_ceiling} clip={cfg.per_token_kl_clip} "
              f"— reweighting the teacher reverse-KL by per-token process-importance (mass-preserving).")

    # L3 off-policy reverse-KD (§8.1 follow-up): if a teacher-rollout JSONL is configured, the buffer
    # is filled from the teacher's OWN sequences (teacher-sampled states) each step instead of
    # student.generate() — the only difference vs on-policy OPD. Both arms share teacher + reverse-KL
    # objective + steps/LR; this isolates on-policy-ness alone.
    offpolicy_chunks = None
    if cfg.offpolicy_teacher_states:
        offpolicy_rows = _load_teacher_rollouts(cfg.offpolicy_teacher_states)
        seqs_per_step = cfg.prompts_per_step * cfg.num_rollouts
        offpolicy_chunks = _offpolicy_step_chunks(offpolicy_rows, seqs_per_step, total_steps, cfg.seed)
        epochs = total_steps * seqs_per_step / max(len(offpolicy_rows), 1)
        print(f"[harness] OFF-POLICY reverse-KD: {len(offpolicy_rows)} teacher rollouts from "
              f"{cfg.offpolicy_teacher_states} -> {seqs_per_step} seqs/step x {total_steps} steps "
              f"(~{epochs:.1f} epochs); student.generate() SKIPPED, reverse-KL lands on teacher states.")

    start_time = time.time()
    for step, batch in _pg.iter_training_batches(dataloader, sampler=None, num_steps=cfg.num_steps):
        step_start = time.time()

        # ---------- 1) experience buffer: on-policy student rollouts OR off-policy teacher states ----------
        model.eval()
        replay_buffer.clear()
        rollout_tokens = 0.0
        rollout_samples = 0.0
        rollout_start = time.time()

        if offpolicy_chunks is not None:
            # L3 off-policy reverse-KD: fill the buffer from the TEACHER's own rollouts (teacher-sampled
            # states). No student.generate(); the reverse-KL loss built in step 2 lands on teacher
            # trajectories. lam==1 here (validated), so there is no outcome branch and no grouping invariant.
            chunk = next(offpolicy_chunks)
            avg_reward, avg_accuracy, avg_format, rollout_tokens, rollout_samples = _fill_offpolicy_buffer(
                cfg, chunk, tokenizer, model, ref_model, device, cpu, replay_buffer,
            )
        else:
            # on-policy: the student generates, the teacher scores its trajectories (OPD / OPSD / expert-RL+OPD).
            rollout_rewards: list[torch.Tensor] = []
            # NOTE on the GRPO/RLOO grouping invariant: group-relative advantage objectives normalize over
            # dim 0 of `rewards` (= a rollout_batch), assuming each rollout_batch contains exactly one
            # prompt's `num_rollouts` rollouts. The loop below builds `entries` as prompts × num_rollouts
            # interleaved, then batches in chunks of `rollout_batch_size`, so the assumption holds iff
            # `rollout_batch_size == num_rollouts`. The base `Config.validate_rollout_batch_size`
            # (policy_gradients/config.py) enforces this whenever num_rollouts > 1.
            # See the GRPO/RLOO grouping note in `_run_rl_loop` — same invariant applies here for the (1-λ)
            # outcome branch (advantage normalization happens within a rollout_batch = one prompt's group).
            entries = [entry for entry in batch for _ in range(cfg.num_rollouts)]
            rollout_accuracy: list[torch.Tensor] = []
            rollout_format: list[torch.Tensor] = []
            with torch.no_grad():
                for rollout_batch in batched(entries, cfg.rollout_batch_size):
                    sequence_ids, action_mask, attention_mask, rewards, _completions, accuracy, format_score = _pg.rollout(
                        model=model, entries=list(rollout_batch), dataset=dataset, tokenizer=tokenizer,
                        max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature,
                        top_p=cfg.top_p, top_k=cfg.top_k, min_p=cfg.min_p,
                    )
                    rollout_rewards.append(rewards.detach().cpu())
                    rollout_accuracy.append(accuracy.detach().cpu())
                    rollout_format.append(format_score.detach().cpu())
                    rollout_tokens += float(action_mask.sum().item())
                    rollout_samples += float(action_mask.size(0))

                    log_probs_old = _pg.compute_log_probs(model, sequence_ids, attention_mask)
                    log_probs_ref = _pg.compute_log_probs(ref_model, sequence_ids, attention_mask)  # None if beta == 0
                    rewards = _pg.apply_reward_kl(rewards, log_probs_old, log_probs_ref, action_mask,
                                                  cfg.beta, cfg.outcome_loss)
                    advantages = _pg.compute_advantages(rewards, cfg.outcome_loss, action_mask,
                                                        None, cfg.gamma, cfg.gae_lambda)  # zero-weighted at lam=1
                    # PRM-as-teacher / OPSD: when the teacher needs entries-aware conditioning, compute its
                    # log-probs now (entries are in scope) and cache in Experience. Skipped for stateless teachers
                    # (plain SameFamilyTeacher's _build_inputs is a no-op; the training loop will recompute then).
                    # For entries-aware teachers this is also strictly cheaper than the previous per-microbatch
                    # teacher forward — frozen teacher is deterministic, so caching = correct.
                    teacher_logprobs_cached = None
                    prm_weights_cached = None
                    if _teacher_needs_entries(teacher):
                        if cfg.prm_reweight:
                            # PRM-reweighted OPSD. variant (c): self-referential answer-info-gain from the
                            # same 7B teacher (one extra no-answer forward). variant (b): a separately-
                            # trained PRM scores process correctness (one forward over the student's
                            # trajectory). Both cache teacher_logprobs (the reverse-KL logits, from the
                            # 7B-SFT answer-conditioned teacher) + prm_weights so the training inner loop
                            # pays nothing extra.
                            teacher_logprobs_cached = teacher.token_logprobs(
                                sequence_ids, attention_mask, action_mask,
                                entries=list(rollout_batch), student_logprobs=log_probs_old,
                            )
                            if cfg.prm_source == "trained" and prm_teacher is not None:
                                # variant (b): trained PRM provides the per-token importance signal.
                                g_t = prm_teacher.step_scores(
                                    sequence_ids, attention_mask, action_mask, entries=list(rollout_batch),
                                )
                            else:
                                # variant (c): self-referential answer-info-gain (one extra no-answer forward).
                                _lp_answer, g_t = teacher.answer_info_gain(
                                    sequence_ids, attention_mask, action_mask, entries=list(rollout_batch),
                                )
                                # teacher_logprobs_cached is lp_answer (the answer-conditioned teacher term).
                                teacher_logprobs_cached = _lp_answer
                            prm_weights_cached = prm_importance_weights(
                                g_t, action_mask, fn=cfg.prm_weight_fn,
                                temp=cfg.prm_temperature, ceiling=cfg.prm_weight_ceiling,
                            )
                        else:
                            teacher_logprobs_cached = teacher.token_logprobs(
                                sequence_ids, attention_mask, action_mask,
                                entries=list(rollout_batch), student_logprobs=log_probs_old,
                            )
                    replay_buffer.add(
                        _pg.Experience(
                            sequence_ids=sequence_ids, attention_mask=attention_mask, action_mask=action_mask,
                            advantages=advantages, log_probs_old=log_probs_old,
                            log_probs_ref=log_probs_ref, values_old=None,
                            teacher_logprobs=teacher_logprobs_cached,
                            prm_weights=prm_weights_cached,
                        ).to(cpu)
                    )

            avg_reward = float(torch.cat(rollout_rewards, dim=0).mean().item())
            avg_accuracy = float(torch.cat(rollout_accuracy, dim=0).mean().item())
            avg_format = float(torch.cat(rollout_format, dim=0).mean().item())

        rollout_time = max(time.time() - rollout_start, 1e-6)
        toks_per_s = rollout_tokens / rollout_time
        seq_len = rollout_tokens / max(rollout_samples, 1.0)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---------- 2) training updates: student fwd + frozen-teacher/outcome losses ----------
        # `lam_t`: effective λ at this step under cfg.lam_schedule (see `current_lam` for the shapes).
        # Default "const" → cfg.lam every step; the OPD collapse-recovery experiment (§7.5) uses
        # step_off / step_on / linear_anneal to test which window of training the teacher term helps in.
        lam_t = current_lam(cfg, step + 1)   # step is 0-indexed; log uses step+1
        model.train()
        experience_sampler = DataLoader(
            dataset=replay_buffer.buffer, batch_size=cfg.train_batch_size, shuffle=True,
            pin_memory=False, drop_last=True, collate_fn=_pg.join_experiences_batch,
        )
        loss_sum, grad_sum, n_updates, max_off_policy = 0.0, 0.0, 0, 0.0
        loss_teacher_sum, loss_outcome_sum = 0.0, 0.0
        loss_micro_n = 0
        rkl_sum, rkl_n = 0.0, 0
        # Per-token KL-signal distribution (teacher_logprobs - log_probs_old, masked to action tokens).
        # Matches the sign convention of `reverse_kl_distill_advantage`. Quantiles are computed per
        # micro-batch and averaged across the experience sampler — cheap diagnostic for §8.2 plot.
        kl_p50_sum = kl_p90_sum = kl_p99_sum = kl_max_sum = kl_heavy_sum = 0.0
        n_kl_micro = 0
        # PRM-reweighted OPSD (variant c) weight distribution: mean should track 1 (mass-preserving);
        # max/p99 detect a softmax blowup that would re-introduce the very outlier problem clipping solves.
        prm_w_mean_sum = prm_w_max_sum = prm_w_p99_sum = 0.0
        n_prm_micro = 0
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss, n_in_acc = 0.0, 0

        for batch_idx, experience in enumerate(experience_sampler):
            experience = experience.to(device)
            log_probs = _pg.compute_log_probs(model, experience.sequence_ids, experience.attention_mask)  # requires grad
            teacher_logprobs = None
            loss_teacher = None
            loss_outcome = None

            if lam_t > 0.0:
                with torch.no_grad():
                    if experience.teacher_logprobs is not None:
                        # Entries-aware teacher (OPSD / PRM-as-teacher) — pre-computed at rollout time.
                        teacher_logprobs = experience.teacher_logprobs
                    else:
                        teacher_logprobs = teacher.token_logprobs(
                            experience.sequence_ids, experience.attention_mask, experience.action_mask,
                            entries=None, student_logprobs=experience.log_probs_old,
                        )
                # Teacher branch is detached advantage times log pi_theta. PRM-reweighted OPSD (variant c)
                # multiplies the (optionally clipped) per-token KL by the cached importance weight w_t;
                # experience.prm_weights is None unless cfg.prm_reweight, so plain OPD/OPSD is unaffected.
                a_teacher = reverse_kl_distill_advantage(
                    log_probs, teacher_logprobs, experience.action_mask, clip=cfg.per_token_kl_clip,
                    prm_weights=experience.prm_weights,
                )
                loss_teacher = -(a_teacher * log_probs)
                loss_teacher = _pg.masked_mean(loss_teacher, mask=experience.action_mask, dim=-1).mean(dim=0)
                loss_teacher_sum += float(loss_teacher.item())

            if lam_t < 1.0:
                if outcome_objective is None:
                    raise RuntimeError("lam_eff < 1 requires an outcome objective; check lam_schedule/outcome_loss setup.")
                loss_outcome = outcome_objective(
                    log_probs=log_probs, experience=experience, values=None,
                )
                loss_outcome_sum += float(loss_outcome.item())

            if lam_t <= 0.0:
                loss = loss_outcome
            elif lam_t >= 1.0:
                loss = loss_teacher
            else:
                loss = lam_t * loss_teacher + (1.0 - lam_t) * loss_outcome

            with torch.no_grad():
                if teacher_logprobs is not None:
                    # reverse-KL still between the rollout policy and the teacher, over generated tokens:
                    rkl = _pg.masked_mean(experience.log_probs_old - teacher_logprobs,
                                          mask=experience.action_mask, dim=-1).mean().item()
                    rkl_sum += float(rkl); rkl_n += 1
                off_pol = _pg.approx_kl(log_probs, experience.log_probs_old, experience.action_mask)
                off_pol = _pg.masked_mean(off_pol, mask=experience.action_mask, dim=-1).max().item()
                max_off_policy = max(max_off_policy, off_pol)
                # Per-token KL-signal distribution (sign-aligned with the teacher branch):
                if teacher_logprobs is not None:
                    kl_signal = (teacher_logprobs - experience.log_probs_old)[experience.action_mask.bool()]
                else:
                    kl_signal = torch.empty(0, device=log_probs.device)
                if kl_signal.numel() > 0:
                    abs_kl = kl_signal.abs()
                    qs = torch.quantile(
                        kl_signal.float(),
                        torch.tensor([0.5, 0.9, 0.99], device=kl_signal.device, dtype=torch.float32),
                    )
                    stats = torch.stack([qs[0], qs[1], qs[2], abs_kl.max(), (abs_kl > 5.0).float().mean()]).tolist()
                    kl_p50_sum += stats[0]; kl_p90_sum += stats[1]; kl_p99_sum += stats[2]
                    kl_max_sum += stats[3]; kl_heavy_sum += stats[4]
                    n_kl_micro += 1
                # PRM-reweight weight distribution (variant c) over action tokens (None unless cfg.prm_reweight):
                if experience.prm_weights is not None:
                    pw = experience.prm_weights[experience.action_mask.bool()]
                    if pw.numel() > 0:
                        prm_w_mean_sum += float(pw.float().mean().item())
                        prm_w_max_sum += float(pw.float().max().item())
                        prm_w_p99_sum += float(torch.quantile(pw.float(), 0.99).item())
                        n_prm_micro += 1

            if not torch.isfinite(loss):
                continue
            (loss / cfg.batch_acc).backward()
            accumulated_loss += float(loss.item())
            n_in_acc += 1
            loss_micro_n += 1

            is_step = (batch_idx + 1) % cfg.batch_acc == 0 or (batch_idx + 1) == len(experience_sampler)
            if is_step:
                grad_norm = clip_grad_norm_(params, max_norm=cfg.max_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                loss_sum += accumulated_loss / max(n_in_acc, 1)
                grad_sum += float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
                n_updates += 1
                accumulated_loss, n_in_acc = 0.0, 0

        step_loss = loss_sum / n_updates if n_updates else float("nan")
        grad_norm_mean = grad_sum / n_updates if n_updates else float("nan")
        reverse_kl_mean = rkl_sum / rkl_n if rkl_n else float("nan")
        # per-micro-batch (n_in_acc * n_updates ~ len(experience_sampler)) component averages — useful
        # for diagnosing the lam-interior: which branch (teacher vs outcome) is moving the loss.
        n_micro = max(loss_micro_n, 1)
        loss_teacher_mean = loss_teacher_sum / n_micro if n_micro else float("nan")
        loss_outcome_mean = (loss_outcome_sum / n_micro) if (needs_outcome_objective and n_micro) else float("nan")
        kl_p50_mean = kl_p50_sum / n_kl_micro if n_kl_micro else float("nan")
        kl_p90_mean = kl_p90_sum / n_kl_micro if n_kl_micro else float("nan")
        kl_p99_mean = kl_p99_sum / n_kl_micro if n_kl_micro else float("nan")
        kl_max_mean = kl_max_sum / n_kl_micro if n_kl_micro else float("nan")
        kl_heavy_mean = kl_heavy_sum / n_kl_micro if n_kl_micro else float("nan")
        prm_w_mean = prm_w_mean_sum / n_prm_micro if n_prm_micro else float("nan")
        prm_w_max_mean = prm_w_max_sum / n_prm_micro if n_prm_micro else float("nan")
        prm_w_p99_mean = prm_w_p99_sum / n_prm_micro if n_prm_micro else float("nan")
        step_time = max(time.time() - step_start, 1e-6)
        hours = (time.time() - start_time) / 3600.0

        metrics = {
            "reward": avg_reward,                   # = accuracy + 0.5·format  (rollout-time; not directly optimized at lam=1)
            "reward/accuracy": avg_accuracy,        # mean reasoning_gym verifier score (0/1 per sample on gsm_symbolic)
            "reward/format": avg_format,            # mean count of {<think>,</think>,<answer>,</answer>} tags / 4
            "loss": step_loss,
            "loss/teacher_term": loss_teacher_mean,           # mean_t [-A^teacher · log π_θ]  (not lam-scaled)
            "loss/outcome_term": loss_outcome_mean,           # clipped outcome objective (e.g. GRPO); NaN at lam=1
            "grad_norm": grad_norm_mean,
            "teacher/reverse_kl": reverse_kl_mean,  # mean_t (log pi_rollout - log pi_T) over generated tokens
            # Per-token KL-signal distribution (teacher - rollout-student, sign-aligned with the teacher term):
            "kl_signal/p50": kl_p50_mean,
            "kl_signal/p90": kl_p90_mean,
            "kl_signal/p99": kl_p99_mean,
            "kl_signal/abs_max": kl_max_mean,
            "kl_signal/heavy_tail_frac": kl_heavy_mean,
            # PRM-reweighted OPSD (variant c) weight distribution (NaN unless cfg.prm_reweight):
            "prm/weight_mean": prm_w_mean,          # sanity: mass-preserving -> ~1.0
            "prm/weight_max": prm_w_max_mean,       # softmax blowup detector (a new outlier == clip re-needed)
            "prm/weight_p99": prm_w_p99_mean,
            "meta/lam_eff": lam_t,                  # effective λ at this step (cfg.lam_schedule-aware)
            "off_policy/max_level": max_off_policy,
            "updates/optimizer_steps": float(n_updates),
            "throughput/tokens_per_sec": toks_per_s,
            "throughput/tokens_per_step": rollout_tokens,
            "seq/length_tokens_per_sample": seq_len,
            "time/step_seconds": step_time,
            "time/hours_elapsed": hours,
        }
        metrics.update(_pg.get_gpu_metrics())
        log.log(_pg.filter_numeric_metrics(metrics), step=step)
        print(f"[harness] step {step + 1}/{total_steps}  reward={avg_reward:+.4f} (acc={avg_accuracy:.3f} fmt={avg_format:.3f})  "
              f"loss={step_loss:.4f}  rev_kl={reverse_kl_mean:+.4f}  grad_norm={grad_norm_mean:.3f}  "
              f"off_pol={max_off_policy:.4f}  {toks_per_s:.0f} tok/s  {step_time:.1f}s")
        _maybe_eval(cfg, model, tokenizer, eval_dataset, log, step, total_steps, is_main=True, ddp=False)

    print(f"[harness] distill training complete ({total_steps} steps, {(time.time() - start_time) / 60:.1f} min).")
    # Free the (large, e.g. 13B) teacher before the CPU state-dict gather so the single-file
    # safetensors write can't get OOM-killed mid-stream (cf. exp7c arm D 115138: 14.6GB save
    # truncated to 11.2GB -> "file not fully covered"). Teacher is unused past this point.
    import gc
    try:
        del teacher
    except NameError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _save_checkpoint(cfg, model, tokenizer, is_main=True)


def train_policy_gradient(cfg: ResearchConfig, log: Logger) -> None:
    """The on-policy branch (alpha == 1): RL corner (lam == 0, `_run_rl_loop`) or the teacher
    reverse-KL path (lam > 0, `_run_distill_loop` — OPD / OPSD / expert-RL+OPD)."""
    if 0.0 < cfg.alpha < 1.0:
        raise NotImplementedError(
            "Off-policy mixing (0 < alpha < 1) not implemented — needs an off-policy buffer of size "
            "cfg.offpolicy_buffer_size + a PPO-style IS correction (is_clip_lo/hi). See ../meta-algorithm-alpha-lambda.md."
        )

    dist_env = _pg.setup_distributed()  # single-process: enabled=False, is_main_process=True; torchrun: NCCL up, device set
    # Distinct RNG per rank so each rank's rollouts (temperature sampling) and data shuffle differ; DDP
    # broadcasts rank-0's model init on construction, so different init seeds across ranks are harmless.
    _pg.seed_everything(cfg.seed + dist_env.rank)
    try:
        if cfg.lam == 0.0:
            _run_rl_loop(cfg, log, dist_env)         # RL corner — DDP-aware
        else:
            _run_distill_loop(cfg, log, dist_env)    # lam>0 — single-GPU only (raises under torchrun)
    finally:
        _pg.cleanup_distributed(dist_env)


def main(cfg: ResearchConfig) -> None:
    is_main = int(os.environ.get("RANK", "0")) == 0
    log = _build_logger(cfg, is_main=is_main)
    try:
        if cfg.recipe == "sft" or (cfg.alpha == 0.0 and cfg.teacher.kind == "dataset"):
            train_sft(cfg, log)
        else:
            train_policy_gradient(cfg, log)
    finally:
        log.finish()


def _parse_scalar(s: str):
    """Best-effort parse of a CLI override value: bool -> null -> int -> float -> str.

    Only `null` / `~` map to None — NOT the string `"none"`, which is a legitimate value
    (e.g. `--set teacher.kind=none` must stay the string "none", not become Python None).
    """
    low = s.strip().lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "~"}:
        return None
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def main_cli() -> None:
    import yaml  # local import: only main_cli needs it

    ap = argparse.ArgumentParser(description="Unified (alpha, lambda, pi_T) post-training trainer.")
    ap.add_argument("--config", required=True, help="path to a YAML ResearchConfig (see harness/configs/)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config key, e.g. --set model_name=allenai/OLMo-2-0425-1B-Instruct "
                         "--set num_steps=5 --set teacher.model_name=allenai/OLMo-2-1124-7B-SFT "
                         "--set teacher.kind=none (repeatable; dotted keys nest into sub-objects like teacher/data)")
    args = ap.parse_args()
    with open(args.config) as f:
        raw = yaml.safe_load(f) or {}
    for item in args.set:
        if "=" not in item:
            ap.error(f"--set expects KEY=VALUE, got {item!r}")
        key, val = item.split("=", 1)
        parts = [p.strip() for p in key.strip().split(".")]
        node = raw
        for p in parts[:-1]:                       # dotted keys (e.g. teacher.model_name) nest into sub-dicts
            if not isinstance(node.get(p), dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = _parse_scalar(val)
    cfg = ResearchConfig(**raw)
    print(f"[harness] config: recipe={cfg.recipe} alpha={cfg.alpha} lam={cfg.lam} "
          f"teacher={cfg.teacher.kind} model={cfg.model_name} loss={cfg.outcome_loss}")
    main(cfg)


if __name__ == "__main__":
    main_cli()
