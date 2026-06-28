"""Minimal SFT loop for the §7.2 positive control: train an OLMo-2 checkpoint on the verifier-
accepted teacher rollouts produced by `harness.rft_generate`.

Goal: turn `allenai/OLMo-2-1124-7B-SFT` into a task-specialized same-base teacher
`-7B-SFT-gsm` by fine-tuning on `(templated_prompt, completion)` pairs where the completion was
verified-correct on `gsm_symbolic`.  We then use this specialized teacher in step 3 (re-run Exp 4's
λ-interior sweep) to test whether state-coverage / teacher-trajectory-overlap is the bottleneck.

Loss: causal LM cross-entropy on **completion tokens only** (prompt tokens are masked out via the
standard `-100` label trick).  Cosine LR schedule with warmup, AdamW (bf16 params + bf16 grads,
fp32 optimizer state), gradient accumulation.  Single-GPU; the 7B + Adam states ≈ 80 GB on bf16
mixed but fits the 80 GB H100 with `gradient_checkpointing=True` (default).

Usage:
    python -m harness.train_sft_rft \
        --model_name allenai/OLMo-2-1124-7B-SFT \
        --data_path rft_data/gsm_symbolic_from_7B-Instruct_seed4242.jsonl \
        --output_dir harness/checkpoints/teacher_7B-SFT-gsm \
        --num_epochs 2 \
        --per_device_batch_size 1 \
        --grad_accum 8 \
        --lr 5e-6 \
        --warmup_ratio 0.05
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import _pg


class RFTDataset(Dataset):
    """One example per line in the JSONL; the SFT label-mask is computed at __getitem__ time.

    Each line is the dict produced by `harness.rft_generate`:
        {"prompt": str, "templated_prompt": str, "completion": str, "answer": str, "accuracy": float, "format": float}

    We tokenize `templated_prompt` and `completion` separately, concatenate, and produce a labels
    tensor that's `-100` on prompt tokens (so they don't contribute to the loss) and the actual
    token ids on completion tokens.
    """

    def __init__(self, path: str, tokenizer, max_length: int = 2048):
        self.examples: list[dict] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"[sft-rft] loaded {len(self.examples)} examples from {path}", flush=True)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        prompt = ex["templated_prompt"]
        completion = ex["completion"]
        # Tokenize prompt (no special tokens — the chat template already has them) + completion + EOS.
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors=None)["input_ids"]
        comp_ids = self.tokenizer(completion, add_special_tokens=False, return_tensors=None)["input_ids"]
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            eos_id = self.tokenizer.pad_token_id
        comp_ids = comp_ids + [eos_id]  # explicit EOS so the model learns to stop

        input_ids = prompt_ids + comp_ids
        labels = [-100] * len(prompt_ids) + comp_ids  # mask out prompt tokens

        # Truncate from the LEFT of the prompt if too long (keep the completion intact); if the
        # completion alone is too long, truncate it from the right.
        if len(input_ids) > self.max_length:
            if len(comp_ids) >= self.max_length:
                input_ids = comp_ids[: self.max_length]
                labels = comp_ids[: self.max_length]
            else:
                overflow = len(input_ids) - self.max_length
                input_ids = input_ids[overflow:]
                labels = labels[overflow:]
        return {"input_ids": input_ids, "labels": labels}


def _collate(batch: list[dict], pad_id: int) -> dict:
    """Right-pad input_ids to the longest in batch; labels get padded with -100."""
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, ex in enumerate(batch):
        n = len(ex["input_ids"])
        input_ids[i, :n] = torch.tensor(ex["input_ids"], dtype=torch.long)
        labels[i, :n] = torch.tensor(ex["labels"], dtype=torch.long)
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def _cosine_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal SFT loop on verifier-accepted teacher rollouts (RFT data).")
    ap.add_argument("--model_name", default="allenai/OLMo-2-1124-7B-SFT",
                    help="HF model id of the checkpoint to fine-tune.")
    ap.add_argument("--data_path", required=True, help="JSONL produced by harness.rft_generate.")
    ap.add_argument("--output_dir", required=True, help="Where to save_pretrained the result.")
    ap.add_argument("--num_epochs", type=int, default=2)
    ap.add_argument("--per_device_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gradient_checkpointing", action="store_true", default=True)
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[sft-rft] loading {args.model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, attn_implementation=_pg.get_attn_implementation(),
    ).to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    dataset = RFTDataset(args.data_path, tokenizer, max_length=args.max_length)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    dataloader = DataLoader(
        dataset, batch_size=args.per_device_batch_size, shuffle=True, drop_last=True,
        collate_fn=lambda b: _collate(b, pad_id), pin_memory=False,
    )
    steps_per_epoch = math.ceil(len(dataloader) / args.grad_accum)
    total_steps = steps_per_epoch * args.num_epochs
    num_warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"[sft-rft] {len(dataset)} examples, {len(dataloader)} micro-batches/epoch, "
          f"grad_accum={args.grad_accum} → {steps_per_epoch} optimizer steps/epoch, "
          f"{total_steps} total steps (warmup {num_warmup_steps})", flush=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scheduler = _cosine_with_warmup(optimizer, num_warmup_steps, total_steps)

    t0 = time.time()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.num_epochs):
        epoch_loss, epoch_n = 0.0, 0
        accumulated_loss, n_in_acc = 0.0, 0
        for micro_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            labels=batch["labels"])
            loss = outputs.loss
            is_last_micro = (micro_idx + 1) == len(dataloader)
            remainder = len(dataloader) % args.grad_accum
            accum_denom = remainder if is_last_micro and remainder else args.grad_accum
            (loss / accum_denom).backward()
            accumulated_loss += float(loss.item())
            n_in_acc += 1
            is_step = (micro_idx + 1) % args.grad_accum == 0 or (micro_idx + 1) == len(dataloader)
            if is_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step_loss = accumulated_loss / max(n_in_acc, 1)
                epoch_loss += step_loss; epoch_n += 1
                if global_step % args.log_every == 0:
                    elapsed = (time.time() - t0) / 60.0
                    lr_now = scheduler.get_last_lr()[0]
                    print(f"[sft-rft] epoch {epoch+1}/{args.num_epochs}  step {global_step}/{total_steps}  "
                          f"loss={step_loss:.4f}  lr={lr_now:.2e}  grad_norm={float(grad_norm):.2f}  "
                          f"elapsed={elapsed:.1f} min", flush=True)
                global_step += 1
                accumulated_loss, n_in_acc = 0.0, 0
        avg_loss = epoch_loss / max(epoch_n, 1)
        print(f"[sft-rft] epoch {epoch+1} done; avg_loss={avg_loss:.4f}", flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_disable()
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    print(f"[sft-rft] DONE: {global_step} steps in {(time.time()-t0)/60.0:.1f} min → {out}", flush=True)


if __name__ == "__main__":
    main()
