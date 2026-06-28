"""
prune_dtheta_eval — Exp 3 follow-up: prune the bottom-p% of |Δθ| (global magnitude),
then re-evaluate pass@k. Tests whether the "broader-tier" arms identified in §6
(clipped low-λ, RL baseline) are functionally redundant or carry small-but-load-bearing
changes. The §6 result was static (top-K% mass concentration); this script makes it
dynamic.

Mask semantics: GLOBAL magnitude pruning — find threshold τ such that |Δθ| < τ for the
bottom p% of weights ACROSS THE ENTIRE MODEL (not per-tensor). For weights with
|Δθ| < τ, the pruned model uses θ_init; for the rest, θ_trained.

  python -m harness.prune_dtheta_eval \\
      --base allenai/OLMo-2-0425-1B-SFT \\
      --ckpt harness/checkpoints/clip1.0-lam0.10-s42-71271 \\
      --prune-pct 0.5 --task gsm_symbolic \\
      --n-prompts 64 --n-samples 16 \\
      --out figs/dtheta/prune/clip1_lam010_s42_p50.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import reasoning_gym as rg
from reasoning_gym.composite import DatasetSpec

from . import _pg
from .eval_passk import evaluate_passk


def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Load a model's state_dict into CPU fp32 tensors."""
    t0 = time.time()
    mdl = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True,
    )
    sd = {k: v.detach().cpu() for k, v in mdl.state_dict().items()}
    del mdl
    print(f"[prune] loaded sd from {path} in {time.time() - t0:.1f}s ({len(sd)} tensors)")
    return sd


def compute_global_threshold(base_sd: dict, ckpt_sd: dict, prune_pct: float) -> dict:
    """Find threshold τ such that the bottom `prune_pct` of MOVED weights (|Δθ| > 0)
    have |Δθ| ≤ τ. Returns a dict with τ + summary stats.

    Why "moved weights" and not "all weights": bf16 storage means many weights are
    bit-identical between base and ckpt (~86% for GRPO-v2-s42); pruning by
    percentile-of-all-weights makes prune_pct meaningless once it falls inside the
    zero-spike. Defining prune_pct on the moved subset gives a clean sweep:
    p=0 = full ckpt, p=1 = revert all moved weights to base.
    """
    parts = []
    total_n = 0
    for k in base_sd:
        if k not in ckpt_sd or base_sd[k].shape != ckpt_sd[k].shape:
            continue
        absd = (ckpt_sd[k].float() - base_sd[k].float()).abs().flatten()
        parts.append(absd)
        total_n += absd.numel()
    all_abs = torch.cat(parts)
    del parts

    # Only the nonzero-Δθ subset defines the "moved" weights.
    moved_mask = all_abs > 0.0
    n_moved = int(moved_mask.sum().item())
    n_zero = total_n - n_moved
    print(f"[prune] base ckpt sparsity: {n_zero/1e6:.1f}M of {total_n/1e6:.1f}M weights are bit-identical "
          f"({n_zero/total_n*100:.1f}%); moved = {n_moved/1e6:.1f}M ({n_moved/total_n*100:.1f}%)")

    if prune_pct <= 0.0 or n_moved == 0:
        return {"tau": -1.0, "n_moved": n_moved, "n_total": total_n, "n_zero": n_zero,
                "prune_pct_target": prune_pct}

    moved_abs = all_abs[moved_mask]
    k = max(1, min(int(round(prune_pct * moved_abs.numel())), moved_abs.numel()))
    tau = torch.kthvalue(moved_abs, k).values.item()
    n_at_tau = int((moved_abs == tau).sum().item())
    n_below_tau = int((moved_abs < tau).sum().item())
    print(f"[prune] global threshold (on moved weights) τ = {tau!r} "
          f"(target {prune_pct*100:.1f}% of {n_moved/1e6:.1f}M moved → k={k/1e6:.1f}M)")
    print(f"[prune] within moved subset: |Δθ| <  τ: {n_below_tau/n_moved*100:.2f}%  "
          f"|Δθ| == τ: {n_at_tau/n_moved*100:.2f}%  "
          f"|Δθ| <= τ: {(n_below_tau + n_at_tau)/n_moved*100:.2f}%")
    return {"tau": tau, "n_moved": n_moved, "n_total": total_n, "n_zero": n_zero,
            "prune_pct_target": prune_pct}


def apply_prune_to_state_dict(base_sd: dict, ckpt_sd: dict, thresh_info: dict) -> tuple[dict, dict]:
    """Build the pruned state_dict: for moved weights with |Δθ| ≤ τ, revert to base;
    keep ckpt elsewhere. (Unmoved weights are by definition equal to base already, so
    "reverting" them is a no-op — but it's correct semantics.)
    """
    tau = thresh_info["tau"]
    pruned = {}
    n_total_real = 0
    n_pruned_real = 0     # of moved weights only
    n_moved_real = 0
    if tau < 0.0:
        # no pruning requested
        for k in ckpt_sd:
            pruned[k] = ckpt_sd[k]
        stats = {**thresh_info, "actual_prune_frac_of_moved": 0.0,
                 "actual_prune_frac_of_total": 0.0,
                 "n_pruned": 0}
        print(f"[prune] prune disabled — full ckpt passed through")
        return pruned, stats
    for k in ckpt_sd:
        if k not in base_sd or base_sd[k].shape != ckpt_sd[k].shape:
            pruned[k] = ckpt_sd[k]
            continue
        absd = (ckpt_sd[k].float() - base_sd[k].float()).abs()
        moved = absd > 0.0
        # Mask: revert iff weight moved AND |Δθ| ≤ τ
        mask = moved & (absd <= tau)
        n_total_real += absd.numel()
        n_moved_real += int(moved.sum().item())
        n_pruned_real += int(mask.sum().item())
        pruned[k] = torch.where(mask, base_sd[k].to(ckpt_sd[k].dtype), ckpt_sd[k])
    actual_of_moved = n_pruned_real / max(n_moved_real, 1)
    actual_of_total = n_pruned_real / max(n_total_real, 1)
    print(f"[prune] actually pruned {n_pruned_real/1e6:.1f}M weights "
          f"({actual_of_moved*100:.2f}% of moved, {actual_of_total*100:.2f}% of total)")
    stats = {**thresh_info,
             "actual_prune_frac_of_moved": actual_of_moved,
             "actual_prune_frac_of_total": actual_of_total,
             "n_pruned": n_pruned_real,
             "n_moved_check": n_moved_real}
    return pruned, stats


def _eval_dataset(task: str, n_prompts: int, seed: int):
    return rg.create_dataset("composite", size=n_prompts, seed=seed,
                             datasets=[DatasetSpec(name=task, weight=1.0, config={})])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prune-pct", type=float, required=True,
                    help="Fraction of weights to revert to base (by smallest |Δθ|).")
    ap.add_argument("--task", default="gsm_symbolic")
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--k", default="1,2,4,8,16")
    ap.add_argument("--temps", default="0.6")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--gen-batch-size", type=int, default=8)
    ap.add_argument("--eval-seed", type=int, default=1_000_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    k_values = sorted({int(x) for x in args.k.split(",") if x.strip()})
    temps = [float(x) for x in args.temps.split(",") if x.strip()]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[prune-eval] base={args.base} ckpt={args.ckpt} prune_pct={args.prune_pct} device={device}")

    # 1. Load base + ckpt state dicts (CPU)
    base_sd = _load_state_dict(args.base)
    ckpt_sd = _load_state_dict(args.ckpt)

    # 2. Compute threshold and build pruned state dict
    thresh_info = compute_global_threshold(base_sd, ckpt_sd, args.prune_pct)
    pruned_sd, stats = apply_prune_to_state_dict(base_sd, ckpt_sd, thresh_info)
    del base_sd
    import gc; gc.collect()

    # 3. Load the ckpt model, swap in the pruned state dict, move to GPU
    print(f"[prune-eval] loading ckpt model for eval...")
    t0 = time.time()
    model, tokenizer = _pg.load_model(args.ckpt, device, gradient_checkpointing=False)
    model.eval()
    if getattr(tokenizer, "chat_template", None) is None:
        raise RuntimeError(f"{args.ckpt!r} has no chat_template.")
    # Replace state dict
    missing, unexpected = model.load_state_dict(pruned_sd, strict=False)
    if missing or unexpected:
        print(f"[prune-eval] WARN: missing={len(missing)} unexpected={len(unexpected)} keys when loading pruned sd")
    del pruned_sd, ckpt_sd
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[prune-eval] model ready ({time.time() - t0:.1f}s)")

    # 4. Evaluate
    all_metrics: dict[str, dict[str, float]] = {}
    for T in temps:
        ds = _eval_dataset(args.task, args.n_prompts, args.eval_seed)
        m = evaluate_passk(model, tokenizer, ds,
                           n_prompts=args.n_prompts, n_samples=args.n_samples,
                           k_values=k_values, temperature=T,
                           max_new_tokens=args.max_new_tokens, gen_batch_size=args.gen_batch_size,
                           compute_self_bleu=False)
        all_metrics[f"T={T}"] = m
        # short print
        print(f"[prune-eval] T={T}: p@1={m.get('eval/pass@1', float('nan')):.3f}  p@16={m.get('eval/pass@16', float('nan')):.3f}  ent={m.get('eval/token_entropy', float('nan')):.2f}")

    # 5. Write JSON
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = {
        "base": args.base,
        "ckpt": args.ckpt,
        "prune_pct_target": args.prune_pct,
        "prune_stats": stats,
        "task": args.task,
        "n_prompts": args.n_prompts,
        "n_samples": args.n_samples,
        "k_values": k_values,
        "metrics_by_temp": all_metrics,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[prune-eval] wrote {args.out}")


if __name__ == "__main__":
    main()
