"""
delta_theta_snapshot — Exp 3 (sparse-vs-dense) Δθ measurement utility.

Loads a base-init model + a trained checkpoint, computes per-parameter
Δθ = θ_trained − θ_init, and reports sparsity / geometry proxies:

  Per-tensor:
    - |Δθ|: mean, p50, p90, p99 of absolute deviation
    - top-1%/5%/20% mass concentration: fraction of total ||Δθ||_2² lying in
      the top-K% absolute values (1.0 = perfectly sparse, k_pct = perfectly dense)
    - effective rank (2D matrices only): exp(H(σ_i / Σ σ_j)) where H is Shannon
      entropy of the normalized singular-value spectrum

  Aggregate (model-wide, weighted by tensor size or weight-only):
    - identical macro-stats
    - parameter "changed" fractions at thresholds {1e-5, 1e-4, 1e-3, 1e-2}

  Output:  a JSON blob per (base, ckpt) pair, plus an optional per-tensor CSV
  for downstream plotting.

Why this matters: §6 / §8.3 hypothesis — OPD inherits update geometry from the
teacher. If true: RL-teacher OPD → sparse subnetwork (RL-like); SFT-teacher OPD
→ dense cloud (SFT-like). The recent literature (RL's Razor) predicts a clean
geometric distinction; this script lets us check it.

Usage:
  python -m harness.delta_theta_snapshot \
      --base allenai/OLMo-2-0425-1B-SFT \
      --ckpt harness/checkpoints/clip1.0-lam0.10-s42-71271 \
      --out figs/dtheta_clip1_lam010_s42.json

  # or batch via the wrapper script:
  python -m harness.delta_theta_snapshot --batch
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


# Module categories so we can compare "attention vs MLP" sparsity (the RL-vs-SFT
# story may differ by submodule).
def _categorize(name: str) -> str:
    n = name.lower()
    if "embed" in n or "lm_head" in n:
        return "embed"
    if "layernorm" in n or "rmsnorm" in n or n.endswith(".norm.weight"):
        return "norm"
    if "self_attn" in n or "attention" in n:
        if "q_proj" in n or "k_proj" in n or "v_proj" in n: return "attn_qkv"
        if "o_proj" in n: return "attn_o"
        return "attn_other"
    if "mlp" in n or "feed_forward" in n:
        if "gate" in n or "up_proj" in n:   return "mlp_in"
        if "down_proj" in n:                return "mlp_down"
        return "mlp_other"
    return "other"


def _quantiles(absdt: torch.Tensor, qs=(0.5, 0.9, 0.99)):
    # torch.quantile is slow & memory-heavy on >10M; fall back to sort + index
    flat = absdt.flatten().float()
    if flat.numel() == 0:
        return [float("nan")] * len(qs)
    # cheap: torch.kthvalue is faster than full sort
    n = flat.numel()
    out = []
    for q in qs:
        k = max(1, int(round(q * n)))
        v = torch.kthvalue(flat, k).values.item()
        out.append(v)
    return out


def _top_mass(absdt: torch.Tensor, p: float) -> float:
    """Fraction of total Σ|Δθ|² coming from the top-p fraction of |Δθ| values."""
    flat = absdt.flatten().float()
    if flat.numel() == 0:
        return float("nan")
    sq = flat * flat
    total = sq.sum().item()
    if total <= 0:
        return float("nan")
    k = max(1, int(round(p * flat.numel())))
    topk_sq = torch.topk(sq, k, sorted=False).values.sum().item()
    return topk_sq / total


def _changed_frac(absdt: torch.Tensor, thresh: float) -> float:
    if absdt.numel() == 0:
        return float("nan")
    return (absdt > thresh).float().mean().item()


def _effective_rank(W: torch.Tensor, max_dim: int = 4096) -> float:
    """For 2D ΔW: exp(H(σ_i / Σσ_j))."""
    if W.dim() != 2 or W.numel() == 0:
        return float("nan")
    # truncate to a manageable shape for SVD; ΔW for big embed/MLP is huge.
    a, b = W.shape
    if a > max_dim or b > max_dim:
        # power-iteration-style proxy: use a random projection (preserves spectrum
        # in expectation enough for an order-of-magnitude effective-rank).
        torch.manual_seed(0)
        if a > max_dim:
            P = torch.randn(max_dim, a, device=W.device, dtype=W.dtype) / (a ** 0.5)
            W = P @ W
        if W.shape[1] > max_dim:
            P = torch.randn(W.shape[1], max_dim, device=W.device, dtype=W.dtype) / (W.shape[1] ** 0.5)
            W = W @ P
    try:
        # economy SVD on CPU/GPU; clamp dtype to fp32
        s = torch.linalg.svdvals(W.float())
    except Exception:
        return float("nan")
    s = s[s > 0]
    if s.numel() == 0:
        return float("nan")
    p = s / s.sum()
    H = -(p * p.log()).sum().item()
    return float(torch.exp(torch.tensor(H)).item())


def measure(base_state_dict: dict, ckpt_state_dict: dict, *, sample_eff_rank: bool = True) -> dict:
    """Per-tensor and aggregate Δθ stats."""
    per_tensor = []
    keys = sorted(base_state_dict.keys())
    skipped = []
    for k in keys:
        if k not in ckpt_state_dict:
            skipped.append((k, "missing-in-ckpt"))
            continue
        a = base_state_dict[k]
        b = ckpt_state_dict[k]
        if a.shape != b.shape:
            skipped.append((k, f"shape mismatch {a.shape}/{b.shape}"))
            continue
        # ensure dtype matches for diff
        a32 = a.detach().float()
        b32 = b.detach().float()
        dt = b32 - a32
        absdt = dt.abs()
        cat = _categorize(k)
        qs = _quantiles(absdt)
        rec = {
            "name": k,
            "category": cat,
            "shape": list(a.shape),
            "n_params": int(a.numel()),
            "mean_abs": float(absdt.mean().item()),
            "p50_abs": qs[0],
            "p90_abs": qs[1],
            "p99_abs": qs[2],
            "top1_pct_mass": _top_mass(absdt, 0.01),
            "top5_pct_mass": _top_mass(absdt, 0.05),
            "top20_pct_mass": _top_mass(absdt, 0.20),
            "changed_gt_1e-5": _changed_frac(absdt, 1e-5),
            "changed_gt_1e-4": _changed_frac(absdt, 1e-4),
            "changed_gt_1e-3": _changed_frac(absdt, 1e-3),
            "frob_norm": float(absdt.norm().item()),
            "rel_frob": float(absdt.norm().item() / max(a32.norm().item(), 1e-12)),
        }
        # Effective rank for 2D weights only, and only on a sample (it's expensive)
        if sample_eff_rank and dt.dim() == 2 and "weight" in k:
            rec["effective_rank"] = _effective_rank(dt)
            rec["base_effective_rank"] = _effective_rank(a32)
        per_tensor.append(rec)
        del a32, b32, dt, absdt

    # aggregate across all params (weighted by n_params)
    total_params = sum(r["n_params"] for r in per_tensor)
    def wmean(field):
        if total_params == 0:
            return float("nan")
        return sum(r[field] * r["n_params"] for r in per_tensor if r.get(field) is not None and r[field] == r[field]) / total_params

    agg = {
        "total_params": total_params,
        "n_tensors": len(per_tensor),
        "mean_abs": wmean("mean_abs"),
        "p50_abs": wmean("p50_abs"),
        "p90_abs": wmean("p90_abs"),
        "p99_abs": wmean("p99_abs"),
        "top1_pct_mass": wmean("top1_pct_mass"),
        "top5_pct_mass": wmean("top5_pct_mass"),
        "top20_pct_mass": wmean("top20_pct_mass"),
        "changed_gt_1e-5": wmean("changed_gt_1e-5"),
        "changed_gt_1e-4": wmean("changed_gt_1e-4"),
        "changed_gt_1e-3": wmean("changed_gt_1e-3"),
    }
    # per-category aggregates
    cats = {}
    for r in per_tensor:
        cats.setdefault(r["category"], []).append(r)
    by_category = {}
    for cat, rs in cats.items():
        tp = sum(r["n_params"] for r in rs)
        if tp == 0:
            continue
        by_category[cat] = {
            "n_params": tp,
            "n_tensors": len(rs),
            "mean_abs": sum(r["mean_abs"] * r["n_params"] for r in rs) / tp,
            "top1_pct_mass": sum(r["top1_pct_mass"] * r["n_params"] for r in rs) / tp,
            "top5_pct_mass": sum(r["top5_pct_mass"] * r["n_params"] for r in rs) / tp,
            "changed_gt_1e-4": sum(r["changed_gt_1e-4"] * r["n_params"] for r in rs) / tp,
            "rel_frob_mean": sum(r["rel_frob"] for r in rs) / len(rs),
        }
    return {
        "aggregate": agg,
        "by_category": by_category,
        "per_tensor": per_tensor,
        "skipped": skipped,
    }


def _load_state(path_or_id: str, device: str = "cpu") -> dict:
    """Load just the state dict (no model wrapper) — cheaper, no compile."""
    t0 = time.time()
    mdl = AutoModelForCausalLM.from_pretrained(
        path_or_id, torch_dtype=torch.float32, device_map=device, low_cpu_mem_usage=True,
    )
    sd = {k: v.detach().cpu() for k, v in mdl.state_dict().items()}
    del mdl
    print(f"[Δθ] loaded {path_or_id} in {time.time() - t0:.1f}s ({len(sd)} tensors)")
    return sd


def run_one(base: str, ckpt: str, out: str, label: str | None = None,
            sample_eff_rank: bool = True, save_per_tensor: bool = False) -> dict:
    print(f"[Δθ] base={base}  ckpt={ckpt}  out={out}")
    base_sd = _load_state(base)
    ckpt_sd = _load_state(ckpt)
    t0 = time.time()
    res = measure(base_sd, ckpt_sd, sample_eff_rank=sample_eff_rank)
    res["base"] = base
    res["ckpt"] = ckpt
    res["label"] = label or os.path.basename(ckpt.rstrip("/"))
    # if not saving per-tensor in JSON, drop it
    if not save_per_tensor:
        # but keep a compact view: keep top-10 most-changed tensors by frob
        top = sorted(res["per_tensor"], key=lambda r: -r["frob_norm"])[:10]
        res["top10_by_frob"] = top
        res["per_tensor"] = None
    print(f"[Δθ] measured in {time.time() - t0:.1f}s")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[Δθ] wrote {out}")
    return res


# -- batch mode: snapshot all the canonical checkpoints in one call ----------

CANONICAL_BATCH = [
    # label, base, ckpt
    # Exp 1 winners
    ("rl_baseline_s42",        "allenai/OLMo-2-0425-1B-SFT", "harness/checkpoints/rl-baseline-s42"),
    # § 7.5 GRPO multi-seed
    ("grpo_v2_s42",            "allenai/OLMo-2-0425-1B-SFT", "harness/checkpoints/grpo-distill-seed42-71209"),
    ("grpo_v2_s43",            "allenai/OLMo-2-0425-1B-SFT", "harness/checkpoints/grpo-distill-s43-71242"),
    # § 7.7 clipped low-λ winners (seed 42 only — keep batch cheap)
    ("clip1_lam005_s42",       "allenai/OLMo-2-0425-1B-SFT", "harness/checkpoints/clip1.0-lam0.05-s42-71271"),
    ("clip1_lam010_s42",       "allenai/OLMo-2-0425-1B-SFT", "harness/checkpoints/clip1.0-lam0.10-s42-71271"),
    ("clip1_lam020_s42",       "allenai/OLMo-2-0425-1B-SFT", "harness/checkpoints/clip1.0-lam0.20-s42-71271"),
    # § 7.7 dead-zone (large λ) + pure OPD corner
    ("clip1_lam050_s42",       "allenai/OLMo-2-0425-1B-SFT", "harness/checkpoints/clip1.0-lam0.50-s42-71271"),
    ("clip1_lam085_s42",       "allenai/OLMo-2-0425-1B-SFT", "harness/checkpoints/clip1.0-lam0.85-s42-71271"),
    ("clip1_lam100_s42",       "allenai/OLMo-2-0425-1B-SFT", "harness/checkpoints/clip1.0-lam1.0-s42-71250"),
    # § 7.2 unclipped v2.1 — bimodal-breakthrough sample
    ("v21_lam005_s42_unclip",  "allenai/OLMo-2-0425-1B-SFT", "harness/checkpoints/exp4gsm_lam0.05_seed42"),
    ("v21_lam100_s42_unclip",  "allenai/OLMo-2-0425-1B-SFT", "harness/checkpoints/exp4gsm_lam1.0_seed42"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default="figs/dtheta")
    ap.add_argument("--batch", action="store_true",
                    help="Process the canonical batch (10 ckpts) into --out-dir")
    ap.add_argument("--no-effrank", action="store_true",
                    help="Skip ΔW effective-rank (much faster on big matrices)")
    ap.add_argument("--save-per-tensor", action="store_true")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    sample_er = not args.no_effrank

    if args.batch:
        # Cache the base state-dict once across the batch (the canonical batch
        # uses one base; if some entries use a different base we'd reload).
        base_cache: dict[str, dict] = {}
        results = []
        for label, base, ckpt in CANONICAL_BATCH:
            out = os.path.join(args.out_dir, f"dtheta_{label}.json")
            if os.path.exists(out):
                print(f"[Δθ] skipping {label} (already exists at {out})")
                with open(out) as f: results.append(json.load(f))
                continue
            try:
                if base not in base_cache:
                    base_cache[base] = _load_state(base)
                base_sd = base_cache[base]
                ckpt_sd = _load_state(ckpt)
                t0 = time.time()
                res = measure(base_sd, ckpt_sd, sample_eff_rank=sample_er)
                print(f"[Δθ] measured {label} in {time.time() - t0:.1f}s")
                res["base"] = base
                res["ckpt"] = ckpt
                res["label"] = label
                if not args.save_per_tensor:
                    top = sorted(res["per_tensor"], key=lambda r: -r["frob_norm"])[:10]
                    res["top10_by_frob"] = top
                    res["per_tensor"] = None
                Path(out).parent.mkdir(parents=True, exist_ok=True)
                with open(out, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"[Δθ] wrote {out}")
                del ckpt_sd
                results.append(res)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[Δθ] FAILED {label}: {e}")
                results.append({"label": label, "error": str(e)})
        # summary table
        summary_path = os.path.join(args.out_dir, "dtheta_summary.json")
        summary = []
        for r in results:
            agg = r.get("aggregate", {})
            summary.append({
                "label": r.get("label"),
                "ckpt": r.get("ckpt"),
                "total_params": agg.get("total_params"),
                "mean_abs": agg.get("mean_abs"),
                "p99_abs": agg.get("p99_abs"),
                "top1_pct_mass": agg.get("top1_pct_mass"),
                "top5_pct_mass": agg.get("top5_pct_mass"),
                "changed_gt_1e-4": agg.get("changed_gt_1e-4"),
                "changed_gt_1e-3": agg.get("changed_gt_1e-3"),
            })
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[Δθ] batch summary: {summary_path}")
        for s in summary:
            if s.get("total_params") is None:
                print(f"  {s['label']:<25} ERR")
                continue
            print(f"  {s['label']:<25}  top1%mass={s['top1_pct_mass']:.3f}  top5%mass={s['top5_pct_mass']:.3f}  "
                  f"changed>1e-4={s['changed_gt_1e-4']:.3f}  >1e-3={s['changed_gt_1e-3']:.3f}  "
                  f"p99|Δθ|={s['p99_abs']:.2e}")
        return

    assert args.base and args.ckpt and (args.out or args.label), "need --base, --ckpt, --out (or --label)"
    out = args.out or os.path.join(args.out_dir, f"dtheta_{args.label}.json")
    run_one(args.base, args.ckpt, out, label=args.label,
            sample_eff_rank=sample_er, save_per_tensor=args.save_per_tensor)


if __name__ == "__main__":
    main()
