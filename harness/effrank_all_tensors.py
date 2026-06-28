"""
effrank_all_tensors — Exp 3 follow-up: full per-tensor effective rank pass.

The §6.3 result was based on the top-10-by-frob slice of each checkpoint, which
covered every attn_qkv layer (always dominant) but missed many mlp_in/mlp_down
layers per arm. This script:
  1. loads base + ckpt state dicts (cached base across the batch),
  2. computes effective rank for EVERY 2D weight tensor (ΔW and base W),
  3. writes a compact JSON per ckpt containing only the effrank fields.

  python -m harness.effrank_all_tensors --batch --out-dir figs/dtheta/effrank_full
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from .delta_theta_snapshot import _categorize, _effective_rank


CANONICAL_BATCH = [
    ("rl_baseline_s42",        "allenai/OLMo-2-0425-1B-SFT", "harness/checkpoints/rl-baseline-s42"),
    ("grpo_v2_s42",            "allenai/OLMo-2-0425-1B-SFT", "harness/checkpoints/grpo-distill-seed42-71209"),
    ("grpo_v2_s43",            "allenai/OLMo-2-0425-1B-SFT", "harness/checkpoints/grpo-distill-s43-71242"),
    ("clip1_lam005_s42",       "allenai/OLMo-2-0425-1B-SFT", "harness/checkpoints/clip1.0-lam0.05-s42-71271"),
    ("clip1_lam010_s42",       "allenai/OLMo-2-0425-1B-SFT", "harness/checkpoints/clip1.0-lam0.10-s42-71271"),
    ("clip1_lam020_s42",       "allenai/OLMo-2-0425-1B-SFT", "harness/checkpoints/clip1.0-lam0.20-s42-71271"),
    ("clip1_lam050_s42",       "allenai/OLMo-2-0425-1B-SFT", "harness/checkpoints/clip1.0-lam0.50-s42-71271"),
    ("clip1_lam085_s42",       "allenai/OLMo-2-0425-1B-SFT", "harness/checkpoints/clip1.0-lam0.85-s42-71271"),
    ("clip1_lam100_s42",       "allenai/OLMo-2-0425-1B-SFT", "harness/checkpoints/clip1.0-lam1.0-s42-71250"),
    ("v21_lam005_s42_unclip",  "allenai/OLMo-2-0425-1B-SFT", "harness/checkpoints/exp4gsm_lam0.05_seed42"),
    ("v21_lam100_s42_unclip",  "allenai/OLMo-2-0425-1B-SFT", "harness/checkpoints/exp4gsm_lam1.0_seed42"),
]


def _load_state(path: str) -> dict:
    t0 = time.time()
    mdl = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True,
    )
    sd = {k: v.detach().cpu() for k, v in mdl.state_dict().items()}
    del mdl
    print(f"[effrank-full] loaded {path} in {time.time() - t0:.1f}s")
    return sd


def measure_one(base_sd: dict, ckpt_sd: dict) -> dict:
    """Per-tensor effective rank for every 2D 'weight' tensor."""
    out = {}
    skipped = 0
    for k in sorted(base_sd.keys()):
        if k not in ckpt_sd:
            continue
        a = base_sd[k]
        b = ckpt_sd[k]
        if a.shape != b.shape or a.dim() != 2 or "weight" not in k:
            continue
        if not k.endswith(".weight"):
            continue
        a32 = a.detach().float()
        b32 = b.detach().float()
        dt = b32 - a32
        cat = _categorize(k)
        try:
            er = _effective_rank(dt)
            ber = _effective_rank(a32)
            frob = float(dt.norm().item())
            base_frob = float(a32.norm().item())
        except Exception as e:
            skipped += 1
            continue
        out[k] = {
            "category": cat,
            "shape": list(a.shape),
            "effective_rank": er,
            "base_effective_rank": ber,
            "frob_norm": frob,
            "rel_frob": frob / max(base_frob, 1e-12),
        }
        del a32, b32, dt
    return {"per_tensor": out, "skipped_n": skipped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--out-dir", default="figs/dtheta/effrank_full")
    ap.add_argument("--base", default=None, help="single-ckpt mode: base path/HF id")
    ap.add_argument("--ckpt", default=None, help="single-ckpt mode")
    ap.add_argument("--label", default=None, help="single-ckpt mode: file label")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    base_cache: dict[str, dict] = {}

    if args.batch:
        results = []
        for label, base, ckpt in CANONICAL_BATCH:
            out = os.path.join(args.out_dir, f"effrank_{label}.json")
            if os.path.exists(out):
                print(f"[effrank-full] skip {label}, already at {out}")
                continue
            try:
                if base not in base_cache:
                    base_cache[base] = _load_state(base)
                base_sd = base_cache[base]
                ckpt_sd = _load_state(ckpt)
                t0 = time.time()
                res = measure_one(base_sd, ckpt_sd)
                print(f"[effrank-full] measured {label}: {len(res['per_tensor'])} 2D weights in {time.time()-t0:.1f}s")
                res["label"] = label
                res["base"] = base
                res["ckpt"] = ckpt
                with open(out, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"[effrank-full] wrote {out}")
                del ckpt_sd
                import gc; gc.collect()
                results.append(res)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[effrank-full] FAIL {label}: {e}")
    else:
        assert args.base and args.ckpt and args.label
        base_sd = _load_state(args.base)
        ckpt_sd = _load_state(args.ckpt)
        res = measure_one(base_sd, ckpt_sd)
        res["label"] = args.label
        res["base"] = args.base
        res["ckpt"] = args.ckpt
        out = os.path.join(args.out_dir, f"effrank_{args.label}.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[effrank-full] wrote {out}")


if __name__ == "__main__":
    main()
