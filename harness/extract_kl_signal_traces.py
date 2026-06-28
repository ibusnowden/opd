"""
Extract per-step `kl_signal/*` traces from offline W&B binaries for the
71208 seed-43 λ-sweep and produce the §8.2 mechanism figure.

The §7.7 open-follow-up predicts: breakthrough arms (λ=0.05, λ=0.50) should
show heavy_tail_frac plateauing as rollout-acc lifts; dead-zone arms
(λ=0.10–0.35, λ=0.70–1.00) should show it monotonically rising.

This script:
  1. Scans every offline-run-* directory under research/wandb/
  2. Filters to runs whose display_name starts with `exp4gsm_lam`
     and whose seed is 43  (i.e. the 71208 sweep).
  3. Extracts per-step kl_signal/{p50,p90,p99,abs_max,heavy_tail_frac}
     and reward/accuracy traces.
  4. Saves them to research/figs/kl_signal_71208.npz and plots
     research/figs/exp4_kl_signal_mechanism.png.
"""
from __future__ import annotations
import os, glob, json, sys
from collections import defaultdict
import numpy as np

from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb


def extract_run(wandb_dir: str):
    """Return (display_name, seed, lam, history_dict) or None."""
    wfile = [f for f in os.listdir(wandb_dir) if f.endswith(".wandb")]
    if not wfile:
        return None
    wfile = os.path.join(wandb_dir, wfile[0])
    ds = DataStore()
    try:
        ds.open_for_scan(wfile)
    except Exception:
        return None

    display = seed = lam = None
    history = defaultdict(list)
    keys_of_interest = {
        "_step",
        "reward/accuracy",
        "reward",
        "kl_signal/p50",
        "kl_signal/p90",
        "kl_signal/p99",
        "kl_signal/abs_max",
        "kl_signal/heavy_tail_frac",
        "teacher/reverse_kl",
        "loss/teacher_term",
        "loss/outcome_term",
    }
    while True:
        r = ds.scan_data()
        if r is None:
            break
        rec = pb.Record()
        rec.ParseFromString(r)
        t = rec.WhichOneof("record_type")
        if t == "run":
            display = rec.run.display_name
            for c in rec.run.config.update:
                if c.key == "seed":
                    seed = c.value_json
                if c.key == "lam":
                    lam = c.value_json
        elif t == "history":
            for item in rec.history.item:
                key = item.key or ".".join(item.nested_key)
                if key in keys_of_interest:
                    try:
                        v = json.loads(item.value_json)
                    except Exception:
                        continue
                    history[key].append(float(v))
    return display, seed, lam, dict(history)


def collect_71208(root: str = "/project/inniang/research/wandb"):
    """Find the seed-43 unclipped lambda sweep (71208) and return per-lambda traces."""
    by_lam = {}
    for d in sorted(glob.glob(os.path.join(root, "offline-run-20260515_151*"))):
        info = extract_run(d)
        if info is None:
            continue
        display, seed, lam, hist = info
        if not (display and display.startswith("exp4gsm_lam") and seed == "43"):
            continue
        lam_f = float(lam) if lam is not None else None
        by_lam[lam_f] = {"display": display, "dir": os.path.basename(d), "history": hist}
    return by_lam


def collect_71395(root: str = "/project/inniang/research/wandb"):
    """Find the clipped 3-seed × 4-lambda 71395 sweep and return per-(seed, lambda) traces."""
    runs = {}
    for d in sorted(glob.glob(os.path.join(root, "offline-run-2026051[9-]_*")) +
                    glob.glob(os.path.join(root, "offline-run-20260520_*"))):
        info = extract_run(d)
        if info is None:
            continue
        display, seed, lam, hist = info
        if not (display and display.startswith("clip1.0-lam")):
            continue
        lam_f = float(lam) if lam is not None else None
        runs[(int(seed), lam_f)] = {"display": display, "dir": os.path.basename(d), "history": hist}
    return runs


def main():
    by_lam = collect_71208()
    print(f"found {len(by_lam)} seed-43 71208 (unclipped) runs: {sorted(by_lam)}")

    out_npz = "/project/inniang/research/figs/kl_signal_71208.npz"
    blob = {}
    for lam, run in sorted(by_lam.items()):
        h = run["history"]
        for k, v in h.items():
            blob[f"lam{lam}/{k}"] = np.asarray(v, dtype=np.float64)
    np.savez_compressed(out_npz, **blob)
    print(f"saved npz: {out_npz} (keys={len(blob)})")

    # also: clipped 71395 sweep
    clip_runs = collect_71395()
    print(f"found {len(clip_runs)} seed×lam clipped 71395 runs: {sorted(clip_runs.keys())}")
    out_npz_clip = "/project/inniang/research/figs/kl_signal_71395_clipped.npz"
    blob_c = {}
    for (seed, lam), run in sorted(clip_runs.items()):
        h = run["history"]
        for k, v in h.items():
            blob_c[f"s{seed}_lam{lam}/{k}"] = np.asarray(v, dtype=np.float64)
    np.savez_compressed(out_npz_clip, **blob_c)
    print(f"saved npz: {out_npz_clip} (keys={len(blob_c)})")

    # also dump a small summary json for downstream parsing
    summary = {}
    for lam, run in sorted(by_lam.items()):
        h = run["history"]
        step = h.get("_step", [])
        # final 50-step means; the open-follow-up's predicted contrast is in the *trend*, not the mean,
        # but the mean already separates breakthrough from dead-zone.
        tail = lambda key: float(np.mean(h.get(key, [np.nan])[-50:])) if h.get(key) else float("nan")
        summary[f"lam={lam}"] = {
            "n_steps": len(step),
            "final50_reward_accuracy": tail("reward/accuracy"),
            "final50_kl_p99": tail("kl_signal/p99"),
            "final50_heavy_tail_frac": tail("kl_signal/heavy_tail_frac"),
            "final50_abs_max": tail("kl_signal/abs_max"),
            "final50_teacher_reverse_kl": tail("teacher/reverse_kl"),
        }
    summary_path = "/project/inniang/research/figs/kl_signal_71208_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved summary: {summary_path}")
    for k, v in summary.items():
        print(f"  {k:>10s}  p99(final50)={v['final50_kl_p99']:.3f}  "
              f"heavy_tail(final50)={v['final50_heavy_tail_frac']:.4f}  "
              f"acc(final50)={v['final50_reward_accuracy']:.3f}")


if __name__ == "__main__":
    main()
