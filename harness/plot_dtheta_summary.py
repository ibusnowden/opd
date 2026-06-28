"""
plot_dtheta_summary — render the Exp 3 (sparse-vs-dense) overview figures
from research/figs/dtheta/dtheta_*.json.

Two figures:
  1. Bar chart: per-arm top-1%/5%/20% mass concentration  → "how sparse is Δθ"
  2. Heatmap: per-arm × per-category changed_gt_1e-4      → "which submodules each arm uses"

Also prints a flat summary table to stdout. Run AFTER `delta_theta_snapshot.py --batch`.
"""
from __future__ import annotations

import glob
import json
import os
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR = "/project/inniang/research/figs/dtheta"

# Canonical display order — RL-sparse on the left, OPD-blend, then SFT-dense on the right.
ORDER = [
    "rl_baseline_s42",
    "grpo_v2_s42",
    "grpo_v2_s43",
    "clip1_lam005_s42",
    "clip1_lam010_s42",
    "clip1_lam020_s42",
    "clip1_lam050_s42",
    "clip1_lam085_s42",
    "clip1_lam100_s42",
    "v21_lam005_s42_unclip",
    "v21_lam100_s42_unclip",
]


def main():
    runs = OrderedDict()
    for f in sorted(glob.glob(os.path.join(DIR, "dtheta_*.json"))):
        if os.path.basename(f) == "dtheta_summary.json":
            continue  # skip the aggregate summary file
        with open(f) as fh:
            j = json.load(fh)
        if isinstance(j, list):
            continue  # skip any aggregate list files
        label = j.get("label") or os.path.basename(f).replace("dtheta_", "").replace(".json", "")
        if "aggregate" not in j or j["aggregate"].get("total_params") is None:
            print(f"skip (no aggregate): {label}")
            continue
        runs[label] = j

    labels = [l for l in ORDER if l in runs] + [l for l in runs if l not in ORDER]
    aggs = [runs[l]["aggregate"] for l in labels]
    print(f"\n{'arm':<24} {'top1%':>6} {'top5%':>6} {'>1e-4':>7} {'>1e-3':>7} {'p99|Δθ|':>10}")
    for l, a in zip(labels, aggs):
        print(f"{l:<24} {a['top1_pct_mass']:>6.3f} {a['top5_pct_mass']:>6.3f} "
              f"{a['changed_gt_1e-4']:>7.3f} {a['changed_gt_1e-3']:>7.3f} "
              f"{a['p99_abs']:>10.2e}")

    # --- Figure 1: top-K mass concentration ---
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    x = np.arange(len(labels))
    w = 0.28
    ax.bar(x - w, [a["top1_pct_mass"]  for a in aggs], width=w, label="top-1% mass", color="#3b528b")
    ax.bar(x,     [a["top5_pct_mass"]  for a in aggs], width=w, label="top-5% mass", color="#21918c")
    ax.bar(x + w, [a["top20_pct_mass"] for a in aggs], width=w, label="top-20% mass", color="#fde725")
    ax.axhline(0.01, ls=":", color="grey", lw=1, alpha=0.7)
    ax.axhline(0.05, ls=":", color="grey", lw=1, alpha=0.7)
    ax.axhline(0.20, ls=":", color="grey", lw=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("fraction of total ||Δθ||² in top-K%")
    ax.set_title("Exp 3 — sparsity of Δθ across arms (1.0 = perfectly sparse; K_pct = perfectly dense)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.3, axis="y")
    out1 = os.path.join(DIR, "exp3_dtheta_sparsity.png")
    fig.savefig(out1, dpi=150)
    print(f"\nwrote {out1}")

    # --- Figure 2: by-category heatmap (changed_gt_1e-4) ---
    cats = ["embed", "attn_qkv", "attn_o", "mlp_in", "mlp_down", "norm"]
    M = np.full((len(cats), len(labels)), np.nan)
    for j, l in enumerate(labels):
        bc = runs[l].get("by_category", {})
        for i, c in enumerate(cats):
            if c in bc:
                M[i, j] = bc[c].get("changed_gt_1e-4", np.nan)
    fig, ax = plt.subplots(figsize=(11, 4.2), constrained_layout=True)
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=np.nanmax(M))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats)
    for i in range(len(cats)):
        for j in range(len(labels)):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < (np.nanmax(M)/2) else "black", fontsize=8)
    plt.colorbar(im, ax=ax, label="fraction of weights |Δθ| > 1e-4")
    ax.set_title("Exp 3 — changed-fraction by submodule × arm")
    out2 = os.path.join(DIR, "exp3_dtheta_by_category.png")
    fig.savefig(out2, dpi=150)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
