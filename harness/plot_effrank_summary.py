"""
plot_effrank_summary — Exp 3.3 figure: per-arm effective rank of ΔW
on attn_qkv (the submodule with the largest frob norm in all arms).

Reads the per-tensor records from figs/dtheta/dtheta_*.json (effective_rank field
present after the --no-effrank-OFF batch). Renders a bar chart of
(arm, ratio = mean attn_qkv ΔW eff_rank / mean attn_qkv base eff_rank).

Output:
  research/figs/dtheta/exp3_attn_qkv_effrank_ratio.png
"""
from __future__ import annotations
import glob, json, os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARM_ORDER = [
    ("rl_baseline_s42",       "RL baseline",         "broader"),
    ("clip1_lam005_s42",      "clip λ=0.05",         "broader"),
    ("clip1_lam010_s42",      "clip λ=0.10",         "broader"),
    ("clip1_lam020_s42",      "clip λ=0.20",         "broader"),
    ("v21_lam005_s42_unclip", "v2.1 unclipped λ=0.05", "broader"),
    ("grpo_v2_s42",           "GRPO-v2 s42",         "sharper"),
    ("grpo_v2_s43",           "GRPO-v2 s43",         "sharper"),
    ("clip1_lam050_s42",      "clip λ=0.50",         "sharper"),
    ("clip1_lam085_s42",      "clip λ=0.85",         "sharper"),
    ("clip1_lam100_s42",      "pure OPD λ=1",        "sharper"),
    ("v21_lam100_s42_unclip", "v2.1 unclipped λ=1",  "sharper"),
]

TIER_COLOR = {"broader": "#1b6ca8", "sharper": "#c0392b"}


def load_attn_qkv_effrank(label: str) -> tuple[float, float] | None:
    f = f"/project/inniang/research/figs/dtheta/dtheta_{label}.json"
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    # per_tensor was nulled at batch-save time; top10_by_frob has the same fields
    pt = j.get("per_tensor") or j.get("top10_by_frob")
    if not pt:
        return None
    ranks, base_ranks = [], []
    for r in pt:
        if r.get("category") != "attn_qkv":
            continue
        er = r.get("effective_rank")
        ber = r.get("base_effective_rank")
        if er is not None and er == er:  # not NaN
            ranks.append(er)
        if ber is not None and ber == ber:
            base_ranks.append(ber)
    if not ranks or not base_ranks:
        return None
    return (sum(ranks) / len(ranks), sum(base_ranks) / len(base_ranks))


def main():
    rows = []
    for label, display, tier in ARM_ORDER:
        v = load_attn_qkv_effrank(label)
        if v is None:
            continue
        mean_r, mean_b = v
        rows.append((display, tier, mean_r, mean_b, mean_r / max(mean_b, 1e-9)))

    print(f"{'arm':<24} {'tier':<8} {'ΔW rank':>10} {'base rank':>10} {'ratio':>7}")
    for display, tier, r, b, ratio in rows:
        print(f"{display:<24} {tier:<8} {r:>10.1f} {b:>10.1f} {ratio:>7.3f}")

    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
        "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
    })

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    labels = [r[0] for r in rows]
    ratios = [r[4] for r in rows]
    colors = [TIER_COLOR[r[1]] for r in rows]
    x = list(range(len(labels)))
    bars = ax.bar(x, ratios, color=colors, edgecolor="white", linewidth=1.2)
    ax.axhline(1.0, ls="--", color="#555", lw=1.0, alpha=0.7)
    ax.text(len(labels) - 0.4, 1.005, "base eff_rank", color="#555", fontsize=10, va="bottom", ha="right")

    # value labels on bars
    for xi, ratio, r in zip(x, ratios, rows):
        ax.text(xi, ratio + 0.012, f"{ratio:.3f}",
                ha="center", va="bottom", fontsize=10, color=TIER_COLOR[r[1]])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0.70, 1.12)
    ax.set_ylabel("effective rank ratio  (ΔW attn_qkv mean) / (base attn_qkv mean)")
    ax.set_title("Broader-tier ΔW is LOWER-rank than base; sharper-tier matches or exceeds base", pad=12)
    ax.grid(alpha=0.3, axis="y", linewidth=0.6)

    # tier legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=TIER_COLOR["broader"], label="broader-tier arms (more weights moved, more dispersed)"),
               Patch(facecolor=TIER_COLOR["sharper"], label="sharper-tier arms (fewer weights moved, more concentrated)")]
    ax.legend(handles=handles, loc="upper left", frameon=False)

    fig.text(0.01, -0.03,
             "Source: per-tensor effective rank of ΔW for attn_qkv weight matrices, averaged across all "
             "transformer layers. Effective rank = exp(H(σ_i / Σσ_j)) on the singular spectrum of ΔW; "
             "matrices >4096 in either dim use a random-projection proxy. Data from `delta_theta_snapshot.py` "
             "batch (11 ckpts); base = allenai/OLMo-2-0425-1B-SFT.",
             ha="left", va="top", fontsize=8.5, color="#555", style="italic")

    out = "/project/inniang/research/figs/dtheta/exp3_attn_qkv_effrank_ratio.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
