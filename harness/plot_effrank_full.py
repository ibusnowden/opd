"""
plot_effrank_full — multi-submodule effective rank summary.

Reads research/figs/dtheta/effrank_full/effrank_*.json (output of effrank_all_tensors.py
which covered EVERY 2D weight tensor, not just the top-10 by frob).

For each (arm, submodule category): mean ΔW effective rank / mean base effective rank.
Render a grouped bar chart and a per-submodule line plot.

Output:
  research/figs/dtheta/exp3_effrank_full.png
"""
from __future__ import annotations
import glob, json, os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = [
    ("rl_baseline_s42",        "RL baseline",            "broader"),
    ("clip1_lam005_s42",       "clip λ=0.05",            "broader"),
    ("clip1_lam010_s42",       "clip λ=0.10",            "broader"),
    ("clip1_lam020_s42",       "clip λ=0.20",            "broader"),
    ("v21_lam005_s42_unclip",  "v2.1 unclipped λ=0.05",  "broader"),
    ("grpo_v2_s42",            "GRPO-v2 s42",            "sharper"),
    ("grpo_v2_s43",            "GRPO-v2 s43",            "sharper"),
    ("clip1_lam050_s42",       "clip λ=0.50",            "sharper"),
    ("clip1_lam085_s42",       "clip λ=0.85",            "sharper"),
    ("clip1_lam100_s42",       "pure OPD λ=1",           "sharper"),
    ("v21_lam100_s42_unclip",  "v2.1 unclipped λ=1",     "sharper"),
]
CATS = ["attn_qkv", "attn_o", "mlp_in", "mlp_down", "embed"]
CAT_LABEL = {"attn_qkv": "attn (q/k/v)", "attn_o": "attn (out)",
             "mlp_in": "mlp (in)", "mlp_down": "mlp (down)",
             "embed": "embed + lm_head"}
TIER_COLOR = {"broader": "#1b6ca8", "sharper": "#c0392b"}


def load_ratios():
    rows = []
    for label, display, tier in ORDER:
        f = f"/project/inniang/research/figs/dtheta/effrank_full/effrank_{label}.json"
        if not os.path.exists(f):
            continue
        j = json.load(open(f))
        by_cat = defaultdict(list)
        by_cat_base = defaultdict(list)
        for k, r in j["per_tensor"].items():
            cat = r["category"]
            er = r.get("effective_rank")
            ber = r.get("base_effective_rank")
            if er is not None and er == er and ber is not None and ber == ber and ber > 0:
                by_cat[cat].append(er)
                by_cat_base[cat].append(ber)
        ratios = {}
        for c in CATS:
            if by_cat[c] and by_cat_base[c]:
                ratios[c] = (sum(by_cat[c]) / len(by_cat[c])) / (sum(by_cat_base[c]) / len(by_cat_base[c]))
            else:
                ratios[c] = float("nan")
        rows.append((display, tier, ratios))
    return rows


def main():
    rows = load_ratios()
    if not rows:
        print("no data")
        return

    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
        "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 11,
    })

    # Single panel: line per submodule, x = arm
    fig, ax = plt.subplots(figsize=(12, 5.6), constrained_layout=True)
    labels = [r[0] for r in rows]
    x = np.arange(len(labels))

    cat_colors = {"attn_qkv": "#2e8b57", "attn_o": "#79c270",
                  "mlp_in": "#7d3c98", "mlp_down": "#aa84c6",
                  "embed":  "#c0392b"}

    for c in CATS:
        y = [r[2].get(c, float("nan")) for r in rows]
        ax.plot(x, y, marker="o", lw=2.0, ms=7, color=cat_colors[c],
                label=CAT_LABEL[c])

    ax.axhline(1.0, ls="--", color="#555", lw=1.0, alpha=0.7)
    ax.text(len(labels) - 0.4, 1.003, "base eff_rank", color="#555",
            fontsize=10, va="bottom", ha="right")

    # Shade broader/sharper tiers
    tiers = [r[1] for r in rows]
    n_broad = sum(1 for t in tiers if t == "broader")
    ax.axvspan(-0.5, n_broad - 0.5, alpha=0.05, color=TIER_COLOR["broader"])
    ax.axvspan(n_broad - 0.5, len(labels) - 0.5, alpha=0.05, color=TIER_COLOR["sharper"])
    ax.text((n_broad - 1) / 2, 1.105, "broader tier  (§6.1 dynamic: sharp cliff at p≈85%)",
            color=TIER_COLOR["broader"], fontsize=10, ha="center", va="top", weight="bold")
    ax.text((n_broad + len(labels) - 1) / 2, 1.105,
            "sharper tier  (§6.1 dynamic: gradual decline)",
            color=TIER_COLOR["sharper"], fontsize=10, ha="center", va="top", weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylim(0.92, 1.12)
    ax.set_ylabel("effective rank ratio  (ΔW mean) / (base mean)")
    ax.set_title("Full per-tensor effrank pass: tier separation is small & confined to attention", pad=14)
    ax.grid(alpha=0.3, axis="y", linewidth=0.6)
    ax.legend(loc="upper left", frameon=False, ncol=1, title="submodule")

    fig.text(0.01, -0.04,
             "Source: effective rank of ΔW for every 2D weight tensor (48 attn_qkv, 16 attn_o, 32 mlp_in, "
             "16 mlp_down, 2 embed), averaged within submodule. Compare to figs/dtheta/exp3_attn_qkv_effrank_ratio.png "
             "which used only top-10-by-frob (an artifact-prone slice).",
             ha="left", va="top", fontsize=8.5, color="#555", style="italic")

    out = "/project/inniang/research/figs/dtheta/exp3_effrank_full.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"wrote {out}")

    # print summary table
    print(f"\n{'arm':<24} | tier     | " + " | ".join(f'{CAT_LABEL[c]:>12}' for c in CATS))
    print('-' * 110)
    for display, tier, ratios in rows:
        cells = " | ".join(f"{ratios.get(c, float('nan')):>12.3f}" for c in CATS)
        print(f"{display:<24} | {tier:<8} | {cells}")


if __name__ == "__main__":
    main()
