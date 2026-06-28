"""
plot_prune_curves — Exp 3 follow-up figure. Reads research/figs/dtheta/prune_sweep_*/
and renders pass@1 + pass@16 vs prune fraction per arm, overlaid.
"""
from __future__ import annotations
import argparse, glob, json, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(d):
    groups = defaultdict(dict)
    meta = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        j = json.load(open(f))
        name = os.path.basename(f).replace(".json", "")
        arm, p_tag = name.rsplit("_p", 1)
        p = int(p_tag) / 100.0
        m06 = j["metrics_by_temp"]["T=0.6"]
        groups[arm][p] = {
            "p1": m06["eval/pass@1"], "p16": m06.get("eval/pass@16", float("nan")),
            "ent": m06.get("eval/token_entropy", float("nan")),
            "pruned_frac_total": j["prune_stats"].get("actual_prune_frac_of_total", 0.0),
            "pruned_frac_moved": j["prune_stats"].get("actual_prune_frac_of_moved", 0.0),
            "frac_moved": (1 - j["prune_stats"]["n_zero"] / j["prune_stats"]["n_total"])
                            if j["prune_stats"]["n_total"] else float("nan"),
        }
    return dict(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="figs/dtheta/prune_sweep_72097")
    ap.add_argument("--out-prefix", default="figs/dtheta/exp3_prune_curves")
    args = ap.parse_args()

    runs = load(args.dir)
    # Canonical order: broader tier (RL baseline, clip1 low-λ) vs sharper tier (GRPO, pure OPD).
    arm_order = ["rl_baseline_s42", "grpo_v2_s42", "clip1_lam010_s42", "clip1_lam100_s42"]
    arm_label = {
        "rl_baseline_s42":   "RL baseline (broader)",
        "grpo_v2_s42":       "GRPO-v2 (sharper)",
        "clip1_lam010_s42":  "clip1 λ=0.10 (broader)",
        "clip1_lam100_s42":  "clip1 λ=1.00 / pure OPD (dead)",
    }
    arm_color = {
        "rl_baseline_s42":   "#1f77b4",   # blue
        "grpo_v2_s42":       "#d62728",   # red
        "clip1_lam010_s42":  "#2ca02c",   # green
        "clip1_lam100_s42":  "#9467bd",   # purple
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for arm in arm_order:
        if arm not in runs: continue
        d = runs[arm]
        ps = sorted(d.keys())
        # x-axis: actual fraction of TOTAL params reverted (more comparable across arms than nominal pct)
        x = [d[p]["pruned_frac_total"] * 100 for p in ps]
        y1 = [d[p]["p1"] for p in ps]
        y16 = [d[p]["p16"] for p in ps]
        col = arm_color[arm]
        axes[0].plot(x, y1, "o-", color=col, lw=2, label=arm_label[arm])
        axes[1].plot(x, y16, "o-", color=col, lw=2, label=arm_label[arm])

    axes[0].set_xlabel("% of TOTAL params reverted to base init")
    axes[0].set_ylabel("pass@1 (T=0.6, gsm_symbolic)")
    axes[0].set_title("Exp 3 — prune-degradation: pass@1 vs prune fraction")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=9, frameon=False)
    axes[1].set_xlabel("% of TOTAL params reverted to base init")
    axes[1].set_ylabel("pass@16 (T=0.6, gsm_symbolic)")
    axes[1].set_title("Exp 3 — prune-degradation: pass@16 vs prune fraction")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=9, frameon=False)

    out1 = args.out_prefix + ".png"
    fig.savefig(out1, dpi=150)
    print(f"wrote {out1}")

    # also: a "pruning the moved-subset" x-axis variant
    fig2, ax = plt.subplots(1, 1, figsize=(7.5, 5), constrained_layout=True)
    for arm in arm_order:
        if arm not in runs: continue
        d = runs[arm]
        ps = sorted(d.keys())
        x = [d[p]["pruned_frac_moved"] * 100 for p in ps]
        y1 = [d[p]["p1"] for p in ps]
        col = arm_color[arm]
        ax.plot(x, y1, "o-", color=col, lw=2, label=arm_label[arm])
    ax.set_xlabel("% of MOVED weights (|Δθ|>0) reverted to base (by smallest |Δθ|)")
    ax.set_ylabel("pass@1 (T=0.6, gsm_symbolic)")
    ax.set_title("Exp 3 — prune-degradation, x-axis on the moved-weights subset")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, frameon=False)
    out2 = args.out_prefix + "_moved_x.png"
    fig2.savefig(out2, dpi=150)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
