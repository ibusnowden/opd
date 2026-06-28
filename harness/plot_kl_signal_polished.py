"""
plot_kl_signal_polished — TM-style polished §8.2 figure. Two-row contrast:
unclipped (71208, seed-43) collapses; clipped (71395, 3-seed mean ± min/max band)
recovers. Reads only the measured npz files; no fabricated values.

Inputs:
  research/figs/kl_signal_71208.npz           (unclipped, seed-43 λ sweep)
  research/figs/kl_signal_71395_clipped.npz   (clipped, 3 seeds × 4 λ values)

Output:
  research/figs/exp4_kl_signal_mechanism_polished.png
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

UNCLIPPED = "/project/inniang/research/figs/kl_signal_71208.npz"
CLIPPED = "/project/inniang/research/figs/kl_signal_71395_clipped.npz"
OUT = "/project/inniang/research/figs/exp4_kl_signal_mechanism_polished.png"


def smooth(x: np.ndarray, w: int = 25) -> np.ndarray:
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


def main():
    z_u = np.load(UNCLIPPED)
    z_c = np.load(CLIPPED)
    lams = [0.05, 0.10, 0.20, 0.35]
    seeds_c = [43, 44, 45]
    # Cool palette — low λ in blue, high λ in green
    cmap = plt.cm.viridis
    colors = {lam: cmap(0.15 + i * 0.22) for i, lam in enumerate(lams)}

    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
        "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.2), constrained_layout=True,
                              sharex="col")

    # Column 0: kl_signal/p99
    # Column 1: reward/accuracy
    panels = [
        ("kl_signal/p99",    "per-token KL signal: p99 across moved tokens"),
        ("reward/accuracy",  "rollout accuracy on gsm_symbolic"),
    ]

    for col, (key, title) in enumerate(panels):
        ax_u = axes[0, col]
        ax_c = axes[1, col]

        # --- unclipped row ---
        for lam in lams:
            k = f"lam{lam}/{key}"
            if k not in z_u.files:
                continue
            y = z_u[k]
            if y.size == 0:
                continue
            x = np.arange(len(y))
            ax_u.plot(x, smooth(y, 25), color=colors[lam], lw=2.2,
                      label=f"λ = {lam:.2f}")

        ax_u.grid(alpha=0.3, linewidth=0.6)
        if col == 0:
            ax_u.set_yscale("log")
            ax_u.set_ylabel("UNCLIPPED\n(SLURM 71208, seed-43)",
                            fontsize=11, color="#444")
        ax_u.set_title(f"unclipped — {title}", fontsize=11.5)

        # --- clipped row ---
        for lam in lams:
            ys = []
            for s in seeds_c:
                kk = f"s{s}_lam{lam}/{key}"
                if kk in z_c.files:
                    ys.append(z_c[kk])
            if not ys:
                continue
            L = min(len(y) for y in ys)
            stack = np.stack([y[:L] for y in ys])
            mean = smooth(stack.mean(axis=0), 25)
            lo = smooth(stack.min(axis=0), 25)
            hi = smooth(stack.max(axis=0), 25)
            x = np.arange(L)
            ax_c.fill_between(x, lo, hi, color=colors[lam], alpha=0.18)
            ax_c.plot(x, mean, color=colors[lam], lw=2.2,
                      label=f"λ = {lam:.2f}")

        ax_c.grid(alpha=0.3, linewidth=0.6)
        ax_c.set_xlabel("training step")
        if col == 0:
            ax_c.set_yscale("log")
            ax_c.set_ylabel("CLIPPED  (clip = 1.0)\n(SLURM 71395, seeds 43/44/45)",
                            fontsize=11, color="#444")
        ax_c.set_title(f"clipped — {title}", fontsize=11.5)

        # Share y-limits between unclipped and clipped panels in same column
        ymin = min(ax_u.get_ylim()[0], ax_c.get_ylim()[0])
        ymax = max(ax_u.get_ylim()[1], ax_c.get_ylim()[1])
        ax_u.set_ylim(ymin, ymax)
        ax_c.set_ylim(ymin, ymax)

    # Reference line at clip = 1.0 in the p99 panels
    for ax in (axes[0, 0], axes[1, 0]):
        ax.axhline(1.0, ls="--", color="#888", lw=1.2, alpha=0.8)
    axes[1, 0].text(20, 1.05, "clip = 1.0", color="#666", fontsize=10, va="bottom")

    # Legend (one, rightmost top panel)
    axes[0, 1].legend(loc="upper right", frameon=False, title="teacher dose")

    fig.suptitle("Unclipped per-token KL drives the collapse; clipping at |kl|≤1 recovers it",
                 fontsize=14, y=1.02)

    # Footer source caption
    fig.text(0.01, -0.04,
             "Source: kl_signal/p99 and reward/accuracy per-step traces extracted from offline W&B binaries "
             "(SLURM 71208 unclipped seed-43, SLURM 71395 clipped seeds 43/44/45). Lines smoothed with a 25-step "
             "moving average; shaded bands are min/max across the 3 clipped seeds.",
             ha="left", va="top", fontsize=8.5, color="#555", style="italic")

    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
