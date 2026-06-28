"""
§8.2 mechanism figure: clipped (71395, clip=1.0, 3 seeds × 4 λ) vs
unclipped (71208, seed-43 only, full λ sweep) traces of
`kl_signal/{p99, heavy_tail_frac}` and `reward/accuracy` vs training step.

Story (the §7.7 prediction made concrete):
  - Unclipped low-λ arms develop a *rising* per-token KL p99 and heavy-tail
    fraction over training, and the rollout accuracy collapses around
    step 100-150.
  - Clipped low-λ arms hold p99 below the |kl| ≤ 1 cap by construction,
    keep heavy_tail_frac flat, and the rollout accuracy *recovers* and
    stabilizes around step 150-300.

Inputs:
  research/figs/kl_signal_71208.npz         (seed-43, unclipped)
  research/figs/kl_signal_71395_clipped.npz (seeds 43/44/45, clip=1.0)
Output:
  research/figs/exp4_kl_signal_mechanism.png
"""
from __future__ import annotations
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

UNCLIPPED = "/project/inniang/research/figs/kl_signal_71208.npz"
CLIPPED   = "/project/inniang/research/figs/kl_signal_71395_clipped.npz"
OUT       = "/project/inniang/research/figs/exp4_kl_signal_mechanism.png"


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

    # the 4 λ values we have on BOTH sides:
    lams = [0.05, 0.10, 0.20, 0.35]
    seeds_c = [43, 44, 45]
    cmap = plt.cm.viridis
    colors = {lam: cmap(i / (len(lams) - 1)) for i, lam in enumerate(lams)}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), constrained_layout=True)

    panel_specs = [
        ("kl_signal/p99",             "p99 of teacher KL signal",       True),
        ("kl_signal/heavy_tail_frac", "heavy-tail frac  (|kl| > 5)",    True),
        ("reward/accuracy",           "rollout accuracy",               False),
    ]

    for col, (key, title, log_y) in enumerate(panel_specs):
        # Top row: unclipped (71208 seed-43)
        ax_u = axes[0, col]
        for lam in lams:
            full_key = f"lam{lam}/{key}"
            if full_key not in z_u.files:
                continue
            y = z_u[full_key]
            if y.size == 0:
                continue
            x = np.arange(len(y))
            ax_u.plot(x, smooth(y, 25), color=colors[lam], lw=1.9,
                      label=f"λ={lam:.2f}")
        ax_u.set_title(f"unclipped (71208, s43) — {title}")
        ax_u.grid(alpha=0.3)
        if log_y: ax_u.set_yscale("log")
        if col == 0:
            ax_u.set_ylabel("unclipped")

        # Bottom row: clipped (71395, 3-seed band)
        ax_c = axes[1, col]
        for lam in lams:
            ys = []
            for s in seeds_c:
                k = f"s{s}_lam{lam}/{key}"
                if k in z_c.files:
                    ys.append(z_c[k])
            if not ys:
                continue
            L = min(len(y) for y in ys)
            stack = np.stack([y[:L] for y in ys])
            mean = smooth(stack.mean(axis=0), 25)
            lo = smooth(stack.min(axis=0), 25)
            hi = smooth(stack.max(axis=0), 25)
            x = np.arange(L)
            ax_c.fill_between(x, lo, hi, color=colors[lam], alpha=0.18)
            ax_c.plot(x, mean, color=colors[lam], lw=1.9,
                      label=f"λ={lam:.2f}")
        ax_c.set_title(f"clipped, clip=1.0 (71395, s43/44/45) — {title}")
        ax_c.set_xlabel("training step")
        ax_c.grid(alpha=0.3)
        if log_y: ax_c.set_yscale("log")
        if col == 0:
            ax_c.set_ylabel("clip=1.0")

        # share y-limits row-wise (top & bottom) for fair visual comparison
        if log_y:
            ymin = min(ax_u.get_ylim()[0], ax_c.get_ylim()[0])
            ymax = max(ax_u.get_ylim()[1], ax_c.get_ylim()[1])
            ax_u.set_ylim(ymin, ymax); ax_c.set_ylim(ymin, ymax)
        else:
            ymin = min(ax_u.get_ylim()[0], ax_c.get_ylim()[0])
            ymax = max(ax_u.get_ylim()[1], ax_c.get_ylim()[1])
            ax_u.set_ylim(ymin, ymax); ax_c.set_ylim(ymin, ymax)

    # legends only on the rightmost column to avoid clutter
    axes[0, 2].legend(loc="upper left", fontsize=9, frameon=False)
    axes[1, 2].legend(loc="upper left", fontsize=9, frameon=False)

    fig.suptitle("§8.2 mechanism — per-token KL signal: unclipped collapse vs clipped recovery (gsm_symbolic, 500 steps)",
                 fontsize=12.5)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")

    # contrast table
    print("\nfinal-50-step means:")
    print(f"{'arm':<25} {'p99':>8} {'heavy_tail':>11} {'acc':>8}")
    for lam in lams:
        # unclipped
        u_p99 = float(np.mean(z_u[f"lam{lam}/kl_signal/p99"][-50:]))
        u_ht  = float(np.mean(z_u[f"lam{lam}/kl_signal/heavy_tail_frac"][-50:]))
        u_acc = float(np.mean(z_u[f"lam{lam}/reward/accuracy"][-50:]))
        print(f"unclip s43      λ={lam:.2f}   {u_p99:>8.3f} {u_ht:>11.4f} {u_acc:>8.3f}")
        # clipped (3-seed mean)
        for s in seeds_c:
            c_p99 = float(np.mean(z_c[f"s{s}_lam{lam}/kl_signal/p99"][-50:]))
            c_ht  = float(np.mean(z_c[f"s{s}_lam{lam}/kl_signal/heavy_tail_frac"][-50:]))
            c_acc = float(np.mean(z_c[f"s{s}_lam{lam}/reward/accuracy"][-50:]))
            print(f"clip=1.0 s{s}    λ={lam:.2f}   {c_p99:>8.3f} {c_ht:>11.4f} {c_acc:>8.3f}")
        print()


if __name__ == "__main__":
    main()
