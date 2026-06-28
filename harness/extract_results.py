"""extract_results — parse `[harness] step …` lines + `eval @ step …` lines from the per-arm log
files of an opd-different-teachers / opd-lambda-sweep / rl-sweep job, emit per-arm CSVs of the
training traces + a JSON of the held-out eval points, and draw the standard plots referenced from
`research/RESULTS.md` (reward vs step, rev_kl vs step, token_entropy vs step, pass@k vs k @ final).

Pure log parsing — no model load, no GPU.  Run from research/:
    python -m harness.extract_results --jobid 71148 --out-dir results/exp1_opd_teachers
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict


_STEP_RE = re.compile(
    r"\[harness\] step (?P<step>\d+)/(?P<total>\d+)\s+"
    r"reward=(?P<reward>[+-]?\d+(?:\.\d+)?)\s+"
    r"loss=(?P<loss>[+-]?\d+(?:\.\d+)?)\s+"
    r"(?:rev_kl=(?P<rev_kl>[+-]?\d+(?:\.\d+)?)\s+)?"
    r"grad_norm=(?P<grad_norm>[+-]?\d+(?:\.\d+)?)\s+"
    r"off_pol=(?P<off_pol>[+-]?\d+(?:\.\d+)?)\s+"
    r"(?P<tps>\d+)\s+tok/s"
)
_EVAL_HEAD = re.compile(r"\[harness\] eval @ step (?P<step>\d+)/(?P<total>\d+)\s+\(T=(?P<temp>[\d.]+)\):\s*(?P<rest>.*)$")
_EVAL_KV = re.compile(r"(?P<k>[A-Za-z_@0-9]+)=(?P<v>[+-]?\d+(?:\.\d+)?)")


def parse_log(path: str) -> dict:
    arm = os.path.basename(path).replace(".log", "")
    # strip leading "opdT_<i>_" or similar — keep the arm tag
    arm = re.sub(r"^[A-Za-z]+T?_\d+_", "", arm)
    arm = re.sub(r"_\d+$", "", arm)
    steps: list[dict] = []
    evals: list[dict] = []
    with open(path) as f:
        for line in f:
            m = _STEP_RE.search(line)
            if m:
                d = {k: (float(v) if v is not None else None) for k, v in m.groupdict().items() if k != "step" and k != "total" and k != "tps"}
                d["step"] = int(m.group("step"))
                d["tps"] = int(m.group("tps"))
                steps.append(d)
                continue
            h = _EVAL_HEAD.match(line)
            if h:
                d = {"step": int(h.group("step")), "temp": float(h.group("temp"))}
                for kv in _EVAL_KV.finditer(h.group("rest")):
                    try:
                        d[kv.group("k")] = float(kv.group("v"))
                    except ValueError:
                        pass
                evals.append(d)
    return {"arm": arm, "steps": steps, "evals": evals}


def write_csv(parsed: dict, out_csv: str) -> None:
    if not parsed["steps"]:
        return
    keys = ["step", "reward", "loss", "rev_kl", "grad_norm", "off_pol", "tps"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for s in parsed["steps"]:
            w.writerow({k: s.get(k, "") for k in keys})


def plot_trace(parsed_all: list[dict], metric: str, ylabel: str, out_png: str, *, log_y: bool = False) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[extract_results] skipping plot {out_png}: matplotlib missing ({e})")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    # group seeds: take the arm tag without the trailing seed suffix as the "family"
    by_family: dict[str, list[dict]] = defaultdict(list)
    for p in parsed_all:
        fam = re.sub(r"-s\d+$", "", p["arm"])
        by_family[fam].append(p)
    for fam in sorted(by_family):
        for p in by_family[fam]:
            xs = [s["step"] for s in p["steps"] if metric in s and s[metric] is not None]
            ys = [s[metric] for s in p["steps"] if metric in s and s[metric] is not None]
            if xs:
                ax.plot(xs, ys, label=p["arm"], alpha=0.85, linewidth=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.set_title(f"{ylabel} vs step")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[extract_results] wrote {out_png}")


def plot_passk_at_final(parsed_all: list[dict], out_png: str) -> None:
    """For each arm, take the last eval (step=total) and plot pass@k vs k."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[extract_results] skipping plot {out_png}: matplotlib missing ({e})")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for p in sorted(parsed_all, key=lambda x: x["arm"]):
        if not p["evals"]:
            continue
        last = max(p["evals"], key=lambda d: d["step"])
        ks, vs = [], []
        for k, v in last.items():
            if isinstance(k, str) and k.startswith("pass@"):
                try:
                    ks.append(int(k.split("@", 1)[1])); vs.append(float(v))
                except Exception:
                    pass
        order = sorted(range(len(ks)), key=lambda i: ks[i])
        if ks:
            ax.plot([ks[i] for i in order], [vs[i] for i in order], marker="o", label=p["arm"], alpha=0.85)
    ax.set_xlabel("k")
    ax.set_xscale("log", base=2)
    ax.set_ylabel("pass@k (held-out)")
    ax.set_title("pass@k @ final step (T=0.6)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[extract_results] wrote {out_png}")


def main_cli() -> None:
    ap = argparse.ArgumentParser(description="Parse per-arm SLURM logs into CSV traces + plots.")
    ap.add_argument("--jobid", type=int, required=True, help="SLURM job id to parse (matches harness/logs/*_<jobid>.log)")
    ap.add_argument("--log-glob", default=None, help="optional explicit glob (default: harness/logs/*_<jobid>.log)")
    ap.add_argument("--out-dir", default=None, help="default: results/exp_<jobid>")
    args = ap.parse_args()

    pat = args.log_glob or f"harness/logs/*_{args.jobid}.log"
    paths = sorted(p for p in glob.glob(pat) if os.path.isfile(p))
    if not paths:
        raise SystemExit(f"no per-arm logs matched {pat!r}")
    out = args.out_dir or f"results/exp_{args.jobid}"
    os.makedirs(out, exist_ok=True)

    parsed_all: list[dict] = []
    for p in paths:
        parsed = parse_log(p)
        parsed_all.append(parsed)
        write_csv(parsed, os.path.join(out, f"{parsed['arm']}.csv"))

    # JSON dump of the held-out eval points across arms
    eval_dump = {p["arm"]: p["evals"] for p in parsed_all}
    with open(os.path.join(out, "evals.json"), "w") as f:
        json.dump(eval_dump, f, indent=2)

    # Plots
    plot_trace(parsed_all, "reward", "rollout reward (acc + 0.5·fmt)", os.path.join(out, "reward_over_steps.png"))
    plot_trace(parsed_all, "rev_kl", "teacher reverse-KL  log π_rollout − log π_T", os.path.join(out, "revkl_over_steps.png"))
    plot_trace(parsed_all, "grad_norm", "grad_norm (pre-clip)", os.path.join(out, "gradnorm_over_steps.png"))
    plot_passk_at_final(parsed_all, os.path.join(out, "passk_at_final.png"))

    print(f"[extract_results] {len(parsed_all)} arms parsed; arms = {[p['arm'] for p in parsed_all]}")
    print(f"[extract_results] wrote -> {out}/")


if __name__ == "__main__":
    main_cli()
