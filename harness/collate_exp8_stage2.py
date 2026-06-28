#!/usr/bin/env python
# collate_exp8_stage2.py — Exp 8 stage-2 (fixed-hint OPD) matched-seed collation.
#
# The stage-2 runs are launched one-seed-per-SLURM-job, so each job writes its
# eval JSON into a job-id-keyed dir (results/exp8_fixed_hint_opd_<arm>_<jobid>/).
# This script globs ALL such dirs, groups by arm (best/placebo) × seed, and
# prints the matched best-vs-placebo table that the per-task-hint-search-gepa
# experiment hinges on.
#
# Anchors (same λ=1 / clip=1.0 / 7B-SFT teacher / T=0.6 / 64×16 / eval-seed 1e6):
#   §7.7 logit pure-OPD       0.029   (lower bound — no hint, no answer conditioning)
#   §7.10 OPSD answer-cond    0.188 (s42) / 0.252 (4-seed)   (privileged upper bound)
#
# Verdict logic (per the run_exp8_fixed_hint_opd.sh design):
#   placebo mean ≈ best mean  →  the rescue is "any appended conditioning string",
#                                 i.e. the §7.10 rescue is per-problem PRIVILEGE;
#                                 task-level hint search is a dead end at λ=1.
#   best mean ≫ placebo mean  →  the rescue is task-RELEVANT distribution-shift
#                                 onto answer-shaped paths; unbiased (§8.4 frontier).
#
# Usage:
#   python -m harness.collate_exp8_stage2
#   python -m harness.collate_exp8_stage2 --results-glob 'results/exp8_fixed_hint_opd_*'
#
import argparse, glob, json, os, re, sys
from collections import defaultdict


def load_eval(path):
    try:
        d = json.load(open(path))
    except Exception as e:
        return None, f"unreadable ({e})"
    t = d.get("metrics_by_temp", {}).get("T=0.6")
    if t is None:
        return None, "no T=0.6 metrics"
    return {
        "p1": t.get("eval/pass@1"),
        "p16": t.get("eval/pass@16"),
        "tok_ent": t.get("eval/token_entropy"),
        "solved_any": t.get("eval/solved_any"),
    }, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-glob", default="results/exp8_fixed_hint_opd_*",
                    help="glob (relative to RESEARCH root) for stage-2 result dirs")
    ap.add_argument("--anchors", action="store_true", default=True,
                    help="print the §7.7/§7.10 anchor lines")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # research/
    pattern = os.path.join(root, args.results_glob)
    dirs = sorted(glob.glob(pattern))
    # only keep dirs that look like stage-2 (have eval_*_s*.json) and are arms best|placebo
    arm_re = re.compile(r"exp8_fixed_hint_opd_(best|placebo)_\d+$")

    # arm -> { seed -> {metrics..., dir, fname} }
    table = defaultdict(dict)
    for d in dirs:
        base = os.path.basename(d)
        m = arm_re.match(base)
        if not m:
            continue
        arm = m.group(1)
        for f in sorted(glob.glob(os.path.join(d, "eval_*_s*.json"))):
            fm = re.search(r"eval_[a-z]+_s(\d+)\.json", os.path.basename(f))
            if not fm:
                continue
            seed = fm.group(1)
            metrics, err = load_eval(f)
            if metrics is None:
                print(f"[skip] {base}/{os.path.basename(f)}: {err}")
                continue
            table[arm][seed] = {**metrics, "dir": base, "file": os.path.basename(f)}

    arms = sorted(table.keys())
    if not arms:
        print(f"No eval_*_s*.json found under {pattern}", file=sys.stderr)
        sys.exit(1)

    print("=== Exp 8 stage 2 — fixed-hint OPD (matched best-vs-placebo control) ===")
    print("Protocol: pure OPD λ=1, clip=1.0, teacher=7B-SFT + condition_on=fixed_hint, "
          "gsm_symbolic, 500 steps, T=0.6, 64×16, eval-seed 1e6.")
    if args.anchors:
        print("\nAnchors (same protocol):  §7.7 logit pure-OPD 0.029 (lower) | "
              "§7.10 OPSD answer-cond 0.188 (s42) / 0.252 (4-seed) (privileged upper)")
    # print the distinct hint per arm (dirs are per-job, hint.txt is identical across jobs of the same arm)
    seen_arm_hint = {}
    for d in dirs:
        base = os.path.basename(d)
        m = arm_re.match(base)
        if not m:
            continue
        arm = m.group(1)
        hp = os.path.join(d, "hint.txt")
        if os.path.exists(hp):
            h = open(hp).read().strip()
            if arm not in seen_arm_hint:
                seen_arm_hint[arm] = h
    if seen_arm_hint:
        print("\nHints:")
        for arm in sorted(seen_arm_hint):
            print(f"  {arm:8s}: {seen_arm_hint[arm]}")

    print()
    header = f"{'arm/seed':<22}{'p@1':>9}{'p@16':>9}{'tok_ent':>10}{'solved_any':>12}  source"
    print(header)
    print("-" * len(header))
    for arm in arms:
        seeds = sorted(table[arm].keys(), key=lambda s: int(s))
        p1s = []
        for s in seeds:
            r = table[arm][s]
            print(f"{arm + '/s' + s:<22}{r['p1']:>9.4f}{r['p16']:>9.4f}"
                  f"{r['tok_ent']:>10.4f}{r['solved_any']:>12.4f}  {r['dir']}/{r['file']}")
            p1s.append(r["p1"])
        if p1s:
            n = len(p1s)
            mean = sum(p1s) / n
            if n > 1:
                sd = (sum((x - mean) ** 2 for x in p1s) / (n - 1)) ** 0.5
            else:
                sd = float("nan")
            print(f"{arm + ' MEAN':<22}{mean:>9.4f}{'':>9}{'':>10}{'':>12}  (n={n}, sd={sd:.4f})")
        print()

    # verdict
    if "best" in table and "placebo" in table and table["best"] and table["placebo"]:
        b = [table["best"][s]["p1"] for s in table["best"]]
        p = [table["placebo"][s]["p1"] for s in table["placebo"]]
        bmean = sum(b) / len(b)
        pmean = sum(p) / len(p)
        diff = bmean - pmean
        print("=== Verdict ===")
        print(f"best mean p@1   = {bmean:.4f} (n={len(b)}, seeds={sorted(table['best'].keys())})")
        print(f"placebo mean    = {pmean:.4f} (n={len(p)}, seeds={sorted(table['placebo'].keys())})")
        print(f"best − placebo  = {diff:+.4f}")
        print()
        if len(b) < 3 or len(p) < 3:
            print(f"[caution] seed counts best={len(b)}, placebo={len(p)}; the design's "
                  "bimodality guard wants ≥3 per arm. Treat with care.")
        # thresholds from the run script: <0.08 PRIVILEGE, >0.13 DISTRIBUTION-SHIFT
        # but the LOAD-BEARING comparison is best vs placebo, not vs absolute anchors:
        if abs(diff) < 0.04 and pmean >= 0.10:
            print("→ placebo ≈ best: the rescue is 'any appended conditioning string', NOT "
                  "task-relevant distribution-shift. The §7.10 rescue is per-problem PRIVILEGE; "
                  "task-level hint search is a DEAD END at λ=1 (unbiased, but doesn't reach "
                  "answer-shaped paths).")
        elif diff > 0.04 and bmean >= 0.13:
            print("→ best ≫ placebo: the rescue is task-RELEVANT distribution-shift onto "
                  "answer-shaped paths; this version keeps the §8.4 unbiasedness leg → a real "
                  "step toward dense+on-policy+un-privileged.")
        else:
            print("→ AMBIGUOUS: best and placebo differ but not cleanly (best mean "
                  f"{bmean:.4f} vs placebo {pmean:.4f}). Inspect per-seed bimodality before "
                  "claiming either way; consider more seeds.")
    else:
        missing = [a for a in ("best", "placebo") if a not in table or not table[a]]
        print(f"[incomplete] missing arm(s): {missing}. Re-run once the jobs finish.")


if __name__ == "__main__":
    main()
