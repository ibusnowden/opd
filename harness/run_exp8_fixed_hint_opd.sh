#!/usr/bin/env bash
# run_exp8_fixed_hint_opd.sh — Exp 8 STAGE 2: pure OPD (lam=1, clip=1.0) from a FIXED, NON-privileged,
# task-level hint teacher (the GEPA-style stage-1 winner). per-task-hint-search-gepa.md.
#
# Question (sharpened by §7.10 + §7.12): §7.10 showed a per-problem ANSWER-conditioned teacher rescues
# pure OPD from the §7.2/§7.7 dead corner (logit λ=1 0.029 → OPSD 0.188 s42 / 0.252 4-seed) — but spends
# the §8.4 unbiasedness leg (privileged info). Does a SEARCHED task-level hint (entry-independent, no
# per-problem leak) do the same?  Read p@1 @ step 500, T=0.6, 64×16, eval-seed 1e6:
#     ≈ 0.03–0.05  → the §7.10 rescue is per-problem PRIVILEGE; a task-level shift can't reach
#                    answer-shaped paths. Hint search is a dead end at λ=1.
#     ≈ 0.15–0.25  → the rescue is distribution-shift onto answer-shaped trajectories GENERALLY, and
#                    THIS version is unbiased → a real step toward the §8.4 frontier (dense+on-policy+un-privileged).
#
# Two arms (submit this script once per arm via env):
#   ARM=best    HINT="<stage-1 winner>"          SEEDS="42 43"   # headline + bimodality guard
#   ARM=placebo HINT="<task-irrelevant string>"  SEEDS="42"      # control: isolates TASK-RELEVANT shift
#                                                                 # from "any appended conditioning string"
# Anchors (same 7B-SFT teacher MODEL, same λ=1/clip=1.0/protocol/eval): §7.7 logit pure-OPD 0.029 (lower),
# §7.10 OPSD answer-conditioned 0.188 / 0.252 (privileged upper). Exp 8 sits the non-privileged hint between.
#
#   ARM=best    HINT="Ensure each step logically follows from the previous, showing all calculations clearly; aim for a simple, straightforward solution." SEEDS="42 43" sbatch /project/inniang/research/harness/run_exp8_fixed_hint_opd.sh
#   ARM=placebo HINT="Respond in a calm, friendly, and encouraging tone."                                                                                  SEEDS="42"    sbatch /project/inniang/research/harness/run_exp8_fixed_hint_opd.sh
#
#SBATCH --job-name=exp8-fixed-hint-opd
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:h100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=/project/inniang/research/harness/logs/%x-%j.out

set -euo pipefail

cd /project/inniang/research
export PYTHONPATH=/project/inniang/research:${PYTHONPATH:-}
export HF_HOME=/project/inniang/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/project/inniang/.venv/bin/python
ROOT=/project/inniang/research
CFG=harness/configs/exp8_fixed_hint_opd.yaml
CKPT_DIR=$ROOT/harness/checkpoints

ARM="${ARM:?set ARM=best|placebo}"
HINT="${HINT:?set HINT=\"<the fixed task-level hint>\"}"
SEEDS="${SEEDS:-42 43}"
JOBID="${SLURM_JOB_ID:-local$$}"
RES_DIR=$ROOT/results/exp8_fixed_hint_opd_${ARM}_${JOBID}
mkdir -p "$RES_DIR"
printf '%s\n' "$HINT" > "$RES_DIR/hint.txt"   # provenance

echo "[exp8-s2] node=$(hostname) job=$JOBID arm=$ARM seeds='$SEEDS'"
echo "[exp8-s2] hint: $HINT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---------------------------------------------------------------------------
# Step 0 — cheap fail-fast: config validates with this hint + routes to the distill loop.
# ---------------------------------------------------------------------------
HINT="$HINT" $PY - "$CFG" <<'PYEOF'
import os, sys, yaml
from harness.config import ResearchConfig
raw = yaml.safe_load(open(sys.argv[1]))
raw["teacher"]["fixed_hint"] = os.environ["HINT"]
cfg = ResearchConfig(**raw)
assert cfg.recipe == "hint_opd", cfg.recipe
assert cfg.alpha == 1.0 and cfg.lam == 1.0 and cfg.per_token_kl_clip == 1.0
assert cfg.teacher.kind == "self" and cfg.teacher.condition_on == "fixed_hint" and cfg.teacher.fixed_hint
from harness import unified_trainer  # import-time smoke
print(f"[exp8-s2] config OK: recipe={cfg.recipe} teacher={cfg.teacher.model_name} "
      f"clip={cfg.per_token_kl_clip} steps={cfg.num_steps} lr={cfg.lr}")
PYEOF

# ---------------------------------------------------------------------------
# Step 1 — train pure hint-OPD per seed (matched to the §7.10 OPSD protocol).
# ---------------------------------------------------------------------------
for seed in $SEEDS; do
  run="hint-opd-${ARM}-s${seed}"
  echo "[exp8-s2] STEP 1: train seed=$seed -> $run"
  $PY -m harness.unified_trainer --config "$CFG" \
    --set teacher.fixed_hint="$HINT" \
    --set seed="$seed" \
    --set wandb_run_name="$run" \
    --set ckpt_dir="$CKPT_DIR/$run"
done

# ---------------------------------------------------------------------------
# Step 2 — eval pass@k at the §7.10 matched protocol (T=0.6, 64×16, eval-seed 1e6).
# ---------------------------------------------------------------------------
for seed in $SEEDS; do
  run="hint-opd-${ARM}-s${seed}"
  echo "[exp8-s2] STEP 2: eval $run"
  $PY -m harness.eval_passk \
    --ckpt "$CKPT_DIR/$run" --task gsm_symbolic \
    --n-prompts 64 --n-samples 16 --k 1,2,4,8,16 --temps 0.6 \
    --max-new-tokens 1024 --eval-seed 1000000 --no-self-bleu \
    --gen-batch-size 8 \
    --out "$RES_DIR/eval_${ARM}_s${seed}.json"
done

# ---------------------------------------------------------------------------
# Step 3 — summarize p@1 / p@16 vs the §7.7 (0.029) + §7.10 (0.188) anchors.
# ---------------------------------------------------------------------------
echo "[exp8-s2] STEP 3: summary"
$PY - "$RES_DIR" "$ARM" "$SEEDS" <<'PYEOF'
import json, os, sys
res, arm, seeds = sys.argv[1], sys.argv[2], sys.argv[3].split()
def m(seed):
    p = os.path.join(res, f"eval_{arm}_s{seed}.json")
    if not os.path.exists(p): return None
    return json.load(open(p)).get("metrics_by_temp", {}).get("T=0.6")
print(f"\n=== Exp 8 stage 2 — hint-OPD (arm={arm}) ===")
print("anchors @ same λ=1/clip=1.0/teacher=7B-SFT/protocol:  §7.7 logit pure-OPD 0.029 (lower) | §7.10 OPSD answer 0.188 (privileged upper)")
print(f"{'arm/seed':<22}{'p@1':>8}{'p@16':>8}{'tok_ent':>9}")
p1s = []
for seed in seeds:
    t = m(seed)
    if t is None:
        print(f"{arm}/s{seed:<14}{'MISSING':>8}"); continue
    p1 = t.get('eval/pass@1', float('nan')); p1s.append(p1)
    print(f"{arm+'/s'+seed:<22}{p1:>8.4f}{t.get('eval/pass@16',float('nan')):>8.4f}"
          f"{t.get('eval/token_entropy',float('nan')):>9.4f}")
if p1s:
    mean = sum(p1s)/len(p1s)
    verdict = ("PRIVILEGE (hint dead end at λ=1)" if mean < 0.08 else
               "DISTRIBUTION-SHIFT suffices (unbiased rescue → §8.4 frontier)" if mean > 0.13 else
               "AMBIGUOUS (between anchors — inspect)")
    print(f"\nmean p@1 = {mean:.4f}  →  reading: {verdict}")
PYEOF

echo "[exp8-s2] DONE. results in $RES_DIR"
ls -la "$RES_DIR"
