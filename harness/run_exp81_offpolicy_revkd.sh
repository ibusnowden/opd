#!/usr/bin/env bash
# run_exp81_offpolicy_revkd.sh — §8.1 follow-up (L3): off-policy reverse-KD vs on-policy OPD.
#
# Isolates on-policy-ness ALONE (the confound L2/§8.1 could not remove). Holds the teacher
# (7B-Instruct) and the per-token reverse-KL objective (clip=null) FIXED and changes ONLY the states
# the loss lands on:
#   off-policy revKD : reverse-KL on the TEACHER's own rollouts (teacher-sampled states)  [trained here]
#   on-policy OPD    : reverse-KL on STUDENT rollouts (opd-instruct7b-s42/s43, p@1 ~0.005) [re-eval'd]
# Everything else (1B-SFT init, lr 1e-5, 8x8 seqs/step, 500 steps) is identical to the on-policy arm.
#
#   sbatch /project/inniang/research/harness/run_exp81_offpolicy_revkd.sh
#
#SBATCH --job-name=exp81-offpolicy-revkd
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:h100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
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
CFG=harness/configs/exp81_offpolicy_revkd.yaml
CKPT_DIR=$ROOT/harness/checkpoints
RES_DIR=$ROOT/results/exp81_offpolicy_revkd_${SLURM_JOB_ID:-local}
ALL=$ROOT/rft_data/gsm_symbolic_from_OLMo-2-1124-7B-Instruct_seed4242_ALL.jsonl
mkdir -p "$RES_DIR"

echo "[revkd] node=$(hostname) job=${SLURM_JOB_ID:-local$$}"
echo "[revkd] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# ---------------------------------------------------------------------------
# Step 0 — cheap fail-fast: teacher rollouts exist + the config validates (catches
# import/config/plumbing errors in seconds, before any GPU/model-load work).
# ---------------------------------------------------------------------------
[ -s "$ALL" ] || { echo "[revkd] FATAL: missing teacher rollouts $ALL"; exit 1; }
$PY - <<'PYEOF'
from harness.config import load_config
cfg = load_config("harness/configs/exp81_offpolicy_revkd.yaml")
assert cfg.recipe == "meta_a0_l1_same_family", cfg.recipe
assert cfg.offpolicy_teacher_states and cfg.lam == 1.0 and cfg.per_token_kl_clip is None
from harness import unified_trainer  # import-time smoke for the off-policy helpers
print(f"[revkd] config OK: recipe={cfg.recipe} teacher={cfg.teacher.model_name} "
      f"clip={cfg.per_token_kl_clip} steps={cfg.num_steps} states={cfg.offpolicy_teacher_states}")
PYEOF

# ---------------------------------------------------------------------------
# Step 1 — train off-policy reverse-KD, seeds 42 & 43 (matched to opd-instruct7b).
# ---------------------------------------------------------------------------
train_revkd () {  # $1=seed
  local seed="$1" run="offpolicy-revkd-instruct7b-s$1"
  echo "[revkd] STEP 1: train seed=$seed -> $run"
  $PY -m harness.unified_trainer --config "$CFG" \
    --set seed="$seed" \
    --set wandb_run_name="$run" \
    --set ckpt_dir="$CKPT_DIR/$run"
}

train_revkd 42
train_revkd 43

# ---------------------------------------------------------------------------
# Step 2 — eval pass@k at the §8.1 matched protocol (T=0.6, 64x16, eval-seed 1e6)
# on the new off-policy revKD ckpts AND re-eval the on-policy OPD-Instruct ckpts.
# ---------------------------------------------------------------------------
eval_ckpt () {  # $1=ckpt  $2=label
  local ckpt="$1" label="$2"
  echo "[revkd] STEP 2: eval $label  ($ckpt)"
  $PY -m harness.eval_passk \
    --ckpt "$ckpt" --task gsm_symbolic \
    --n-prompts 64 --n-samples 16 --k 1,2,4,8,16 --temps 0.6 \
    --max-new-tokens 1024 --eval-seed 1000000 --no-self-bleu \
    --gen-batch-size 8 \
    --out "$RES_DIR/eval_${label}.json"
}

eval_ckpt "$CKPT_DIR/offpolicy-revkd-instruct7b-s42" "revkd_offpolicy_s42"
eval_ckpt "$CKPT_DIR/offpolicy-revkd-instruct7b-s43" "revkd_offpolicy_s43"
eval_ckpt "$CKPT_DIR/opd-instruct7b-s42"             "opd_onpolicy_s42"
eval_ckpt "$CKPT_DIR/opd-instruct7b-s43"             "opd_onpolicy_s43"

# ---------------------------------------------------------------------------
# Step 3 — summarize p@1 / p@16 into one table for the §8.1 L3 row.
# ---------------------------------------------------------------------------
echo "[revkd] STEP 3: summary"
$PY - "$RES_DIR" <<'PYEOF'
import json, os, sys
res = sys.argv[1]
def m(label):  # return the T=0.6 metrics dict for an arm, or None
    p = os.path.join(res, f"eval_{label}.json")
    if not os.path.exists(p): return None
    d = json.load(open(p))
    return d.get("metrics_by_temp", {}).get("T=0.6")
print(f"{'arm':<26}{'p@1':>8}{'p@16':>8}{'tok_ent':>9}")
for label in ["revkd_offpolicy_s42","revkd_offpolicy_s43","opd_onpolicy_s42","opd_onpolicy_s43"]:
    t = m(label)
    if t is None:
        print(f"{label:<26}{'MISSING':>8}"); continue
    print(f"{label:<26}{t.get('eval/pass@1',float('nan')):>8.4f}"
          f"{t.get('eval/pass@16',float('nan')):>8.4f}{t.get('eval/token_entropy',float('nan')):>9.4f}")
PYEOF

echo "[revkd] DONE. results in $RES_DIR"
ls -la "$RES_DIR"
