#!/usr/bin/env bash
# run_exp81_scale_warmstart.sh — §8.1 scale/capability test (the falsification check for the L3 reversal).
#
# L3 (§8.1 follow-up) showed on-policy OPD is dead (p@1 0.008) while off-policy reverse-KD is alive
# (0.298) — at the 1B-SFT/7B-Instruct gap, where the COLD student's on-policy rollouts are pass@1≈0
# (incoherent). The mechanism predicts this should INVERT once the student is competent enough that
# its own rollouts reach answer-shaped regions. We test it by WARM-STARTING the 1B student from
# sft-rft-correct (p@1 0.388 — answer-shaped on-policy rollouts, and still BELOW the teacher's 0.46
# so OPD has upward headroom), holding teacher (7B-Instruct), objective (reverse-KL, clip=null),
# LR (1e-5), and #steps (500) fixed, and running BOTH arms from that init:
#   on-policy OPD (student states)      — does it revive from the 0.008 dead corner?
#   off-policy reverse-KD (teacher states, control) — does it stay strong from a good init too?
# Cold-start baselines (on 0.008 / off 0.298) come from results/exp81_offpolicy_revkd_99626/.
#
# Prediction: if warm on-policy OPD ≫ 0.008 (revives) and ≈/≥ warm off-policy, on-policy-ness flips
# from liability back to virtue at adequate competence — the post's headline re-emerges, and the L3
# reversal is confirmed to be a capacity-gap effect, not a universal one.
#
#   sbatch /project/inniang/research/harness/run_exp81_scale_warmstart.sh
#
#SBATCH --job-name=exp81-scale-warmstart
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:h100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=20:00:00
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
CFG=harness/configs/exp81_offpolicy_revkd.yaml     # base: teacher=7B-Instruct, clip=null, 500 steps, lr 1e-5
CKPT=$ROOT/harness/checkpoints
RES_DIR=$ROOT/results/exp81_scale_warmstart_${SLURM_JOB_ID:-local}
ALL=$ROOT/rft_data/gsm_symbolic_from_OLMo-2-1124-7B-Instruct_seed4242_ALL.jsonl
mkdir -p "$RES_DIR"

echo "[scale] node=$(hostname) job=${SLURM_JOB_ID:-local$$}"
echo "[scale] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# ---------------------------------------------------------------------------
# Step 0 — fail-fast: warm-start ckpts + teacher rollouts exist + config validates.
# ---------------------------------------------------------------------------
for s in 42 43; do
  [ -d "$CKPT/sft-rft-correct-s$s" ] || { echo "[scale] FATAL: missing warm-start init $CKPT/sft-rft-correct-s$s"; exit 1; }
done
[ -s "$ALL" ] || { echo "[scale] FATAL: missing teacher rollouts $ALL"; exit 1; }
$PY - <<'PYEOF'
from harness.config import load_config
cfg = load_config("harness/configs/exp81_offpolicy_revkd.yaml")
assert cfg.lam == 1.0 and cfg.per_token_kl_clip is None and cfg.teacher.model_name.endswith("7B-Instruct")
from harness import unified_trainer  # import smoke
print(f"[scale] base config OK: teacher={cfg.teacher.model_name} clip={cfg.per_token_kl_clip} steps={cfg.num_steps}")
PYEOF

# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
train_onpolicy_opd () {  # $1=seed  (warm-start init = sft-rft-correct-s$1; on-policy OPD: alpha=1, no offpolicy states)
  local s="$1" run="opd-warm-instruct7b-s$1"
  echo "[scale] TRAIN on-policy OPD (warm) seed=$s init=sft-rft-correct-s$s -> $run"
  $PY -m harness.unified_trainer --config "$CFG" \
    --set model_name="$CKPT/sft-rft-correct-s$s" \
    --set alpha=1.0 --set offpolicy_teacher_states=null \
    --set seed="$s" --set wandb_run_name="$run" --set ckpt_dir="$CKPT/$run"
}

train_offpolicy_revkd () {  # $1=seed  (warm-start init; off-policy revKD: alpha=0 + offpolicy states from yaml)
  local s="$1" run="revkd-warm-instruct7b-s$1"
  echo "[scale] TRAIN off-policy revKD (warm) seed=$s init=sft-rft-correct-s$s -> $run"
  $PY -m harness.unified_trainer --config "$CFG" \
    --set model_name="$CKPT/sft-rft-correct-s$s" \
    --set seed="$s" --set wandb_run_name="$run" --set ckpt_dir="$CKPT/$run"
}

eval_ckpt () {  # $1=ckpt  $2=label
  echo "[scale] EVAL $2  ($1)"
  $PY -m harness.eval_passk --ckpt "$1" --task gsm_symbolic \
    --n-prompts 64 --n-samples 16 --k 1,2,4,8,16 --temps 0.6 \
    --max-new-tokens 1024 --eval-seed 1000000 --no-self-bleu --gen-batch-size 8 \
    --out "$RES_DIR/eval_$2.json"
}

# ---------------------------------------------------------------------------
# Step 1 — warm-init baseline (re-eval sft-rft-correct so the table is apples-to-apples).
# Step 2 — train, interleaved by seed so a complete s42 comparison lands first (headline arm first).
# ---------------------------------------------------------------------------
eval_ckpt "$CKPT/sft-rft-correct-s42" "warm_init_s42"
eval_ckpt "$CKPT/sft-rft-correct-s43" "warm_init_s43"

train_onpolicy_opd 42
train_offpolicy_revkd 42
train_onpolicy_opd 43
train_offpolicy_revkd 43

# ---------------------------------------------------------------------------
# Step 3 — eval the four trained checkpoints at the §8.1 matched protocol.
# ---------------------------------------------------------------------------
eval_ckpt "$CKPT/opd-warm-instruct7b-s42"   "opd_warm_s42"
eval_ckpt "$CKPT/revkd-warm-instruct7b-s42" "revkd_warm_s42"
eval_ckpt "$CKPT/opd-warm-instruct7b-s43"   "opd_warm_s43"
eval_ckpt "$CKPT/revkd-warm-instruct7b-s43" "revkd_warm_s43"

# ---------------------------------------------------------------------------
# Step 4 — summary table (vs the cold-start L3 baselines: on 0.008 / off 0.298).
# ---------------------------------------------------------------------------
echo "[scale] SUMMARY"
$PY - "$RES_DIR" <<'PYEOF'
import json, os, sys
res = sys.argv[1]
def m(label):
    p = os.path.join(res, f"eval_{label}.json")
    if not os.path.exists(p): return None
    return json.load(open(p)).get("metrics_by_temp", {}).get("T=0.6")
print(f"{'arm':<22}{'p@1':>8}{'p@16':>8}{'tok_ent':>9}")
for label in ["warm_init_s42","warm_init_s43","opd_warm_s42","opd_warm_s43","revkd_warm_s42","revkd_warm_s43"]:
    t = m(label)
    if t is None: print(f"{label:<22}{'MISSING':>8}"); continue
    print(f"{label:<22}{t.get('eval/pass@1',float('nan')):>8.4f}{t.get('eval/pass@16',float('nan')):>8.4f}{t.get('eval/token_entropy',float('nan')):>9.4f}")
print("cold-start L3 baselines: on-policy OPD 0.008 / off-policy revKD 0.298 (results/exp81_offpolicy_revkd_99626/)")
PYEOF

echo "[scale] DONE. results in $RES_DIR"
ls -la "$RES_DIR"
