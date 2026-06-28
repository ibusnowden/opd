#!/usr/bin/env bash
# run_exp7_armA_only.sh — Exp 7 FAST PATH: just arm A (on-policy pure-OPD) on a 2-GPU non-exclusive
# allocation, so the headline (does on-policy pure-OPD REVIVE at native 7B, or stay in the §7.6 dead
# zone?) lands without waiting for the whole node to drain for the full 4-arm exclusive run (job 115085).
# Identical settings to arm A of run_exp7_scale_7b.sh: alpha=1 lam=1 clip=null, student card0 + teacher
# card1, 250 steps, seed 42, then eval at the §8.1/Exp-5 protocol (T=0.6, 64x16, eval-seed 1e6).
#
#   sbatch /project/inniang/research/harness/run_exp7_armA_only.sh
#
#SBATCH --job-name=exp7-armA
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger01
#SBATCH --gres=gpu:h100_80gb:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --output=/project/inniang/research/harness/logs/%x-%j.out

set -euo pipefail
RESEARCH_ROOT=/project/inniang/research
PYTHON=/project/inniang/.venv/bin/python
cd "$RESEARCH_ROOT"
export PYTHONPATH="$RESEARCH_ROOT:${PYTHONPATH:-}"
export HF_HOME=/project/inniang/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline WANDB_PROJECT="${WANDB_PROJECT:-distill-harness}" WANDB_DIR="$RESEARCH_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="harness/configs/exp7_scale_7b.yaml"
NUM_STEPS="${NUM_STEPS:-250}"
SEED="${SEED:-42}"
JOBID="${SLURM_JOB_ID:-local$$}"
NAME="onpolicy-opd-noclip"
LABEL="${NAME}-s${SEED}-armA-${JOBID}"
CKPT="$RESEARCH_ROOT/harness/checkpoints/$LABEL"
RES_DIR="$RESEARCH_ROOT/results/exp7_armA_${JOBID}"
LOGF="harness/logs/exp7_armA_${NAME}-s${SEED}_${JOBID}.log"
ELOGF="harness/logs/exp7_armA_eval_${NAME}-s${SEED}_${JOBID}.log"
mkdir -p harness/logs "$RES_DIR"

echo "[exp7-armA] node=$(hostname) job=$JOBID steps=$NUM_STEPS seed=$SEED -> $CKPT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# --- train arm A: alpha=1 lam=1 clip=null (== base config), student card0 + teacher card1 ---
CUDA_VISIBLE_DEVICES="0,1" "$PYTHON" -m harness.unified_trainer \
    --config "$CONFIG" \
    --set model_device_id=0 --set teacher.device_id=1 \
    --set seed="$SEED" --set num_steps="$NUM_STEPS" \
    --set alpha=1.0 --set lam=1.0 --set per_token_kl_clip=null \
    --set save_ckpt=true --set ckpt_dir="$CKPT" \
    --set wandb_run_name="$LABEL" \
    > "$LOGF" 2>&1
echo "[exp7-armA] TRAIN done -> $CKPT"

# --- eval at the §8.1/Exp-5 matched protocol ---
echo "[exp7-armA] evaluating (T=0.6, 64x16, k 1..16, eval-seed 1e6)"
CUDA_VISIBLE_DEVICES="0" "$PYTHON" -m harness.eval_passk \
    --ckpt "$CKPT" --task gsm_symbolic \
    --n-prompts 64 --n-samples 16 --k 1,2,4,8,16 --temps 0.6 \
    --max-new-tokens 1024 --eval-seed 1000000 --no-self-bleu --gen-batch-size 8 \
    --out "$RES_DIR/eval_${NAME}_s${SEED}.json" \
    > "$ELOGF" 2>&1
echo "[exp7-armA] EVAL done"

# --- headline ---
"$PYTHON" - "$RES_DIR/eval_${NAME}_s${SEED}.json" <<'PYEOF'
import json, sys
t = json.load(open(sys.argv[1])).get("metrics_by_temp", {}).get("T=0.6", {})
p1, p16, ent = t.get("eval/pass@1"), t.get("eval/pass@16"), t.get("eval/token_entropy")
print(f"[exp7-armA] HEADLINE  p@1={p1}  p@16={p16}  tok_ent={ent}")
print("  1B reference: on-policy pure-OPD was 0.008 (dead zone). off-policy revKD control was 0.298.")
print("  * p@1 >> 0.01 -> on-policy pure-OPD REVIVES at native 7B (1B collapse = capacity artifact).")
print("  * p@1 ~  0.01 -> on-policy reverse-KL instability is SCALE-ROBUST (generalizes §8.1).")
PYEOF

echo "[exp7-armA] DONE. results in $RES_DIR"
ls -la "$RES_DIR"
