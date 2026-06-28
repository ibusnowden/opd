#!/usr/bin/env bash
# run_exp9_train_prm.sh — Stage 1+2: train the step-level PRM for variant (b).
#
# Stage 1 (generate): 7B-SFT teacher rollouts on gsm_symbolic, segment + label each step using
#   metadata.variables (ground-truth intermediate values). Saves JSONL of (prompt, completion,
#   step_token_spans, step_labels) to rft_data/prm_teacher_b/prm_train.jsonl.
# Stage 2 (train): fine-tune OLMo-2-0425-1B-SFT with a scalar head (BCE on step-boundary tokens)
#   to predict P(step correct | trajectory prefix). Saves to harness/checkpoints/prm_step_level_s4242/.
#
# The PRM is then consumed by harness/teachers.py::PRMTeacher as prm_source="trained" in Exp 9.
#
#   sbatch /project/inniang/research/harness/run_exp9_train_prm.sh
#
# 1×H100, ~3h total (2h generate + 1h train). Outputs:
#   rft_data/prm_teacher_b/prm_train.jsonl  (labeled trajectories)
#   harness/checkpoints/prm_step_level_s4242/prm_head.pt  (trained PRM head)
#   harness/checkpoints/prm_step_level_s4242/base_model_name.txt
#   harness/checkpoints/prm_step_level_s4242/tokenizer files
#SBATCH --job-name=exp9-prm-train
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger04
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=/project/inniang/research/harness/logs/%x-%j.out

set -euo pipefail

RESEARCH_ROOT=/project/inniang/research
PYTHON=/project/inniang/.venv/bin/python
cd "$RESEARCH_ROOT"

export PYTHONPATH="$RESEARCH_ROOT:${PYTHONPATH:-}"
export HF_HOME=/project/inniang/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_PROJECT="${WANDB_PROJECT:-distill-harness}"
export WANDB_DIR="$RESEARCH_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TEACHER="${TEACHER:-allenai/OLMo-2-1124-7B-SFT}"
BASE="${BASE:-allenai/OLMo-2-0425-1B-SFT}"
N_ROLLOUTS="${N_ROLLOUTS:-2000}"
SEED="${SEED:-4242}"
DATA_DIR="$RESEARCH_ROOT/rft_data/prm_teacher_b"
CKPT_DIR="$RESEARCH_ROOT/harness/checkpoints/prm_step_level_s${SEED}"
mkdir -p "$DATA_DIR" "$CKPT_DIR"

echo "[exp9-prm-train] node=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "[exp9-prm-train] teacher=$TEACHER base=$BASE n=$N_ROLLOUTS seed=$SEED"
echo "[exp9-prm-train] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"

# ---------------------------------------------------------------------------
# Stage 1 — generate teacher rollouts + per-step labels.
# ---------------------------------------------------------------------------
echo "[exp9-prm-train] Stage 1: generating $N_ROLLOUTS teacher rollouts + per-step labels"
$PYTHON -m harness.train_prm generate \
    --teacher "$TEACHER" \
    --dataset gsm_symbolic \
    --n "$N_ROLLOUTS" \
    --seed "$SEED" \
    --max_new 1024 \
    --temperature 0.6 \
    --device cuda:0 \
    --out_dir "$DATA_DIR"

DATA_FILE="$DATA_DIR/prm_train.jsonl"
if [[ ! -s "$DATA_FILE" ]]; then
    echo "[exp9-prm-train] FAIL: $DATA_FILE not produced or empty"
    exit 1
fi
N_LINES=$(wc -l < "$DATA_FILE")
echo "[exp9-prm-train] Stage 1 done: $N_LINES labeled trajectories at $DATA_FILE"
if [[ "$N_LINES" -lt 200 ]]; then
    echo "[exp9-prm-train] WARN: only $N_LINES trajectories — PRM may underfit. Consider increasing N_ROLLOUTS."
fi

# ---------------------------------------------------------------------------
# Stage 2 — train the PRM head (1B-SFT base + scalar head, BCE on step-boundary positions).
# ---------------------------------------------------------------------------
echo "[exp9-prm-train] Stage 2: training PRM head (base=$BASE, 3 epochs, lr=1e-5)"
$PYTHON -m harness.train_prm train \
    --base "$BASE" \
    --data "$DATA_FILE" \
    --out "$CKPT_DIR" \
    --epochs 3 \
    --lr 1e-5 \
    --batch 4 \
    --max_len 2048 \
    --seed "$SEED" \
    --device cuda:0

if [[ ! -f "$CKPT_DIR/prm_head.pt" ]]; then
    echo "[exp9-prm-train] FAIL: $CKPT_DIR/prm_head.pt not produced"
    exit 1
fi
echo "[exp9-prm-train] Stage 2 done: PRM head at $CKPT_DIR/prm_head.pt"
echo "[exp9-prm-train] DONE. Use --prm_model_path $CKPT_DIR in exp9_prm_teacher_b.yaml."
ls -la "$CKPT_DIR"
