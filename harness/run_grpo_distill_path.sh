#!/usr/bin/env bash
# run_grpo_distill_path.sh — RESULTS.md §7.3 bullet 3: GRPO through `_run_distill_loop` control.
# One arm, λ=0.001 (forces distill path), gsm-specialized teacher, seed 42, 500 steps.
# Compares result to the Exp-1 GRPO baseline pass@1 0.577 (produced by `_run_rl_loop`).
#
#   sbatch /project/inniang/research/harness/run_grpo_distill_path.sh
#
#SBATCH --job-name=grpo-distill-ctrl
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:h100_80gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
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

CONFIG="${CONFIG:-harness/configs/grpo_via_distill_path.yaml}"
NUM_STEPS="${NUM_STEPS:-500}"
SEED="${SEED:-42}"
JOBID="${SLURM_JOB_ID:-local$$}"
mkdir -p harness/logs

label="grpo-distill-seed${SEED}"
logf="harness/logs/${label}_${JOBID}.log"
echo "[grpo-distill] node=$(hostname) job=$JOBID config=$CONFIG steps=$NUM_STEPS seed=$SEED"
echo "[grpo-distill] log=$logf"

"$PYTHON" -m harness.unified_trainer \
    --config "$CONFIG" \
    --set model_device_id=0 \
    --set seed="$SEED" \
    --set num_steps="$NUM_STEPS" \
    --set wandb_run_name="$label-${JOBID}" \
    2>&1 | tee "$logf"

echo "[grpo-distill] done"
