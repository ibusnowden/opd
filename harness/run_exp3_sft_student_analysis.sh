#!/usr/bin/env bash
# run_exp3_sft_student_analysis.sh — Δθ snapshot + full-effrank for the new
# same-tokenizer SFT-control checkpoint. Both run on a single GPU (the SVDs
# are CPU-bound but the model load uses CUDA via transformers). ~10 min wall.
#
#SBATCH --job-name=anly-sft
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:h100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/project/inniang/research/harness/logs/%x-%j.out

set -euo pipefail

RESEARCH_ROOT=/project/inniang/research
PYTHON=/project/inniang/.venv/bin/python
cd "$RESEARCH_ROOT"

export PYTHONPATH="$RESEARCH_ROOT:${PYTHONPATH:-}"
export HF_HOME=/project/inniang/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

BASE="allenai/OLMo-2-0425-1B-SFT"
CKPT="harness/checkpoints/sft-student-rft-gsm-s42"
LABEL="sft_student_s42"

echo "[anly-sft] job=${SLURM_JOB_ID:-local$$} on $(hostname)"

# 1. Δθ snapshot (sparsity, top-K mass, changed-fraction, by-category).
$PYTHON -m harness.delta_theta_snapshot \
    --base "$BASE" --ckpt "$CKPT" \
    --label "$LABEL" \
    --out "figs/dtheta/dtheta_${LABEL}.json" \
    --save-per-tensor

# 2. Full per-tensor effective rank pass (matches §6.4 protocol).
$PYTHON -m harness.effrank_all_tensors \
    --base "$BASE" --ckpt "$CKPT" --label "$LABEL" \
    --out-dir "figs/dtheta/effrank_full"

echo "[anly-sft] DONE."
ls -la figs/dtheta/dtheta_${LABEL}.json
ls -la figs/dtheta/effrank_full/effrank_${LABEL}.json
