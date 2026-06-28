#!/usr/bin/env bash
# run_exp8_hint_search.sh — Exp 8 stage 1: GEPA-style per-task hint search (NO training).
# See harness/hint_search.py for the design. 1B student + 7B-SFT scoring teacher + 7B-Instruct
# mutator, all bf16 co-resident (~30 GB) -> one RTX 6000 Ada (48 GB) is enough; leaves the H100s
# for the Exp 7c arms.
#
#   sbatch /project/inniang/research/harness/run_exp8_hint_search.sh            # full search
#   SMOKE=1 sbatch /project/inniang/research/harness/run_exp8_hint_search.sh    # plumbing smoke
#
#SBATCH --job-name=exp8-hint-search
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
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
JOBID="${SLURM_JOB_ID:-local$$}"
RES_DIR=/project/inniang/research/results/exp8_hint_search_${JOBID}
mkdir -p "$RES_DIR"

echo "[exp8] node=$(hostname) job=$JOBID smoke=${SMOKE:-0}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

extra=()
[ "${SMOKE:-0}" = "1" ] && extra+=(--smoke)

$PY -m harness.hint_search \
    --task gsm_symbolic --seed 42 \
    --n-prompts 64 --n-rollouts 4 \
    --gen-prompts 32 --gen-samples 2 \
    --population 6 --children 6 --generations 3 \
    --beta 0.2 \
    --out "$RES_DIR/search_gsm_symbolic_seed42.json" \
    "${extra[@]}"

echo "[exp8] DONE. results in $RES_DIR"
ls -la "$RES_DIR"
