#!/usr/bin/env bash
# run_smoke_exp7.sh — 2-GPU, 3-step smoke for the exp7 7B scale test. De-risks the ONE unproven thing:
# full-FT 7B-SFT student + bnb 8-bit Adam on card 0 + frozen 7B-Instruct teacher on card 1 fit & run.
# Runs arm A (on-policy pure-OPD) — the heaviest path (on-policy generation + backward). If A fits,
# arms C/D (same path) and B (off-policy, lighter) fit too. Prints peak GPU memory + per-step tok/s.
#
#   sbatch /project/inniang/research/harness/run_smoke_exp7.sh
#
#SBATCH --job-name=exp7-smoke
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger01
#SBATCH --gres=gpu:h100_80gb:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=00:40:00
#SBATCH --output=/project/inniang/research/harness/logs/%x-%j.out

set -euo pipefail
RESEARCH_ROOT=/project/inniang/research
PYTHON=/project/inniang/.venv/bin/python
cd "$RESEARCH_ROOT"
export PYTHONPATH="$RESEARCH_ROOT:${PYTHONPATH:-}"
export HF_HOME=/project/inniang/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline WANDB_PROJECT="${WANDB_PROJECT:-distill-harness}" WANDB_DIR="$RESEARCH_ROOT"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

JOBID="${SLURM_JOB_ID:-local$$}"
logf="harness/logs/exp7_smoke_armA_${JOBID}.log"
echo "[exp7-smoke] node=$(hostname) job=$JOBID -> $logf"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# arm A: alpha=1 lam=1 clip=null, student card 0 + teacher card 1, 3 steps, no eval, no ckpt save.
CUDA_VISIBLE_DEVICES="0,1" "$PYTHON" -m harness.unified_trainer \
    --config harness/configs/exp7_scale_7b.yaml \
    --set model_device_id=0 --set teacher.device_id=1 \
    --set num_steps=3 --set seed=42 \
    --set alpha=1.0 --set lam=1.0 --set per_token_kl_clip=null \
    --set eval_every=0 --set save_ckpt=false \
    --set wandb_run_name="exp7-smoke-armA-${JOBID}" \
    2>&1 | tee "$logf"

echo "[exp7-smoke] ---- peak memory (both cards) ----"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
echo "[exp7-smoke] ---- per-step throughput ----"
grep -E "step [0-9]+/3" "$logf" || true
echo "[exp7-smoke] DONE — if you see 3 steps with finite loss and no OOM, the 7B+8bit path is good."
