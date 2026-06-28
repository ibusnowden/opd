#!/usr/bin/env bash
# run_smoke_prm.sh — cheap 1-GPU validation of PRM-reweighted OPSD before the full 4-arm sweep.
#   1) harness.smoke_prm_reweight (now WITH CUDA -> runs the answer_info_gain end-to-end check)
#   2) a 2-step arm-C train (clip=null, prm_reweight=true) to exercise the full loop:
#      rollout -> answer_info_gain caching -> training-loop reweight -> prm/* diagnostics.
#
#   sbatch /project/inniang/research/harness/run_smoke_prm.sh
#
#SBATCH --job-name=exp6-smoke-prm
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:h100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:40:00
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

echo "[smoke-prm] node=$(hostname) job=${SLURM_JOB_ID:-local$$}"
echo "[smoke-prm] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

echo "[smoke-prm] === 1) unit + answer_info_gain GPU smoke ==="
$PY -m harness.smoke_prm_reweight

echo "[smoke-prm] === 2) 2-step arm-C train (clip=null, prm_reweight=true) ==="
$PY -m harness.unified_trainer \
    --config harness/configs/exp6_prm_reweighted_opsd.yaml \
    --set model_device_id=0 \
    --set num_steps=2 \
    --set per_token_kl_clip=null \
    --set prm_reweight=true \
    --set data.size=64 \
    --set max_new_tokens=256 \
    --set eval_every=0 \
    --set save_ckpt=false \
    --set wandb_run_name="smoke-prm-armC-${SLURM_JOB_ID:-local}"

echo "[smoke-prm] DONE — if both steps printed loss/grad_norm and prm/weight_mean ~1, the full sweep is safe to launch."
