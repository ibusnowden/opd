#!/usr/bin/env bash
# run_exp4_lam1.0_repro.sh — reproducibility check for the v2 71177 λ=1.0 result.
# v2 71177 λ=1.0 seed-42 got pass@1=0.044; Exp 1's OPD ← -7B-SFT seed-42/43-mean was 0.014.
# At λ=1.0 the 2026-05-13 refactor should be a no-op (the `outcome_objective is None` branch
# falls through to the unchanged `UnifiedTokenLoss`). Re-run to disambiguate single-seed variance
# vs. silent behavioral change.  Single-arm, one rtx_6000 (cheaper than tying up an H100).
#
#   sbatch /project/inniang/research/harness/run_exp4_lam1.0_repro.sh
#
#SBATCH --job-name=exp4-lam1.0-repro
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/project/inniang/research/harness/logs/%x-%j.out

set -euo pipefail

cd /project/inniang/research
export PYTHONPATH=/project/inniang/research:${PYTHONPATH:-}
export HF_HOME=/project/inniang/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_PROJECT="${WANDB_PROJECT:-distill-harness}"
export WANDB_DIR=/project/inniang/research
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "[exp4-lam1.0-repro] node=$(hostname) job=${SLURM_JOB_ID} $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
/project/inniang/.venv/bin/python -m harness.unified_trainer \
  --config harness/configs/exp4_lambda_interior.yaml \
  --set model_device_id=0 \
  --set lam=1.0 \
  --set seed=42 \
  --set num_steps=500 \
  --set wandb_run_name="exp4_lam1.0_seed42_repro"
