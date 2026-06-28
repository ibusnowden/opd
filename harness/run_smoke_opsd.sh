#!/usr/bin/env bash
#SBATCH --job-name=smoke-opsd
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:h100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/project/inniang/research/harness/logs/%x-%j.out

set -euo pipefail
cd /project/inniang/research
export PYTHONPATH=/project/inniang/research:${PYTHONPATH:-}
export HF_HOME=/project/inniang/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

/project/inniang/.venv/bin/python -m harness.smoke_opsd
