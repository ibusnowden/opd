#!/usr/bin/env bash
# run_h100_ddp.sh — ONE RL-corner run spread across all 8 H100s on itiger01 via DDP (torchrun).
# Unlike run_h100_sweep.sh (8 independent single-GPU runs), this is a single training run: the
# OLMo-2-1B student is DDP-wrapped, each rank rolls out its own shard of prompts, gradients all-reduce
# on .backward(), metrics are reduced across ranks, only rank 0 logs to W&B.  ~8× the rollout/train
# throughput of one GPU at the same per-rank batch.  (RL corner only — `_run_distill_loop` / lam>0 is
# single-GPU; PPO + DDP raises — use a group-relative objective. FSDP for a 7B+ *student* is a TODO.)
#
# Logging is W&B, run OFFLINE in-job; `wandb sync research/wandb/offline-run-*` after.  HF snapshots
# (allenai/OLMo-2-0425-1B-Instruct) must be pre-staged under $HF_HOME.
#
# Usage (run from anywhere — paths are absolute):
#   sbatch /project/inniang/research/harness/run_h100_ddp.sh                              # rl_grpo.yaml
#   CONFIG=harness/configs/rl_grpo.yaml NUM_STEPS=20 sbatch /project/inniang/research/harness/run_h100_ddp.sh   # short test
#   NPROC=2 sbatch --gres=gpu:h100_80gb:2 --cpus-per-task=16 /project/inniang/research/harness/run_h100_ddp.sh  # 2-GPU smoke
#
#SBATCH --job-name=rl-ddp
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger01
#SBATCH --gres=gpu:h100_80gb:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
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
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# nproc per node: default to the GPU count of the allocation (falls back to 8).
NPROC="${NPROC:-${SLURM_GPUS_ON_NODE:-8}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( ${SLURM_CPUS_PER_TASK:-64} / NPROC ))}"
[ "$OMP_NUM_THREADS" -ge 1 ] || export OMP_NUM_THREADS=1

CONFIG="${CONFIG:-harness/configs/rl_grpo.yaml}"
NUM_STEPS="${NUM_STEPS:-500}"
# rl_grpo.yaml defaults to the *base* OLMo-2-0425-1B (no chat template); the reasoning_gym rollout
# needs apply_chat_template, so override to the instruct checkpoint (override again via MODEL=...).
MODEL="${MODEL:-allenai/OLMo-2-0425-1B-Instruct}"
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 1; }
mkdir -p harness/logs

echo "[run_h100_ddp] node=$(hostname) job=${SLURM_JOB_ID:-local} config=$CONFIG model=$MODEL nproc_per_node=$NPROC omp_threads=$OMP_NUM_THREADS num_steps=$NUM_STEPS python=$PYTHON"
echo "[run_h100_ddp] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"

# `--set KEY=VALUE` are top-level overrides; everything else comes from $CONFIG.
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
     -m harness.unified_trainer --config "$CONFIG" --set model_name="$MODEL" --set num_steps="$NUM_STEPS"
