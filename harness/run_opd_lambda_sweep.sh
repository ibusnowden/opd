#!/usr/bin/env bash
# run_opd_lambda_sweep.sh — fill one 8×H100 node (itiger01) with an 8-point λ sweep of the
# (α=1, λ, π_T=same_family) corner: pure OPD at λ=1, "expert RL + OPD" at 0<λ<1.  One process per
# GPU, pinned with CUDA_VISIBLE_DEVICES; each = OLMo-2-0425-1B-Instruct student ← OLMo-2-1124-7B-Instruct
# frozen teacher (~30 GB on one 80 GB card), `_run_distill_loop` (harness/unified_trainer.py).
#
#   λ = 1     -> pure OPD: A_t = log π_T - log π_θ                         (Lu & Thinking Machines 2025)
#   0 < λ < 1 -> A_t = λ·(log π_T - log π_θ) + (1-λ)·A^outcome_t           (../expert-rl-plus-opd.md)
# The λ=0 (pure RL / no teacher) baseline is the GRPO run from the objective sweep — run_h100_sweep.sh /
# rl_grpo.yaml — not repeated here (the config validator forbids λ=0 with a non-`none` teacher anyway).
#
# Logging is W&B, run OFFLINE in-job; `wandb sync research/wandb/offline-run-*` after.  HF snapshots
# (allenai/OLMo-2-0425-1B-Instruct + allenai/OLMo-2-1124-7B-Instruct) must be pre-staged under $HF_HOME.
#
# Usage (run from anywhere — paths are absolute):
#   sbatch /project/inniang/research/harness/run_opd_lambda_sweep.sh
#   NUM_STEPS=20 sbatch /project/inniang/research/harness/run_opd_lambda_sweep.sh   # short test sweep
#
#SBATCH --job-name=opd-lambda-sweep
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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

CONFIG="${CONFIG:-harness/configs/opd.yaml}"
NUM_STEPS="${NUM_STEPS:-500}"
SEED="${SEED:-42}"
JOBID="${SLURM_JOB_ID:-local$$}"
mkdir -p harness/logs

# 8 λ values, one per GPU.  (No λ=0 — that's rl_grpo.yaml; the validator forbids λ=0 + same_family.)
LAMBDAS=(0.05 0.1 0.2 0.35 0.5 0.7 0.85 1.0)

echo "[run_opd_lambda_sweep] node=$(hostname) job=$JOBID config=$CONFIG num_steps=$NUM_STEPS seed=$SEED lambdas=${LAMBDAS[*]}"
echo "[run_opd_lambda_sweep] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"

pids=()
labels=()
for i in "${!LAMBDAS[@]}"; do
  lam="${LAMBDAS[$i]}"
  label="lam${lam}"
  logf="harness/logs/opd_${i}_${label}_${JOBID}.log"
  echo "[run_opd_lambda_sweep] GPU $i  ->  λ=$lam  ($logf)"
  CUDA_VISIBLE_DEVICES="$i" nohup "$PYTHON" -m harness.unified_trainer \
      --config "$CONFIG" \
      --set model_device_id=0 \
      --set seed="$SEED" \
      --set num_steps="$NUM_STEPS" \
      --set lam="$lam" \
      --set wandb_run_name="opd_${label}_seed${SEED}" \
      > "$logf" 2>&1 &
  pids+=("$!")
  labels+=("$label")
  sleep 3   # stagger the (1B student + 7B teacher) loads
done

echo "[run_opd_lambda_sweep] launched ${#pids[@]} runs; waiting..."
rc=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then
    echo "[run_opd_lambda_sweep] OK    GPU $j  ${labels[$j]}"
  else
    code=$?
    echo "[run_opd_lambda_sweep] FAIL  GPU $j  ${labels[$j]}  (exit $code) — see harness/logs/opd_${j}_${labels[$j]}_${JOBID}.log"
    rc=1
  fi
done
echo "[run_opd_lambda_sweep] done (rc=$rc)"
exit "$rc"
