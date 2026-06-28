#!/usr/bin/env bash
# run_exp4_clip_lambda_replicate.sh — replicate the 71271 low-lambda clip=1.0 band.
#
# Job 71271 found the best single-seed clipped interior point so far:
#   seed 42, clip=1.0, λ=0.10 -> pass@1=0.709, pass@16=0.812
#
# This launcher repeats the high-value low-lambda band across seeds 43-45:
#   λ ∈ {0.05, 0.10, 0.20, 0.35}
#
# It intentionally skips λ>=0.50 because 71271 already showed those are negative controls under
# clipping (0.03-0.10 pass@1), and the main question is whether the low-lambda band survives
# multi-seed averaging.
#
# Usage:
#   sbatch /project/inniang/research/harness/run_exp4_clip_lambda_replicate.sh
#   NUM_STEPS=20 sbatch /project/inniang/research/harness/run_exp4_clip_lambda_replicate.sh
#
# Optional one-seed override for local/manual use:
#   SEED=43 sbatch /project/inniang/research/harness/run_exp4_clip_lambda_replicate.sh
#
#SBATCH --job-name=exp4-clip-lam-rep
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger01
#SBATCH --array=43-45
#SBATCH --gres=gpu:h100_80gb:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --output=/project/inniang/research/harness/logs/%x-%A_%a.out

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

CONFIG="${CONFIG:-harness/configs/exp4_lambda_interior_gsm_teacher.yaml}"
NUM_STEPS="${NUM_STEPS:-500}"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-43}}"
CLIP="${CLIP:-1.0}"
JOBID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local$$}}"
TASKID="${SLURM_ARRAY_TASK_ID:-$SEED}"
mkdir -p harness/logs

LAMBDAS=(0.05 0.10 0.20 0.35)

echo "[exp4-clip-lam-rep] node=$(hostname) job=$JOBID task=$TASKID config=$CONFIG num_steps=$NUM_STEPS seed=$SEED clip=$CLIP"
echo "[exp4-clip-lam-rep] teacher=harness/checkpoints/teacher_7B-SFT-gsm/"
echo "[exp4-clip-lam-rep] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"

pids=(); labels=()
for i in "${!LAMBDAS[@]}"; do
  lam="${LAMBDAS[$i]}"
  label="clip${CLIP}-lam${lam}-s${SEED}-rep"
  logf="harness/logs/exp4clr_${i}_${label}_${JOBID}_${TASKID}.log"
  echo "[exp4-clip-lam-rep] GPU $i -> λ=$lam clip=$CLIP seed=$SEED ($logf)"
  CUDA_VISIBLE_DEVICES="$i" nohup "$PYTHON" -m harness.unified_trainer \
      --config "$CONFIG" \
      --set model_device_id=0 \
      --set seed="$SEED" \
      --set num_steps="$NUM_STEPS" \
      --set lam="$lam" \
      --set per_token_kl_clip="$CLIP" \
      --set wandb_run_name="$label-${JOBID}-${TASKID}" \
      > "$logf" 2>&1 &
  pids+=("$!"); labels+=("$label")
  sleep 3
done

echo "[exp4-clip-lam-rep] launched ${#pids[@]} runs; waiting..."
rc=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then echo "[exp4-clip-lam-rep] OK   GPU $j ${labels[$j]}"
  else code=$?; echo "[exp4-clip-lam-rep] FAIL GPU $j ${labels[$j]} (exit $code)"; rc=1; fi
done
echo "[exp4-clip-lam-rep] done (rc=$rc)"
exit "$rc"
