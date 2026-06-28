#!/usr/bin/env bash
# run_exp5_opsd_multiseed.sh — §7.10 follow-up: confirm the single-seed OPSD rescue is stable.
#
# §7.10's seed-42 result (job 75338): pure OPD λ=1 rescued 6.5× by answer-conditioning
# (0.029 → 0.188 p@1); mid-λ λ=0.50 flipped from 0.099 → 0.238; low-λ λ=0.10 slightly worse
# (0.709 → 0.665). Multi-seed needed before the headline survives, given §7.6/§7.7 found
# seed-bimodality elsewhere in this regime.
#
# Array launches one task per seed in {43, 44, 45}. Each task runs 3 λ ∈ {0.10, 0.50, 1.0}
# in parallel on 3 GPUs. Same trainer / teacher / clip / hyperparams as job 75338.
#
# Usage:
#   sbatch /project/inniang/research/harness/run_exp5_opsd_multiseed.sh
#
#SBATCH --job-name=exp5-opsd-ms
#SBATCH --partition=bigTiger
#SBATCH --array=43-45
#SBATCH --gres=gpu:h100_80gb:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=12:00:00
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

CONFIG="${CONFIG:-harness/configs/exp5_opsd_lambda.yaml}"
NUM_STEPS="${NUM_STEPS:-500}"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-43}}"
CLIP="${CLIP:-1.0}"
JOBID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local$$}}"
TASKID="${SLURM_ARRAY_TASK_ID:-$SEED}"
mkdir -p harness/logs

LAMBDAS=(0.10 0.50 1.0)

echo "[exp5-opsd-ms] node=$(hostname) job=$JOBID task=$TASKID config=$CONFIG num_steps=$NUM_STEPS seed=$SEED clip=$CLIP"
echo "[exp5-opsd-ms] teacher: PrivilegedInfoTeacher(kind=self, model=-7B-SFT, condition_on=answer)"
echo "[exp5-opsd-ms] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"

pids=(); labels=()
for i in "${!LAMBDAS[@]}"; do
  lam="${LAMBDAS[$i]}"
  label="opsd-clip${CLIP}-lam${lam}-s${SEED}-ms"
  logf="harness/logs/exp5ms_${i}_${label}_${JOBID}_${TASKID}.log"
  echo "[exp5-opsd-ms] GPU $i -> λ=$lam clip=$CLIP seed=$SEED ($logf)"
  CUDA_VISIBLE_DEVICES="$i" nohup "$PYTHON" -m harness.unified_trainer \
      --config "$CONFIG" \
      --set model_device_id=0 \
      --set seed="$SEED" \
      --set num_steps="$NUM_STEPS" \
      --set lam="$lam" \
      --set per_token_kl_clip="$CLIP" \
      --set ckpt_dir="harness/checkpoints/${label}-${JOBID}-${TASKID}" \
      --set wandb_run_name="$label-${JOBID}-${TASKID}" \
      > "$logf" 2>&1 &
  pids+=("$!"); labels+=("$label")
  sleep 3
done

echo "[exp5-opsd-ms] launched ${#pids[@]} runs; waiting..."
rc=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then echo "[exp5-opsd-ms] OK   GPU $j ${labels[$j]}"
  else code=$?; echo "[exp5-opsd-ms] FAIL GPU $j ${labels[$j]} (exit $code)"; rc=1; fi
done
echo "[exp5-opsd-ms] done (rc=$rc)"
exit "$rc"
