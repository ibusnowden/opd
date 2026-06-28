#!/usr/bin/env bash
# run_exp4_lambda_gsm_teacher.sh — Step 3 of §7.2 positive-control: re-run the 8-arm Exp-4 λ-sweep
# with the task-specialized `-7B-SFT-gsm` teacher.  Compares to the off-the-shelf `-7B-SFT` teacher
# from Exp 4 v2 (71177 seed-42 / 71188 seed-43) — keeps EVERYTHING else identical so the only
# changed variable is teacher task-alignment.  Seed 42 only (1 seed enough for the qualitative
# state-coverage check; can add seed-43 if the result is striking).
#
#   sbatch /project/inniang/research/harness/run_exp4_lambda_gsm_teacher.sh
#
#SBATCH --job-name=exp4-gsm-teacher
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

CONFIG="${CONFIG:-harness/configs/exp4_lambda_interior_gsm_teacher.yaml}"
NUM_STEPS="${NUM_STEPS:-500}"
SEED="${SEED:-42}"
JOBID="${SLURM_JOB_ID:-local$$}"
mkdir -p harness/logs

LAMBDAS=(0.05 0.1 0.2 0.35 0.5 0.7 0.85 1.0)

echo "[exp4-gsm] node=$(hostname) job=$JOBID config=$CONFIG num_steps=$NUM_STEPS seed=$SEED lambdas=${LAMBDAS[*]}"
echo "[exp4-gsm] teacher=harness/checkpoints/teacher_7B-SFT-gsm/ (RFT-SFT specialized -7B-SFT)"
echo "[exp4-gsm] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"

pids=()
labels=()
for i in "${!LAMBDAS[@]}"; do
  lam="${LAMBDAS[$i]}"
  label="exp4gsm_lam${lam}_seed${SEED}"
  logf="harness/logs/${label}_${JOBID}.log"
  echo "[exp4-gsm] GPU $i  ->  λ=$lam  ($logf)"
  CUDA_VISIBLE_DEVICES="$i" nohup "$PYTHON" -m harness.unified_trainer \
      --config "$CONFIG" \
      --set model_device_id=0 \
      --set seed="$SEED" \
      --set num_steps="$NUM_STEPS" \
      --set lam="$lam" \
      --set wandb_run_name="$label" \
      > "$logf" 2>&1 &
  pids+=("$!")
  labels+=("$label")
  sleep 3
done

echo "[exp4-gsm] launched ${#pids[@]} runs; waiting..."
rc=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then
    echo "[exp4-gsm] OK    GPU $j  ${labels[$j]}"
  else
    code=$?
    echo "[exp4-gsm] FAIL  GPU $j  ${labels[$j]}  (exit $code)"
    rc=1
  fi
done
echo "[exp4-gsm] done (rc=$rc)"
exit "$rc"
