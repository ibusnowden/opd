#!/usr/bin/env bash
# run_exp4_lambda.sh — RESULTS.md §4.2 Next Experiment.  The (α=1, λ) interior between the OPD corner
# (λ=1, lost ~40× pass@1 to GRPO in Exp 1) and the RL corner (λ=0, GRPO baseline; already measured as
# rl-baseline-s42 in Exp 1).  8 arms = 8 λ values, one per H100 on itiger01, all using exp4_lambda_interior.yaml
# (student -1B-SFT, teacher -7B-SFT, gsm_symbolic, 500 steps, seed 42, save_ckpt).
#
#   λ ≈ 0    -> pure GRPO         (skipped — measured as Exp-1 rl-baseline; validator forbids λ=0 + same_family)
#   λ small  -> "expert RL + OPD" : a content-aimed prior nudges GRPO  ([[expert-rl-plus-opd]] (#11) prediction: helps)
#   λ ≈ 1    -> pure OPD          (re-measure with the SFT teacher specifically; Exp 1 used all 3 teachers)
#
# After the run: `python -m harness.eval_passk --ckpt harness/checkpoints/exp4_lam<lam>_seed42 --task gsm_symbolic
#                  --n_prompts 128 --n_samples 64 --k 1,2,4,8,16,32,64 --temps 0.6,1.0` for the full pass@k curve.
#
# Usage (run from anywhere — paths are absolute):
#   sbatch /project/inniang/research/harness/run_exp4_lambda.sh
#   NUM_STEPS=20 sbatch /project/inniang/research/harness/run_exp4_lambda.sh    # short smoke
#
#SBATCH --job-name=exp4-lambda
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

CONFIG="${CONFIG:-harness/configs/exp4_lambda_interior.yaml}"
NUM_STEPS="${NUM_STEPS:-500}"
SEED="${SEED:-42}"
JOBID="${SLURM_JOB_ID:-local$$}"
mkdir -p harness/logs

# 8 λ values, one per GPU.  Logarithmic-ish spacing in the small-λ regime where we expect the most action.
# (No λ=0: validator forbids λ=0 + same_family; the GRPO baseline is Exp-1 rl-baseline.)
LAMBDAS=(0.05 0.1 0.2 0.35 0.5 0.7 0.85 1.0)

echo "[exp4-lambda] node=$(hostname) job=$JOBID config=$CONFIG num_steps=$NUM_STEPS seed=$SEED lambdas=${LAMBDAS[*]}"
echo "[exp4-lambda] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"

pids=()
labels=()
for i in "${!LAMBDAS[@]}"; do
  lam="${LAMBDAS[$i]}"
  label="exp4_lam${lam}_seed${SEED}"
  logf="harness/logs/${label}_${JOBID}.log"
  echo "[exp4-lambda] GPU $i  ->  λ=$lam  ($logf)"
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
  sleep 3   # stagger the (1B + 7B) loads
done

echo "[exp4-lambda] launched ${#pids[@]} runs; waiting..."
rc=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then
    echo "[exp4-lambda] OK    GPU $j  ${labels[$j]}"
  else
    code=$?
    echo "[exp4-lambda] FAIL  GPU $j  ${labels[$j]}  (exit $code) — see harness/logs/${labels[$j]}_${JOBID}.log"
    rc=1
  fi
done
echo "[exp4-lambda] done (rc=$rc)"
exit "$rc"
