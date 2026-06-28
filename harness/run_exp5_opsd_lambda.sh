#!/usr/bin/env bash
# run_exp5_opsd_lambda.sh — Exp 5 (prms-as-teachers / OPSD answer-conditioned).
#
# Three λ arms matched to §7.7's clipped low-λ band, plus pure-OPD λ=1.0 (the §7.7 dead corner
# we most want to see rescued by answer-conditioning):
#   λ ∈ {0.10, 0.50, 1.0}
# Seed 42 only (matches §7.7 single-seed pattern; multi-seed if §10 finds a clear winner).
#
# Teacher = PrivilegedInfoTeacher(kind="self", model_name=-7B-SFT, condition_on="answer").
# Implementation: harness/teachers.py + harness/unified_trainer.py rollout-time caching.
#
# 3 arms × 1 seed = 3 evals on 3 GPUs in parallel on itiger01. Wall ~7h (matches §7's per-arm runtime;
# OPSD adds a teacher-forward pass per rollout but that's offset by removing the per-microbatch
# teacher forward in the training inner loop — see the rollout-time caching note in _run_distill_loop).
#
#SBATCH --job-name=exp5-opsd-lam
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger01
#SBATCH --gres=gpu:h100_80gb:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=12:00:00
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

CONFIG="${CONFIG:-harness/configs/exp5_opsd_lambda.yaml}"
NUM_STEPS="${NUM_STEPS:-500}"
SEED="${SEED:-42}"
CLIP="${CLIP:-1.0}"
JOBID="${SLURM_JOB_ID:-local$$}"
mkdir -p harness/logs

LAMBDAS=(0.10 0.50 1.0)

echo "[exp5-opsd-lam] node=$(hostname) job=$JOBID config=$CONFIG num_steps=$NUM_STEPS seed=$SEED clip=$CLIP"
echo "[exp5-opsd-lam] teacher: PrivilegedInfoTeacher(kind=self, model=-7B-SFT, condition_on=answer)"
echo "[exp5-opsd-lam] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"

pids=(); labels=()
for i in "${!LAMBDAS[@]}"; do
  lam="${LAMBDAS[$i]}"
  label="opsd-clip${CLIP}-lam${lam}-s${SEED}"
  logf="harness/logs/exp5_${i}_${label}_${JOBID}.log"
  echo "[exp5-opsd-lam] GPU $i -> λ=$lam clip=$CLIP seed=$SEED ($logf)"
  CUDA_VISIBLE_DEVICES="$i" nohup "$PYTHON" -m harness.unified_trainer \
      --config "$CONFIG" \
      --set model_device_id=0 \
      --set seed="$SEED" \
      --set num_steps="$NUM_STEPS" \
      --set lam="$lam" \
      --set per_token_kl_clip="$CLIP" \
      --set ckpt_dir="harness/checkpoints/${label}-${JOBID}" \
      --set wandb_run_name="$label-${JOBID}" \
      > "$logf" 2>&1 &
  pids+=("$!"); labels+=("$label")
  sleep 3
done

echo "[exp5-opsd-lam] launched ${#pids[@]} runs; waiting..."
rc=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then echo "[exp5-opsd-lam] OK   GPU $j ${labels[$j]}"
  else code=$?; echo "[exp5-opsd-lam] FAIL GPU $j ${labels[$j]} (exit $code)"; rc=1; fi
done
echo "[exp5-opsd-lam] done (rc=$rc)"
exit "$rc"
