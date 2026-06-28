#!/usr/bin/env bash
# run_opd_teachers.sh — roadmap.md's flagship experiment (opd-different-teachers.md, #1) on one 8×H100
# node.  8 runs = {3 same-base 7B teachers (-SFT / -DPO / -Instruct) + a no-teacher GRPO-on-the-student
# baseline} × {seed 42, 43}, one process per GPU.  Every run uses harness/configs/opd_diff_teachers.yaml
# verbatim (student OLMo-2-0425-1B-SFT, task reasoning_gym gsm_symbolic, lam=1 plain OPD, identical
# hyperparams) — the ONLY per-arm `--set` is teacher.model_name (and lam=0 + teacher.kind=none for the
# baseline) + seed.  This is the controlled "is the SFT-teacher vs RL-teacher gap real?" comparison.
#
# Logging is W&B, run OFFLINE in-job; `wandb sync research/wandb/offline-run-*` after.  HF snapshots
# (OLMo-2-0425-1B-SFT, OLMo-2-1124-7B-SFT, OLMo-2-1124-7B-DPO, OLMo-2-1124-7B-Instruct) must be
# pre-staged under $HF_HOME.
#
# Usage (run from anywhere — paths are absolute):
#   sbatch /project/inniang/research/harness/run_opd_teachers.sh
#   NUM_STEPS=20 sbatch /project/inniang/research/harness/run_opd_teachers.sh   # short test sweep
#
#SBATCH --job-name=opd-teachers
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

CONFIG="${CONFIG:-harness/configs/opd_diff_teachers.yaml}"
NUM_STEPS="${NUM_STEPS:-500}"
JOBID="${SLURM_JOB_ID:-local$$}"
mkdir -p harness/logs

# 8 arms, one per GPU.  "label|extra --set flags"
SWEEP=(
  "opd-sft7b-s42|--set teacher.model_name=allenai/OLMo-2-1124-7B-SFT --set seed=42"
  "opd-dpo7b-s42|--set teacher.model_name=allenai/OLMo-2-1124-7B-DPO --set seed=42"
  "opd-instruct7b-s42|--set teacher.model_name=allenai/OLMo-2-1124-7B-Instruct --set seed=42"
  "rl-baseline-s42|--set lam=0 --set teacher.kind=none --set seed=42"
  "opd-sft7b-s43|--set teacher.model_name=allenai/OLMo-2-1124-7B-SFT --set seed=43"
  "opd-dpo7b-s43|--set teacher.model_name=allenai/OLMo-2-1124-7B-DPO --set seed=43"
  "opd-instruct7b-s43|--set teacher.model_name=allenai/OLMo-2-1124-7B-Instruct --set seed=43"
  "rl-baseline-s43|--set lam=0 --set teacher.kind=none --set seed=43"
)

echo "[run_opd_teachers] node=$(hostname) job=$JOBID config=$CONFIG num_steps=$NUM_STEPS"
echo "[run_opd_teachers] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"

pids=(); labels=()
for i in "${!SWEEP[@]}"; do
  label="${SWEEP[$i]%%|*}"
  extra="${SWEEP[$i]#*|}"
  logf="harness/logs/opdT_${i}_${label}_${JOBID}.log"
  echo "[run_opd_teachers] GPU $i  ->  $label  ($logf)"
  CUDA_VISIBLE_DEVICES="$i" nohup "$PYTHON" -m harness.unified_trainer \
      --config "$CONFIG" \
      --set model_device_id=0 \
      --set num_steps="$NUM_STEPS" \
      --set wandb_run_name="$label" \
      $extra \
      > "$logf" 2>&1 &
  pids+=("$!"); labels+=("$label")
  sleep 3   # stagger the (1B student + maybe 7B teacher) loads
done

echo "[run_opd_teachers] launched ${#pids[@]} runs; waiting..."
rc=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then echo "[run_opd_teachers] OK    GPU $j  ${labels[$j]}"
  else code=$?; echo "[run_opd_teachers] FAIL  GPU $j  ${labels[$j]}  (exit $code) — see harness/logs/opdT_${j}_${labels[$j]}_${JOBID}.log"; rc=1; fi
done
echo "[run_opd_teachers] done (rc=$rc)"
exit "$rc"
