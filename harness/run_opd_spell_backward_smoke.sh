#!/usr/bin/env bash
# run_opd_spell_backward_smoke.sh — DIAGNOSTIC: a small OPD run on reasoning_gym/spell_backward
# (the configs' original placeholder, small enough for the 1B student to make obvious progress).
# 3 arms, one per same-base 7B teacher, seed 42 only, 50 steps each.  Decouples "is the OPD path
# broken?" from "is gsm_symbolic too hard at this scale?".  Expected: rev_kl falls + reward rises
# on all 3 arms — if so, the OPD code is fine and the Exp-1 failure is a teacher/task/scale issue,
# not a bug.
#
#   sbatch /project/inniang/research/harness/run_opd_spell_backward_smoke.sh
#
#SBATCH --job-name=opd-spell-smoke
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

CONFIG="harness/configs/opd_spell_backward_smoke.yaml"
JOBID="${SLURM_JOB_ID:-local$$}"
mkdir -p harness/logs

declare -a ARMS=(
  "opd-sft7b|--set teacher.model_name=allenai/OLMo-2-1124-7B-SFT"
  "opd-dpo7b|--set teacher.model_name=allenai/OLMo-2-1124-7B-DPO"
  "opd-instruct7b|--set teacher.model_name=allenai/OLMo-2-1124-7B-Instruct"
)

echo "[opd-spell-smoke] job=$JOBID config=$CONFIG arms=${#ARMS[@]} (each on its own GPU)"
declare -a pids=() labels=()
for i in "${!ARMS[@]}"; do
  label="${ARMS[$i]%%|*}"; extra="${ARMS[$i]#*|}"
  gpu="$i"
  logf="harness/logs/opdSB_${i}_${label}_${JOBID}.log"
  echo "[opd-spell-smoke] GPU $gpu  ->  $label  ($logf)"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" -m harness.unified_trainer \
      --config "$CONFIG" \
      --set model_device_id=0 \
      --set wandb_run_name="${label}-sb-smoke" \
      $extra \
      > "$logf" 2>&1 &
  pids+=("$!"); labels+=("$label")
  sleep 2
done

echo "[opd-spell-smoke] launched ${#pids[@]}; waiting..."
rc=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then echo "[opd-spell-smoke] OK    ${labels[$j]}"
  else code=$?; echo "[opd-spell-smoke] FAIL  ${labels[$j]}  (exit $code)"; rc=1; fi
done
echo "[opd-spell-smoke] done (rc=$rc)"
exit "$rc"
