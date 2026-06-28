#!/usr/bin/env bash
# run_h100_sweep.sh — fill one 8×H100 node (itiger01) with N parallel single-GPU runs of the
# unified (alpha, lambda, pi_T) trainer's RL corner. The harness trainer is single-GPU only
# (multi-GPU/DDP is Phase 3); to "fully utilize the node" we run one process per GPU, pinned with
# CUDA_VISIBLE_DEVICES, 64 CPU / 8 procs = OMP_NUM_THREADS=8 each. Sweep = the implemented
# outcome-loss objectives (grpo / drgrpo / gspo / cispo / rloo / reinforce / ppo) + grpo+KL-to-base.
#
# Self-contained under research/ (harness/ + the vendored policy_gradients/). Logging is W&B, run
# OFFLINE in-job (compute nodes have no internet); `wandb sync research/wandb/offline-run-*` after.
# HF snapshots must be pre-staged under $HF_HOME (compute nodes have no internet) — see the plan.
#
# Usage (run from anywhere — paths are absolute):
#   sbatch /project/inniang/research/harness/run_h100_sweep.sh
#   NUM_STEPS=50 sbatch /project/inniang/research/harness/run_h100_sweep.sh   # short test sweep
#
#SBATCH --job-name=rl-sweep
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger01
#SBATCH --gres=gpu:h100_80gb:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=/project/inniang/research/harness/logs/%x-%j.out

set -euo pipefail

# --- ABSOLUTE paths only (Key Lesson: ${BASH_SOURCE[0]} resolves to the SLURM spool dir in-job) ---
RESEARCH_ROOT=/project/inniang/research
PYTHON=/project/inniang/.venv/bin/python
cd "$RESEARCH_ROOT"

export PYTHONPATH="$RESEARCH_ROOT:${PYTHONPATH:-}"

# --- HuggingFace: offline, shared snapshot cache (compute nodes have no internet) ---
export HF_HOME=/project/inniang/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- Weights & Biases: offline in-job; sync from the login node afterwards ---
export WANDB_MODE=offline
export WANDB_PROJECT="${WANDB_PROJECT:-distill-harness}"
export WANDB_DIR="$RESEARCH_ROOT"

# --- CPU: 64 cores / 8 processes ---
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
# unbuffered stdout/stderr so the per-run logs show step lines live (else block-buffered behind `> file`)
export PYTHONUNBUFFERED=1

MODEL="${MODEL:-allenai/OLMo-2-0425-1B-Instruct}"
NUM_STEPS="${NUM_STEPS:-500}"
SEED="${SEED:-42}"
JOBID="${SLURM_JOB_ID:-local$$}"
mkdir -p harness/logs

# Common batch bumps for the 80 GB H100 (the configs' defaults were tuned for a 48 GB RTX card).
GROUP_SETS="--set prompts_per_step=8 --set num_rollouts=8 --set rollout_batch_size=8 --set train_batch_size=8 --set batch_acc=2 --set max_new_tokens=1024"
SINGLE_SETS="--set num_rollouts=1 --set rollout_batch_size=1 --set prompts_per_step=32 --set train_batch_size=8 --set batch_acc=2 --set max_new_tokens=1024"

# Sweep: one entry per GPU.  "label|extra --set flags"
SWEEP=(
  "grpo|--set outcome_loss=grpo $GROUP_SETS"
  "drgrpo|--set outcome_loss=drgrpo $GROUP_SETS"
  "gspo|--set outcome_loss=gspo $GROUP_SETS"
  "cispo|--set outcome_loss=cispo $GROUP_SETS"
  "rloo|--set outcome_loss=rloo $GROUP_SETS"
  "reinforce|--set outcome_loss=reinforce $SINGLE_SETS"
  "ppo|--set outcome_loss=ppo $SINGLE_SETS"
  "grpo-klbase|--set outcome_loss=grpo --set beta=0.01 $GROUP_SETS"
)

echo "[run_h100_sweep] node=$(hostname) job=$JOBID model=$MODEL num_steps=$NUM_STEPS seed=$SEED"
echo "[run_h100_sweep] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"

pids=()
labels=()
for i in "${!SWEEP[@]}"; do
  label="${SWEEP[$i]%%|*}"
  extra="${SWEEP[$i]#*|}"
  logf="harness/logs/rl_${i}_${label}_${JOBID}.log"
  echo "[run_h100_sweep] GPU $i  ->  $label  ($logf)"
  CUDA_VISIBLE_DEVICES="$i" nohup "$PYTHON" -m harness.unified_trainer \
      --config harness/configs/rl_grpo.yaml \
      --set model_name="$MODEL" \
      --set model_device_id=0 \
      --set seed="$SEED" \
      --set num_steps="$NUM_STEPS" \
      --set wandb_run_name="rl_${label}_seed${SEED}" \
      $extra \
      > "$logf" 2>&1 &
  pids+=("$!")
  labels+=("$label")
  sleep 2   # stagger model loads a touch
done

echo "[run_h100_sweep] launched ${#pids[@]} runs; waiting..."
rc=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then
    echo "[run_h100_sweep] OK    GPU $j  ${labels[$j]}"
  else
    code=$?
    echo "[run_h100_sweep] FAIL  GPU $j  ${labels[$j]}  (exit $code) — see harness/logs/rl_${j}_${labels[$j]}_${JOBID}.log"
    rc=1
  fi
done
echo "[run_h100_sweep] done (rc=$rc)"
exit "$rc"
