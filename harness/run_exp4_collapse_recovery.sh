#!/usr/bin/env bash
# run_exp4_collapse_recovery.sh — RESULTS.md §7.5 bullet 2: OPD collapse-recovery mechanism.
#
# Job 71242 + 71202 + 71208 revealed a striking pattern at λ=0.05, gsm-teacher, across 4 seeds:
#
#     step    1     25    50    100   150   200   250   400   500
#     s42    0.00  0.38  0.41  0.16  0.01  0.01  0.01  0.01  0.32   <- recovers
#     s43    0.00  0.38  0.40  0.02  0.01  0.01  0.01  0.01  0.09   <- partial
#     s44    0.00  0.52  0.38  0.20  0.01  0.01  0.01  0.04  0.35   <- recovers
#     s45    0.00  0.47  0.49  0.01  0.01  0.01  0.01  0.02  0.10   <- partial
#
# ALL 4 seeds collapse around step 100-150; ~half (s42, s44) recover to ~0.32-0.44 pass@1 by step 500;
# the other half (s43, s45) only partially recover to ~0.10.  GRPO (v1 and v2) shows NO such collapse —
# in-training acc rises monotonically.  So the collapse is a property of the teacher-reverse-KL signal,
# not of the data / model / outcome reward.  The mechanism question: is the teacher's per-token signal
# (a) helpful as a warm-up but harmful during collapse,  (b) helpful only as a post-RL refinement,
# (c) dominated by a few destabilising outlier tokens that per-token clipping would tame?
#
# This 8-arm sweep tests all three on the cleanest contrast (1 known-good seed × 1 known-bad seed):
#
#   GPU | label                  | schedule           | per_token_kl_clip | seed
#   ----+------------------------+--------------------+-------------------+-----
#    0  | rec-stepoff-s42        | step_off @ 100     | null              | 42  (good)
#    1  | rec-stepoff-s45        | step_off @ 100     | null              | 45  (bad)
#    2  | rec-stepon-s42         | step_on  @ 200     | null              | 42  (good)
#    3  | rec-stepon-s45         | step_on  @ 200     | null              | 45  (bad)
#    4  | rec-anneal-s42         | linear 50→300      | null              | 42  (good)
#    5  | rec-anneal-s45         | linear 50→300      | null              | 45  (bad)
#    6  | rec-kclip-s42          | const              | 1.0               | 42  (good)
#    7  | rec-kclip-s45          | const              | 1.0               | 45  (bad)
#
# All arms use λ=0.05, the gsm-teacher, and 500 steps — identical to the breakthrough arm.  The only
# changes are the schedule / clip + seed.  Each arm logs the in-loop kl_signal/{p50,p90,p99,abs_max,
# heavy_tail_frac} + the new meta/lam_eff trace, so we can plot the collapse-recovery window against
# the heavy-tail fraction and confirm or refute hypothesis (c) directly.
#
# Predicted outcomes (will be wrong on at least one — that's the point):
#   - rec-stepoff-s45 → high pass@1.  Implies (a): teacher = warm-up only.
#   - rec-stepon-s45  → high pass@1.  Implies (b): teacher = post-RL refinement.
#   - rec-anneal-s45  → high pass@1.  Compatible with (a) — schedules teacher off during collapse.
#   - rec-kclip-s45   → high pass@1.  Implies (c): a few outlier tokens drive the collapse.
#
# If NONE of these rescue the bad seed, the collapse is a property of the joint training trajectory
# that the meta-knobs can't repair → deeper failure mode, next experiment changes the optimizer.
#
# Usage:
#   sbatch /project/inniang/research/harness/run_exp4_collapse_recovery.sh
#   NUM_STEPS=20 sbatch /project/inniang/research/harness/run_exp4_collapse_recovery.sh    # smoke
#
#SBATCH --job-name=exp4-recovery
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
JOBID="${SLURM_JOB_ID:-local$$}"
mkdir -p harness/logs

# Every arm: λ=0.05, gsm-teacher (default in the config), eval @ 250/500.  The intervention is in
# lam_schedule / lam_step / lam_step_end / per_token_kl_clip.
SWEEP=(
  "rec-stepoff-s42|--set lam=0.05 --set lam_schedule=step_off --set lam_step=100 --set seed=42"
  "rec-stepoff-s45|--set lam=0.05 --set lam_schedule=step_off --set lam_step=100 --set seed=45"
  "rec-stepon-s42|--set lam=0.05 --set lam_schedule=step_on --set lam_step=200 --set seed=42"
  "rec-stepon-s45|--set lam=0.05 --set lam_schedule=step_on --set lam_step=200 --set seed=45"
  "rec-anneal-s42|--set lam=0.05 --set lam_schedule=linear_anneal --set lam_step=50 --set lam_step_end=300 --set seed=42"
  "rec-anneal-s45|--set lam=0.05 --set lam_schedule=linear_anneal --set lam_step=50 --set lam_step_end=300 --set seed=45"
  "rec-kclip-s42|--set lam=0.05 --set per_token_kl_clip=1.0 --set seed=42"
  "rec-kclip-s45|--set lam=0.05 --set per_token_kl_clip=1.0 --set seed=45"
)

echo "[exp4-recovery] node=$(hostname) job=$JOBID config=$CONFIG num_steps=$NUM_STEPS"
echo "[exp4-recovery] teacher=harness/checkpoints/teacher_7B-SFT-gsm/ (config default)"
echo "[exp4-recovery] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"

pids=(); labels=()
for i in "${!SWEEP[@]}"; do
  label="${SWEEP[$i]%%|*}"
  extra="${SWEEP[$i]#*|}"
  logf="harness/logs/exp4rec_${i}_${label}_${JOBID}.log"
  echo "[exp4-recovery] GPU $i  ->  $label  ($logf)"
  CUDA_VISIBLE_DEVICES="$i" nohup "$PYTHON" -m harness.unified_trainer \
      --config "$CONFIG" \
      --set model_device_id=0 \
      --set num_steps="$NUM_STEPS" \
      --set wandb_run_name="$label-${JOBID}" \
      $extra \
      > "$logf" 2>&1 &
  pids+=("$!"); labels+=("$label")
  sleep 3
done

echo "[exp4-recovery] launched ${#pids[@]} runs; waiting..."
rc=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then echo "[exp4-recovery] OK    GPU $j  ${labels[$j]}"
  else code=$?; echo "[exp4-recovery] FAIL  GPU $j  ${labels[$j]}  (exit $code) — see harness/logs/exp4rec_${j}_${labels[$j]}_${JOBID}.log"; rc=1; fi
done
echo "[exp4-recovery] done (rc=$rc)"
exit "$rc"
