#!/usr/bin/env bash
# run_exp4_clip_sweep.sh — RESULTS.md §7.7 follow-up: clip-threshold sweep + pure-OPD rescue test.
#
# Job 71249 found that `per_token_kl_clip = 1.0` is the OPD collapse-recovery mechanism (§7.6):
# it rescues both s42 (0.340 → 0.660) and s45 (0.024 → 0.633), removes the bimodality, and prevents
# the step 100-150 collapse entirely.  step_off / linear_anneal / step_on all FAIL to rescue.
# Mechanism: outlier per-token KL signals drive the collapse; clipping them disarms it.
#
# Two questions remain:
#   (Q1) What's the optimal clip threshold?  We only tested 1.0; try {0.5, 2.0, 5.0} too.
#   (Q2) Does per-token KL clipping rescue PURE OPD (λ=1.0)?  If yes, the §7.2 "OPD dead-zone is
#        fundamental" mechanism claim flips: the dead-zone was an artifact of unclipped REINFORCE
#        updates from heavy-tail tokens, not a property of reverse-KL itself.
#
# This 8-arm sweep tests both on one node:
#
#   GPU | label              | λ     | clip  | seed | tests
#   ----+--------------------+-------+-------+------+-----------------------------------
#    0  | clip0.5-lam0.05-s42| 0.05  | 0.5   | 42   | Q1: tighter clip (good seed)
#    1  | clip0.5-lam0.05-s45| 0.05  | 0.5   | 45   | Q1: tighter clip (bad seed)
#    2  | clip2.0-lam0.05-s42| 0.05  | 2.0   | 42   | Q1: looser clip (good seed)
#    3  | clip2.0-lam0.05-s45| 0.05  | 2.0   | 45   | Q1: looser clip (bad seed)
#    4  | clip5.0-lam0.05-s42| 0.05  | 5.0   | 42   | Q1: near-unclipped → should restore bimodality
#    5  | clip5.0-lam0.05-s45| 0.05  | 5.0   | 45   | Q1: near-unclipped → should restore bimodality
#    6  | clip1.0-lam1.0-s42 | 1.00  | 1.0   | 42   | Q2: PURE OPD rescue test (good seed)
#    7  | clip1.0-lam1.0-s45 | 1.00  | 1.0   | 45   | Q2: PURE OPD rescue test (bad seed)
#
# Predictions:
#   - clip=0.5: likely under-uses the teacher signal (most of it is clipped) — possibly worse than 1.0
#   - clip=2.0: still tames the worst outliers; final p@1 probably comparable to 1.0
#   - clip=5.0: ~unclipped (5.0 was the "heavy_tail" threshold in the diagnostic) → bimodality returns
#   - λ=1.0 + clip=1.0: IF it lands above ~0.3, §7.2 dead-zone story flips.  If it stays ~0.01,
#     pure-OPD failure is structural (state-coverage), and clip helps only in the (α=1, λ<1) interior
#     where the (1-λ) outcome branch is doing the heavy lifting.
#
# Existing data points (for comparison, all gsm-teacher, 500 steps):
#   λ=0.05, clip=null: 4-seed {s42, s43, s44, s45} = {0.340, 0.059, 0.443, 0.024}, mean 0.217 (bimodal)
#   λ=0.05, clip=1.0:  2-seed {s42, s45} = {0.660, 0.633}, mean 0.647 (no bimodality)
#   λ=1.0,  clip=null: 2-seed {s42, s43} = {0.011, 0.005}, mean 0.008 (dead zone)
#
# Usage:
#   sbatch /project/inniang/research/harness/run_exp4_clip_sweep.sh
#   NUM_STEPS=20 sbatch /project/inniang/research/harness/run_exp4_clip_sweep.sh   # smoke
#
#SBATCH --job-name=exp4-clip
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

# All arms: gsm-teacher (default in the config), 500 steps, eval @ 250/500.  Schedule is "const"
# (default); the intervention is in λ + per_token_kl_clip.
SWEEP=(
  "clip0.5-lam0.05-s42|--set lam=0.05 --set per_token_kl_clip=0.5 --set seed=42"
  "clip0.5-lam0.05-s45|--set lam=0.05 --set per_token_kl_clip=0.5 --set seed=45"
  "clip2.0-lam0.05-s42|--set lam=0.05 --set per_token_kl_clip=2.0 --set seed=42"
  "clip2.0-lam0.05-s45|--set lam=0.05 --set per_token_kl_clip=2.0 --set seed=45"
  "clip5.0-lam0.05-s42|--set lam=0.05 --set per_token_kl_clip=5.0 --set seed=42"
  "clip5.0-lam0.05-s45|--set lam=0.05 --set per_token_kl_clip=5.0 --set seed=45"
  "clip1.0-lam1.0-s42|--set lam=1.0 --set per_token_kl_clip=1.0 --set seed=42"
  "clip1.0-lam1.0-s45|--set lam=1.0 --set per_token_kl_clip=1.0 --set seed=45"
)

echo "[exp4-clip] node=$(hostname) job=$JOBID config=$CONFIG num_steps=$NUM_STEPS"
echo "[exp4-clip] teacher=harness/checkpoints/teacher_7B-SFT-gsm/ (config default)"
echo "[exp4-clip] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"

pids=(); labels=()
for i in "${!SWEEP[@]}"; do
  label="${SWEEP[$i]%%|*}"
  extra="${SWEEP[$i]#*|}"
  logf="harness/logs/exp4clip_${i}_${label}_${JOBID}.log"
  echo "[exp4-clip] GPU $i  ->  $label  ($logf)"
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

echo "[exp4-clip] launched ${#pids[@]} runs; waiting..."
rc=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then echo "[exp4-clip] OK    GPU $j  ${labels[$j]}"
  else code=$?; echo "[exp4-clip] FAIL  GPU $j  ${labels[$j]}  (exit $code) — see harness/logs/exp4clip_${j}_${labels[$j]}_${JOBID}.log"; rc=1; fi
done
echo "[exp4-clip] done (rc=$rc)"
exit "$rc"
