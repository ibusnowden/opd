#!/usr/bin/env bash
# run_exp4_multi_seed.sh — RESULTS.md §7.5 bullet 1: multi-seed expansion of the §7.2 / §7.4 results.
#
# The seed-43 column from job 71208 broke the strong reading of §7.2 finding #2: the dual-breakthrough
# location {λ=0.05, λ=0.50} reproduces across seeds, but the magnitude does not (s42 p@1 0.340 / 0.118
# vs s43 0.059 / 0.040).  Job 71209 (A3 GRPO-via-distill control) gave pass@1=0.646, Δ=+0.069 vs the
# v1-path 0.577 baseline — outside the planned ±0.05 consistency band.  With one seed each, we cannot
# tell apart "seed-42 is a positive outlier" from "the v2 code path runs hotter".
#
# This 8-arm node-launcher closes both seed-variance questions on a single 8×H100 node, ~7h:
#
#   GPU | label                | path             | teacher       | lam   | seed
#   ----+----------------------+------------------+---------------+-------+-----
#    0  | gsm-lam0.05-s44      | distill          | 7B-SFT-gsm    | 0.05  | 44
#    1  | gsm-lam0.05-s45      | distill          | 7B-SFT-gsm    | 0.05  | 45
#    2  | gsm-lam0.50-s44      | distill          | 7B-SFT-gsm    | 0.50  | 44
#    3  | gsm-lam0.50-s45      | distill          | 7B-SFT-gsm    | 0.50  | 45
#    4  | rl-baseline-s43      | rl-only (v1)     | none          | 0.0   | 43
#    5  | rl-baseline-s44      | rl-only (v1)     | none          | 0.0   | 44
#    6  | grpo-distill-s43     | distill (v2)     | 7B-SFT-gsm    | 0.001 | 43
#    7  | grpo-distill-s44     | distill (v2)     | 7B-SFT-gsm    | 0.001 | 44
#
# After this run completes we have:
#   - 4 seeds (42, 43, 44, 45) at gsm-teacher × λ ∈ {0.05, 0.50}    → tight CI on the headline magnitude
#   - 3 seeds (42, 43, 44) of pure GRPO v1-path                      → variance of the 0.577 baseline
#   - 3 seeds (42, 43, 44) of GRPO-via-distill v2-path               → can the +0.069 gap be ruled in/out
#
# Each arm gets the in-loop kl_signal/{p50, p90, p99, abs_max, heavy_tail_frac} W&B metrics from the
# Phase-B diagnostic landed in `_run_distill_loop` (also gives s44+s45 the mechanism plot for §8.2).
#
# All three "paths" (distill at λ ∈ {0.05, 0.50}, rl-only baseline, distill at λ=0.001) use the SAME
# config file — exp4_lambda_interior_gsm_teacher.yaml — and toggle the path via per-arm `--set` flags.
# Identical hyperparams to 71202 / 71208 / 71209 (verified by diff of {opd_diff_teachers,
# exp4_lambda_interior_gsm_teacher, grpo_via_distill_path}.yaml on lr/num_steps/batches/eval — all match).
#
# Usage (run from anywhere — paths are absolute):
#   sbatch /project/inniang/research/harness/run_exp4_multi_seed.sh
#   NUM_STEPS=20 sbatch /project/inniang/research/harness/run_exp4_multi_seed.sh    # short smoke test
#
#SBATCH --job-name=exp4-multi-seed
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

# 8 arms, one per GPU.  Format: "label|extra --set flags"
# The gsm-teacher (kind=same_family, model_name=harness/checkpoints/teacher_7B-SFT-gsm) is the
# config's default; rl-baseline arms override to teacher.kind=none + lam=0 to dispatch to _run_rl_loop.
SWEEP=(
  "gsm-lam0.05-s44|--set lam=0.05 --set seed=44"
  "gsm-lam0.05-s45|--set lam=0.05 --set seed=45"
  "gsm-lam0.50-s44|--set lam=0.50 --set seed=44"
  "gsm-lam0.50-s45|--set lam=0.50 --set seed=45"
  "rl-baseline-s43|--set lam=0.0 --set teacher.kind=none --set seed=43"
  "rl-baseline-s44|--set lam=0.0 --set teacher.kind=none --set seed=44"
  "grpo-distill-s43|--set lam=0.001 --set seed=43"
  "grpo-distill-s44|--set lam=0.001 --set seed=44"
)

echo "[exp4-multi-seed] node=$(hostname) job=$JOBID config=$CONFIG num_steps=$NUM_STEPS"
echo "[exp4-multi-seed] teacher (when used)=harness/checkpoints/teacher_7B-SFT-gsm/"
echo "[exp4-multi-seed] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"

pids=(); labels=()
for i in "${!SWEEP[@]}"; do
  label="${SWEEP[$i]%%|*}"
  extra="${SWEEP[$i]#*|}"
  logf="harness/logs/exp4ms_${i}_${label}_${JOBID}.log"
  echo "[exp4-multi-seed] GPU $i  ->  $label  ($logf)"
  CUDA_VISIBLE_DEVICES="$i" nohup "$PYTHON" -m harness.unified_trainer \
      --config "$CONFIG" \
      --set model_device_id=0 \
      --set num_steps="$NUM_STEPS" \
      --set wandb_run_name="$label-${JOBID}" \
      $extra \
      > "$logf" 2>&1 &
  pids+=("$!"); labels+=("$label")
  sleep 3   # stagger the (1B student + maybe 7B teacher) loads
done

echo "[exp4-multi-seed] launched ${#pids[@]} runs; waiting..."
rc=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then echo "[exp4-multi-seed] OK    GPU $j  ${labels[$j]}"
  else code=$?; echo "[exp4-multi-seed] FAIL  GPU $j  ${labels[$j]}  (exit $code) — see harness/logs/exp4ms_${j}_${labels[$j]}_${JOBID}.log"; rc=1; fi
done
echo "[exp4-multi-seed] done (rc=$rc)"
exit "$rc"
