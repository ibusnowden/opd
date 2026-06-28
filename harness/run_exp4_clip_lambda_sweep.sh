#!/usr/bin/env bash
# run_exp4_clip_lambda_sweep.sh — RESULTS.md §7.7 final test:
# re-run the §7.2 8-arm λ-sweep with `per_token_kl_clip = 1.0`.
#
# Job 71249 + 71250 found that kl_clip ∈ [0.5, 2.0] rescues both seeds at λ=0.05 to ~0.66 p@1 (mean
# 0.662), matching GRPO mean (0.661 v1 / 0.687 v2).  But clip=1.0 at λ=1.0 (pure OPD) does NOT
# rescue — both s42 and s45 stayed in the dead zone (0.029, 0.010).  So:
#
#   * State-coverage IS structural for pure OPD (the §7.2 mechanism survives at λ=1).
#   * Per-token outlier clipping rescues the (α=1, 0<λ<1) interior at λ=0.05 (the breakthrough arm).
#
# The OPEN question this run answers: is the §7.2 "dead-zone" at λ ∈ [0.10, 0.85] a clip artifact
# too?  Specifically, the bimodal "two-breakthroughs at λ ∈ {0.05, 0.50}" picture and the dead arms
# at λ ∈ {0.10, 0.20, 0.35, 0.70, 0.85} were derived UNCLIPPED.  Under per-token clipping, does the
# entire interior become a flat plateau of high p@1, with λ=1 sticking out as the only failure?  Or
# does the two-breakthrough structure survive clipping (i.e. the dead-zone reflects something more
# than outlier tokens)?
#
# This 8-arm sweep is the direct test:
#
#   GPU | λ value | per_token_kl_clip | seed
#   ----+---------+-------------------+-----
#    0  | 0.05    | 1.0               | 42   (replication of 71249's win — sanity check)
#    1  | 0.10    | 1.0               | 42   (was 0.015 unclipped — flat-plateau test)
#    2  | 0.20    | 1.0               | 42   (was 0.019 unclipped — flat-plateau test)
#    3  | 0.35    | 1.0               | 42   (was 0.033 unclipped — flat-plateau test)
#    4  | 0.50    | 1.0               | 42   (was 0.118 unclipped — breakthrough test)
#    5  | 0.70    | 1.0               | 42   (was 0.006 unclipped — flat-plateau test)
#    6  | 0.85    | 1.0               | 42   (was 0.021 unclipped — flat-plateau test)
#    7  | 1.00    | 1.0               | 42   (was 0.011 unclipped — control; 71250 already
#                                                showed clip=1.0 doesn't rescue pure OPD; 2nd seed)
#
# Predictions:
#   - **Flat-plateau hypothesis**: λ ∈ [0.05, 0.85] all land at p@1 ≈ 0.65 ± 0.05 with clip=1.0.
#     The dead-zone disappears; the meta-knob plateau becomes "any non-trivial GRPO weight + dense
#     teacher signal + per-token clip ≈ GRPO performance with non-trivial entropy gain".
#   - **Two-breakthroughs survive**: λ ∈ {0.05, 0.50} still pop above the rest; intermediate λs
#     stay in dead-zone or only partially recover.  Would mean clipping helps at specific λs only.
#   - λ=1.00 (control): expected p@1 ≈ 0.01-0.05 (replicating 71250's pure-OPD result at a 2nd seed).
#
# Usage:
#   sbatch /project/inniang/research/harness/run_exp4_clip_lambda_sweep.sh
#   NUM_STEPS=20 sbatch /project/inniang/research/harness/run_exp4_clip_lambda_sweep.sh  # smoke
#
#SBATCH --job-name=exp4-clip-lam
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger01
#SBATCH --gres=gpu:h100_80gb:7
#SBATCH --cpus-per-task=56
#SBATCH --mem=560G
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
CLIP="${CLIP:-1.0}"
JOBID="${SLURM_JOB_ID:-local$$}"
mkdir -p harness/logs

LAMBDAS=(0.05 0.10 0.20 0.35 0.50 0.70 0.85)   # λ=1.00 dropped: 71250 already showed clip=1.0 doesn't rescue pure OPD (2 seeds)

echo "[exp4-clip-lam] node=$(hostname) job=$JOBID config=$CONFIG num_steps=$NUM_STEPS seed=$SEED clip=$CLIP"
echo "[exp4-clip-lam] teacher=harness/checkpoints/teacher_7B-SFT-gsm/"
echo "[exp4-clip-lam] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"

pids=(); labels=()
for i in "${!LAMBDAS[@]}"; do
  lam="${LAMBDAS[$i]}"
  label="clip${CLIP}-lam${lam}-s${SEED}"
  logf="harness/logs/exp4cl_${i}_${label}_${JOBID}.log"
  echo "[exp4-clip-lam] GPU $i  ->  λ=$lam clip=$CLIP  ($logf)"
  CUDA_VISIBLE_DEVICES="$i" nohup "$PYTHON" -m harness.unified_trainer \
      --config "$CONFIG" \
      --set model_device_id=0 \
      --set seed="$SEED" \
      --set num_steps="$NUM_STEPS" \
      --set lam="$lam" \
      --set per_token_kl_clip="$CLIP" \
      --set wandb_run_name="$label-${JOBID}" \
      > "$logf" 2>&1 &
  pids+=("$!"); labels+=("$label")
  sleep 3
done

echo "[exp4-clip-lam] launched ${#pids[@]} runs; waiting..."
rc=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then echo "[exp4-clip-lam] OK    GPU $j  ${labels[$j]}"
  else code=$?; echo "[exp4-clip-lam] FAIL  GPU $j  ${labels[$j]}  (exit $code)"; rc=1; fi
done
echo "[exp4-clip-lam] done (rc=$rc)"
exit "$rc"
