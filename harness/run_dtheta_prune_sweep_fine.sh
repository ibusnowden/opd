#!/usr/bin/env bash
# run_dtheta_prune_sweep_fine.sh — Exp 3.1 follow-up: localize the cliff.
#
# Same 4 ckpts as the §6.1 sweep, but prune fractions in {0.60, 0.70, 0.80, 0.85, 0.95}
# instead of {0, 0.10, 0.30, 0.50, 0.90}. Goal: locate the cliff between p=50% (where
# all healthy arms are flat) and p=90% (where they diverge — GRPO retains 81%, broader
# arms collapse to ~0). And test whether GRPO retains at p=95% or also dies.
#
# 4 ckpts × 5 prune levels = 20 evals on 4 H100s, ~90 min wall.
#
# Usage:
#   sbatch /project/inniang/research/harness/run_dtheta_prune_sweep_fine.sh
#
#SBATCH --job-name=prunefine
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger01
#SBATCH --gres=gpu:h100_80gb:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=/project/inniang/research/harness/logs/%x-%j.out

set -euo pipefail

RESEARCH_ROOT=/project/inniang/research
PYTHON=/project/inniang/.venv/bin/python
cd "$RESEARCH_ROOT"

export PYTHONPATH="$RESEARCH_ROOT:${PYTHONPATH:-}"
export HF_HOME=/project/inniang/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

BASE="${BASE:-allenai/OLMo-2-0425-1B-SFT}"
TASK="${TASK:-gsm_symbolic}"
N_PROMPTS="${N_PROMPTS:-64}"
N_SAMPLES="${N_SAMPLES:-16}"
K_VALUES="${K_VALUES:-1,2,4,8,16}"
TEMPS="${TEMPS:-0.6}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
GEN_BATCH="${GEN_BATCH:-8}"
EVAL_SEED="${EVAL_SEED:-1000000}"
JOBID="${SLURM_JOB_ID:-local$$}"
OUT_DIR="${OUT_DIR:-figs/dtheta/prune_sweep_fine_${JOBID}}"
mkdir -p harness/logs "$OUT_DIR"

declare -a CKPTS=(
  "rl_baseline_s42|harness/checkpoints/rl-baseline-s42"
  "grpo_v2_s42|harness/checkpoints/grpo-distill-seed42-71209"
  "clip1_lam010_s42|harness/checkpoints/clip1.0-lam0.10-s42-71271"
  "clip1_lam100_s42|harness/checkpoints/clip1.0-lam1.0-s42-71250"
)

# Fine grid AROUND the cliff (50% flat → 90% divergent).
declare -a PRUNE_PCTS=( 0.60 0.70 0.80 0.85 0.95 )

echo "[prune-fine] node=$(hostname) job=$JOBID task=$TASK n_prompts=$N_PROMPTS n_samples=$N_SAMPLES temps=$TEMPS out_dir=$OUT_DIR"
echo "[prune-fine] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"

declare -a JOBS=()
for ck in "${CKPTS[@]}"; do
  for p in "${PRUNE_PCTS[@]}"; do
    JOBS+=("${ck}|${p}")
  done
done
echo "[prune-fine] total jobs: ${#JOBS[@]}"

run_wave() {
  local start="$1"
  local end="$2"
  local pids=()
  local labels=()
  local i entry label rest ckpt pct gpu logf outjson p_tag full_label

  for i in $(seq "$start" "$end"); do
    [ "$i" -ge "${#JOBS[@]}" ] && continue
    entry="${JOBS[$i]}"
    label="${entry%%|*}"
    rest="${entry#*|}"
    ckpt="${rest%|*}"
    pct="${rest##*|}"
    p_tag=$(printf "p%03d" "$(awk "BEGIN { printf \"%d\", $pct*100 }")")
    gpu=$(( i - start ))
    full_label="${label}_${p_tag}"
    logf="harness/logs/prunefine_${i}_${full_label}_${JOBID}.log"
    outjson="$OUT_DIR/${full_label}.json"
    echo "[prune-fine] GPU $gpu -> $full_label ($ckpt, prune=$pct) -> $outjson"
    CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" -m harness.prune_dtheta_eval \
        --base "$BASE" --ckpt "$ckpt" --prune-pct "$pct" \
        --task "$TASK" --n-prompts "$N_PROMPTS" --n-samples "$N_SAMPLES" \
        --k "$K_VALUES" --temps "$TEMPS" \
        --max-new-tokens "$MAX_NEW_TOKENS" --gen-batch-size "$GEN_BATCH" \
        --eval-seed "$EVAL_SEED" \
        --out "$outjson" \
        > "$logf" 2>&1 &
    pids+=("$!")
    labels+=("$full_label")
    sleep 2
  done

  echo "[prune-fine] wave ${start}-${end} launched (${#pids[@]} jobs); waiting..."
  local rc=0
  local j code
  for j in "${!pids[@]}"; do
    if wait "${pids[$j]}"; then echo "[prune-fine] OK   ${labels[$j]}"
    else code=$?; echo "[prune-fine] FAIL ${labels[$j]} (exit $code)"; rc=1; fi
  done
  return "$rc"
}

rc=0
# 20 jobs across 4 GPUs: 5 waves of 4
run_wave 0 3 || rc=1
run_wave 4 7 || rc=1
run_wave 8 11 || rc=1
run_wave 12 15 || rc=1
run_wave 16 19 || rc=1
echo "[prune-fine] done (rc=$rc); JSON in $OUT_DIR/"
exit "$rc"
