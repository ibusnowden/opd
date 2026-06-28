#!/usr/bin/env bash
# run_dtheta_prune_sweep.sh — Exp 3 follow-up: prune-degradation curves.
#
# Sweeps GLOBAL magnitude-pruning fraction over the canonical Δθ batch arms,
# re-evaluating pass@1/pass@16 on gsm_symbolic. Question: does the "broader-tier"
# arm (clip1 λ=0.10, RL baseline) lose less from bottom-pruning than the
# "sharper-tier" arm (GRPO-v2-s42, pure OPD λ=1.0)?
#
# 4 arms × 5 prune levels = 20 evals on 7 H100s = 3 waves.
#
# Usage:
#   sbatch /project/inniang/research/harness/run_dtheta_prune_sweep.sh
#
#SBATCH --job-name=prunesweep
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
OUT_DIR="${OUT_DIR:-figs/dtheta/prune_sweep_${JOBID}}"
mkdir -p harness/logs "$OUT_DIR"

# 4 representative arms — one from each (sharper/broader) × (no-teacher/teacher-blend) corner.
# Sharper tier: grpo_v2_s42, clip1_lam100 (pure OPD)
# Broader tier: rl_baseline_s42, clip1_lam010 (the in-dist winner from §7.7)
declare -a CKPTS=(
  "rl_baseline_s42|harness/checkpoints/rl-baseline-s42"
  "grpo_v2_s42|harness/checkpoints/grpo-distill-seed42-71209"
  "clip1_lam010_s42|harness/checkpoints/clip1.0-lam0.10-s42-71271"
  "clip1_lam100_s42|harness/checkpoints/clip1.0-lam1.0-s42-71250"
)

# Prune fractions: 0% (baseline = full ckpt), 10%, 30%, 50%, 90%
declare -a PRUNE_PCTS=( 0.0 0.10 0.30 0.50 0.90 )

echo "[prune-sweep] node=$(hostname) job=$JOBID task=$TASK n_prompts=$N_PROMPTS n_samples=$N_SAMPLES k=$K_VALUES temps=$TEMPS out_dir=$OUT_DIR"
echo "[prune-sweep] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"

# Flat list of (ckpt, prune_pct) jobs
declare -a JOBS=()
for ck in "${CKPTS[@]}"; do
  for p in "${PRUNE_PCTS[@]}"; do
    JOBS+=("${ck}|${p}")
  done
done
echo "[prune-sweep] total jobs: ${#JOBS[@]}"

run_wave() {
  local start="$1"
  local end="$2"
  local pids=()
  local labels=()
  local i entry label rest ckpt pct gpu logf outjson p_tag

  for i in $(seq "$start" "$end"); do
    [ "$i" -ge "${#JOBS[@]}" ] && continue
    entry="${JOBS[$i]}"                       # label|ckpt|pct
    label="${entry%%|*}"
    rest="${entry#*|}"
    ckpt="${rest%|*}"
    pct="${rest##*|}"
    p_tag=$(printf "p%03d" "$(awk "BEGIN { printf \"%d\", $pct*100 }")")
    gpu=$(( i - start ))
    full_label="${label}_${p_tag}"
    logf="harness/logs/prune_${i}_${full_label}_${JOBID}.log"
    outjson="$OUT_DIR/${full_label}.json"
    echo "[prune-sweep] GPU $gpu -> $full_label ($ckpt, prune=$pct) -> $outjson"
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

  echo "[prune-sweep] wave ${start}-${end} launched (${#pids[@]} jobs); waiting..."
  local rc=0
  local j code
  for j in "${!pids[@]}"; do
    if wait "${pids[$j]}"; then echo "[prune-sweep] OK   ${labels[$j]}"
    else code=$?; echo "[prune-sweep] FAIL ${labels[$j]} (exit $code)"; rc=1; fi
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
echo "[prune-sweep] done (rc=$rc); JSON in $OUT_DIR/"
exit "$rc"
