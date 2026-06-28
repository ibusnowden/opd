#!/usr/bin/env bash
# run_exp3_sft_student_prune.sh — Exp 3.5 follow-up.
# Same prune-degradation protocol as §6.1 + §6.2 (jobs 72097 + 72122), but on the
# new same-tokenizer SFT-control ckpt. 10 prune levels {0, 10, 30, 50, 60, 70,
# 80, 85, 90, 95} on harness/checkpoints/sft-student-rft-gsm-s42/.
#
# 10 evals on 4×H100 → 3 waves of 4 (last wave has 2). ~75 min wall.
#
#SBATCH --job-name=prune-sft
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
CKPT="${CKPT:-harness/checkpoints/sft-student-rft-gsm-s42}"
LABEL="${LABEL:-sft_student_s42}"
TASK="${TASK:-gsm_symbolic}"
N_PROMPTS="${N_PROMPTS:-64}"
N_SAMPLES="${N_SAMPLES:-16}"
K_VALUES="${K_VALUES:-1,2,4,8,16}"
TEMPS="${TEMPS:-0.6}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
GEN_BATCH="${GEN_BATCH:-8}"
EVAL_SEED="${EVAL_SEED:-1000000}"
JOBID="${SLURM_JOB_ID:-local$$}"
OUT_DIR="${OUT_DIR:-figs/dtheta/prune_sft_student_${JOBID}}"
mkdir -p harness/logs "$OUT_DIR"

# Same 10 prune levels as §6.1 + §6.2 combined.
declare -a PRUNE_PCTS=( 0.00 0.10 0.30 0.50 0.60 0.70 0.80 0.85 0.90 0.95 )

echo "[prune-sft] node=$(hostname) job=$JOBID ckpt=$CKPT label=$LABEL task=$TASK n_prompts=$N_PROMPTS out_dir=$OUT_DIR"
echo "[prune-sft] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"

declare -a JOBS=()
for p in "${PRUNE_PCTS[@]}"; do
  JOBS+=("$p")
done
echo "[prune-sft] total jobs: ${#JOBS[@]}"

run_wave() {
  local start="$1"
  local end="$2"
  local pids=()
  local labels=()
  local i pct gpu p_tag full_label logf outjson

  for i in $(seq "$start" "$end"); do
    [ "$i" -ge "${#JOBS[@]}" ] && continue
    pct="${JOBS[$i]}"
    p_tag=$(printf "p%03d" "$(awk "BEGIN { printf \"%d\", $pct*100 }")")
    gpu=$(( i - start ))
    full_label="${LABEL}_${p_tag}"
    logf="harness/logs/prunesft_${i}_${full_label}_${JOBID}.log"
    outjson="$OUT_DIR/${full_label}.json"
    echo "[prune-sft] GPU $gpu -> $full_label (prune=$pct) -> $outjson"
    CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" -m harness.prune_dtheta_eval \
        --base "$BASE" --ckpt "$CKPT" --prune-pct "$pct" \
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

  echo "[prune-sft] wave ${start}-${end} launched (${#pids[@]} jobs); waiting..."
  local rc=0
  local j code
  for j in "${!pids[@]}"; do
    if wait "${pids[$j]}"; then echo "[prune-sft] OK   ${labels[$j]}"
    else code=$?; echo "[prune-sft] FAIL ${labels[$j]} (exit $code)"; rc=1; fi
  done
  return "$rc"
}

rc=0
# 10 jobs across 4 GPUs: waves of 4
run_wave 0 3 || rc=1
run_wave 4 7 || rc=1
run_wave 8 9 || rc=1
echo "[prune-sft] done (rc=$rc); JSON in $OUT_DIR/"
exit "$rc"
