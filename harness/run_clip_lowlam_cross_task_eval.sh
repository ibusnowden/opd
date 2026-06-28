#!/usr/bin/env bash
# run_clip_lowlam_cross_task_eval.sh — cross-task pass@k eval for clipped low-lambda winners.
#
# Evaluates:
#   - GRPO-v2 references: seeds 42, 43, 44
#   - clipped low-lambda band: λ ∈ {0.05, 0.10, 0.20}, seeds 42, 43, 44, 45
#
# Primary task: simple_equations, matching the earlier §7.4 cross-task check.
#
# Usage:
#   sbatch /project/inniang/research/harness/run_clip_lowlam_cross_task_eval.sh
#   TASK=gsm_symbolic sbatch /project/inniang/research/harness/run_clip_lowlam_cross_task_eval.sh
#
#SBATCH --job-name=clip-xtask
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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

TASK="${TASK:-simple_equations}"
N_PROMPTS="${N_PROMPTS:-128}"
N_SAMPLES="${N_SAMPLES:-64}"
K_VALUES="${K_VALUES:-1,2,4,8,16,32,64}"
TEMPS="${TEMPS:-0.6,1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
GEN_BATCH="${GEN_BATCH:-8}"
EVAL_SEED="${EVAL_SEED:-1000000}"
JOBID="${SLURM_JOB_ID:-local$$}"
OUT_DIR="${OUT_DIR:-results/passk_clip_lowlam_${TASK}_${JOBID}}"
mkdir -p harness/logs "$OUT_DIR"

declare -a CKPTS=(
  "grpo_v2_s42|harness/checkpoints/grpo-distill-seed42-71209"
  "grpo_v2_s43|harness/checkpoints/grpo-distill-s43-71242"
  "grpo_v2_s44|harness/checkpoints/grpo-distill-s44-71242"
  "clip1_lam005_s42|harness/checkpoints/clip1.0-lam0.05-s42-71271"
  "clip1_lam005_s43|harness/checkpoints/clip1.0-lam0.05-s43-rep-71395-43"
  "clip1_lam005_s44|harness/checkpoints/clip1.0-lam0.05-s44-rep-71395-44"
  "clip1_lam005_s45|harness/checkpoints/clip1.0-lam0.05-s45-rep-71395-45"
  "clip1_lam010_s42|harness/checkpoints/clip1.0-lam0.10-s42-71271"
  "clip1_lam010_s43|harness/checkpoints/clip1.0-lam0.10-s43-rep-71395-43"
  "clip1_lam010_s44|harness/checkpoints/clip1.0-lam0.10-s44-rep-71395-44"
  "clip1_lam010_s45|harness/checkpoints/clip1.0-lam0.10-s45-rep-71395-45"
  "clip1_lam020_s42|harness/checkpoints/clip1.0-lam0.20-s42-71271"
  "clip1_lam020_s43|harness/checkpoints/clip1.0-lam0.20-s43-rep-71395-43"
  "clip1_lam020_s44|harness/checkpoints/clip1.0-lam0.20-s44-rep-71395-44"
  "clip1_lam020_s45|harness/checkpoints/clip1.0-lam0.20-s45-rep-71395-45"
)

echo "[clip-xtask] node=$(hostname) job=$JOBID task=$TASK n_prompts=$N_PROMPTS n_samples=$N_SAMPLES k=$K_VALUES temps=$TEMPS out_dir=$OUT_DIR"
echo "[clip-xtask] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"

run_wave() {
  local start="$1"
  local end="$2"
  local pids=()
  local labels=()
  local i label ckpt gpu logf outjson

  for i in $(seq "$start" "$end"); do
    [ "$i" -ge "${#CKPTS[@]}" ] && continue
    label="${CKPTS[$i]%%|*}"
    ckpt="${CKPTS[$i]#*|}"
    gpu=$(( i - start ))
    logf="harness/logs/clipxtask_${i}_${label}_${JOBID}.log"
    outjson="$OUT_DIR/${label}.json"
    echo "[clip-xtask] GPU $gpu -> $label ($ckpt) -> $outjson"
    CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" -m harness.eval_passk \
        --ckpt "$ckpt" --task "$TASK" \
        --n-prompts "$N_PROMPTS" --n-samples "$N_SAMPLES" \
        --k "$K_VALUES" --temps "$TEMPS" \
        --max-new-tokens "$MAX_NEW_TOKENS" --gen-batch-size "$GEN_BATCH" \
        --eval-seed "$EVAL_SEED" --no-self-bleu \
        --out "$outjson" \
        > "$logf" 2>&1 &
    pids+=("$!")
    labels+=("$label")
    sleep 2
  done

  echo "[clip-xtask] wave ${start}-${end} launched (${#pids[@]} jobs); waiting..."
  local rc=0
  local j code
  for j in "${!pids[@]}"; do
    if wait "${pids[$j]}"; then echo "[clip-xtask] OK   ${labels[$j]}"
    else code=$?; echo "[clip-xtask] FAIL ${labels[$j]} (exit $code)"; rc=1; fi
  done
  return "$rc"
}

rc=0
run_wave 0 7 || rc=1
run_wave 8 14 || rc=1

echo "[clip-xtask] done (rc=$rc); JSON in $OUT_DIR/"
exit "$rc"
