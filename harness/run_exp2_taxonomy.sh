#!/usr/bin/env bash
# run_exp2_taxonomy.sh — Exp 2 (per-token-kl-pivot-vs-style) taxonomy pass.
# Run `diagnose_per_token_kl` across 5 canonical (student, teacher) pairs on
# `gsm_symbolic`, with the new taxonomy buckets {format, uncertain,
# wrong_confident, content} + the clip=1.0 overlay. The teacher is always the
# same-base `OLMo-2-1124-7B-SFT` (the same one used in Exp 1's `opd-sft7b-*`
# and the §7 trainers). 5 arms × ~5 min each = 25 min on 4 GPUs.
#
#SBATCH --job-name=exp2-tax
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger01
#SBATCH --gres=gpu:h100_80gb:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=01:30:00
#SBATCH --output=/project/inniang/research/harness/logs/%x-%j.out

set -euo pipefail

RESEARCH_ROOT=/project/inniang/research
PYTHON=/project/inniang/.venv/bin/python
cd "$RESEARCH_ROOT"

export PYTHONPATH="$RESEARCH_ROOT:${PYTHONPATH:-}"
export HF_HOME=/project/inniang/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

TEACHER="${TEACHER:-allenai/OLMo-2-1124-7B-SFT}"
TASK="${TASK:-gsm_symbolic}"
N_PROMPTS="${N_PROMPTS:-64}"
N_SAMPLES="${N_SAMPLES:-4}"
MAX_NEW="${MAX_NEW:-1024}"
TEMP="${TEMP:-0.6}"
EVAL_SEED="${EVAL_SEED:-2000000}"
CLIP_THRESH="${CLIP_THRESH:-1.0}"
GEN_BATCH="${GEN_BATCH:-2}"
JOBID="${SLURM_JOB_ID:-local$$}"
OUT_DIR="${OUT_DIR:-results/exp2_taxonomy_${JOBID}}"
mkdir -p harness/logs "$OUT_DIR"

# Five canonical (label, student) pairs. Teacher is always same-base 7B-SFT.
# init:        baseline distribution (what does pre-training give us?)
# clip1_lam010: §7.7 winner — recovered, mass should be clipped away
# clip1_lam100: pure OPD λ=1 dead corner — mass should also be clipped
# grpo_v2:     pure RL, the reference (no teacher signal at training; OPD eval still tells us where it lives)
# sft_student: §6.5 third regime
declare -a ARMS=(
  "init|allenai/OLMo-2-0425-1B-SFT"
  "clip1_lam010_s42|harness/checkpoints/clip1.0-lam0.10-s42-71271"
  "clip1_lam100_s42|harness/checkpoints/clip1.0-lam1.0-s42-71250"
  "grpo_v2_s42|harness/checkpoints/grpo-distill-seed42-71209"
  "sft_student_s42|harness/checkpoints/sft-student-rft-gsm-s42"
)

echo "[exp2-tax] node=$(hostname) job=$JOBID teacher=$TEACHER task=$TASK n_prompts=$N_PROMPTS n_samples=$N_SAMPLES T=$TEMP clip=$CLIP_THRESH"
echo "[exp2-tax] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"
echo "[exp2-tax] arms: ${#ARMS[@]}"

declare -a PIDS=()
declare -a LABELS=()
gpu=0
for entry in "${ARMS[@]}"; do
  label="${entry%%|*}"
  student="${entry#*|}"
  outjson="$OUT_DIR/diag_${label}.json"
  toksjson="$OUT_DIR/tokens_${label}.jsonl"
  logf="harness/logs/exp2tax_${label}_${JOBID}.log"
  echo "[exp2-tax] GPU $gpu -> $label  (student=$student)  -> $outjson"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" -m harness.diagnose_per_token_kl \
      --student "$student" \
      --teacher "$TEACHER" \
      --task "$TASK" --n-prompts "$N_PROMPTS" --n-samples "$N_SAMPLES" \
      --max-new-tokens "$MAX_NEW" --temperature "$TEMP" \
      --eval-seed "$EVAL_SEED" \
      --gen-batch-size "$GEN_BATCH" \
      --clip-thresh "$CLIP_THRESH" \
      --out "$outjson" --tokens-out "$toksjson" \
      > "$logf" 2>&1 &
  PIDS+=("$!")
  LABELS+=("$label")
  # 4 GPUs available on the allocation; cycle through. With 5 arms, last one
  # piggybacks on GPU 0 (waits for arm 0 to free its memory). Actually no —
  # both run concurrently on GPU 0, which OOMs the 1B+7B pair. Wait for the
  # first 4 to finish before launching the 5th.
  gpu=$(( gpu + 1 ))
  if [ "$gpu" -eq 4 ]; then
    echo "[exp2-tax] waiting for first wave (4 jobs) to finish..."
    rc=0
    for j in "${!PIDS[@]}"; do
      if wait "${PIDS[$j]}"; then echo "[exp2-tax] OK ${LABELS[$j]}"
      else code=$?; echo "[exp2-tax] FAIL ${LABELS[$j]} (exit $code)"; rc=1; fi
    done
    PIDS=()
    LABELS=()
    gpu=0
  fi
  sleep 2
done

# wait for any remaining (the 5th arm)
if [ ${#PIDS[@]} -gt 0 ]; then
  echo "[exp2-tax] waiting for tail wave (${#PIDS[@]} jobs)..."
  for j in "${!PIDS[@]}"; do
    if wait "${PIDS[$j]}"; then echo "[exp2-tax] OK ${LABELS[$j]}"
    else code=$?; echo "[exp2-tax] FAIL ${LABELS[$j]} (exit $code)"; fi
  done
fi

echo "[exp2-tax] DONE. JSONs in $OUT_DIR/"
ls -la "$OUT_DIR/"
