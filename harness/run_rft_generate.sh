#!/usr/bin/env bash
# run_rft_generate.sh — Step 1 of §7.2 positive-control: RFT generation.
# Rolls `-7B-Instruct` (the strongest math performer in D3) on `gsm_symbolic` prompts, filters by
# verifier, saves accepted (prompt, completion) pairs to JSONL for the downstream SFT phase.
# Uses one H100 — 7B-Instruct bf16 ≈ 14 GB, gen_batch=16 plus activations fits comfortably on 80 GB.
#
#   sbatch /project/inniang/research/harness/run_rft_generate.sh
#
#SBATCH --job-name=rft-gen
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:h100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/project/inniang/research/harness/logs/%x-%j.out

set -euo pipefail

cd /project/inniang/research
export PYTHONPATH=/project/inniang/research:${PYTHONPATH:-}
export HF_HOME=/project/inniang/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

TEACHER="${TEACHER:-allenai/OLMo-2-1124-7B-Instruct}"
TASK="${TASK:-gsm_symbolic}"
N_PROMPTS="${N_PROMPTS:-1500}"
N_SAMPLES="${N_SAMPLES:-4}"
TEMP="${TEMP:-1.0}"
SEED="${SEED:-4242}"
GEN_BATCH="${GEN_BATCH:-16}"

# Output: rft_data/<task>_from_<teacher-tag>_<seed>.jsonl
TEACHER_TAG=$(basename "$TEACHER" | tr '/' '_')
OUT="/project/inniang/research/rft_data/${TASK}_from_${TEACHER_TAG}_seed${SEED}.jsonl"

echo "[rft-gen] node=$(hostname) job=${SLURM_JOB_ID:-local$$} teacher=$TEACHER task=$TASK n=$N_PROMPTS×$N_SAMPLES T=$TEMP seed=$SEED out=$OUT"
echo "[rft-gen] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

/project/inniang/.venv/bin/python -m harness.rft_generate \
  --teacher "$TEACHER" \
  --task "$TASK" \
  --n_prompts "$N_PROMPTS" \
  --n_samples "$N_SAMPLES" \
  --temperature "$TEMP" \
  --max_new_tokens 1024 \
  --seed "$SEED" \
  --gen_batch_size "$GEN_BATCH" \
  --device_id 0 \
  --output "$OUT"

echo "[rft-gen] $(wc -l < "$OUT" 2>/dev/null || echo 0) accepted lines in $OUT"
