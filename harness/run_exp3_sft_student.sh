#!/usr/bin/env bash
# run_exp3_sft_student.sh — Exp 3 follow-up (closing the §6 / §6.4 same-tokenizer
# SFT control hole): SFT the 1B student (allenai/OLMo-2-0425-1B-SFT) on the
# verifier-accepted teacher rollouts already used to train the gsm-teacher
# (rft_data/gsm_symbolic_from_OLMo-2-1124-7B-Instruct_seed4242.jsonl). This gives
# the (α=0, λ=1, π_T=δ_data) corner of the meta-algorithm — the only point in
# the (α=1) sweep we don't yet have for the geometry tier comparison.
#
# Hardware: one H100 80GB. 1B + AdamW fp32 master + bf16 grads fits comfortably
# without gradient_checkpointing; we bump per-device batch to 4 and grad_accum to 4
# (=16 effective) for a faster wall-clock — the teacher SFT used batch=1, grad_accum=8
# and took 14 min on the 7B; the 1B should land in <10 min.
#
# Output checkpoint: harness/checkpoints/sft-student-rft-gsm-s42/  — same naming
# convention as the trainer outputs (no SLURM job-id suffix; this is a one-off).
#
#   sbatch /project/inniang/research/harness/run_exp3_sft_student.sh
#
#SBATCH --job-name=sft-student
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:h100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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

MODEL="${MODEL:-allenai/OLMo-2-0425-1B-SFT}"
DATA="${DATA:-/project/inniang/research/rft_data/gsm_symbolic_from_OLMo-2-1124-7B-Instruct_seed4242.jsonl}"
OUT="${OUT:-/project/inniang/research/harness/checkpoints/sft-student-rft-gsm-s42}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-5e-6}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
BATCH="${BATCH:-4}"
SEED="${SEED:-42}"

echo "[sft-student] node=$(hostname) job=${SLURM_JOB_ID:-local$$} model=$MODEL data=$DATA out=$OUT"
echo "[sft-student] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
[ -s "$DATA" ] || { echo "[sft-student] ERROR: data file empty or missing: $DATA"; exit 2; }
echo "[sft-student] data has $(wc -l < "$DATA") accepted lines"
echo "[sft-student] hp: epochs=$EPOCHS lr=$LR batch=$BATCH grad_accum=$GRAD_ACCUM seed=$SEED"

/project/inniang/.venv/bin/python -m harness.train_sft_rft \
  --model_name "$MODEL" \
  --data_path "$DATA" \
  --output_dir "$OUT" \
  --num_epochs "$EPOCHS" \
  --per_device_batch_size "$BATCH" \
  --grad_accum "$GRAD_ACCUM" \
  --lr "$LR" \
  --warmup_ratio 0.05 \
  --max_length 2048 \
  --seed "$SEED"

echo "[sft-student] DONE. Saved to $OUT"
ls -la "$OUT"
