#!/usr/bin/env bash
# run_train_sft_rft.sh — Step 2 of §7.2 positive-control: SFT -7B-SFT on the verifier-accepted
# rollouts from step 1 (rft_data/<task>_from_<teacher>_seed4242.jsonl).  Output: -7B-SFT-gsm at
# harness/checkpoints/teacher_7B-SFT-gsm/, a task-specialized same-base teacher for step 3.
# One H100; bf16 mixed; 7B + Adam fp32 master + bf16 grads ≈ fits 80 GB with gradient_checkpointing.
#
#   sbatch /project/inniang/research/harness/run_train_sft_rft.sh
#   DATA=rft_data/...jsonl OUT=harness/checkpoints/teacher_7B-SFT-gsm sbatch ...   # overrides
#
#SBATCH --job-name=sft-rft
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:h100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
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

MODEL="${MODEL:-allenai/OLMo-2-1124-7B-SFT}"
DATA="${DATA:-/project/inniang/research/rft_data/gsm_symbolic_from_OLMo-2-1124-7B-Instruct_seed4242.jsonl}"
OUT="${OUT:-/project/inniang/research/harness/checkpoints/teacher_7B-SFT-gsm}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-5e-6}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
BATCH="${BATCH:-1}"

echo "[sft-rft] node=$(hostname) job=${SLURM_JOB_ID:-local$$} model=$MODEL data=$DATA out=$OUT"
echo "[sft-rft] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
[ -s "$DATA" ] || { echo "[sft-rft] ERROR: data file empty or missing: $DATA"; exit 2; }
echo "[sft-rft] data has $(wc -l < "$DATA") accepted lines"

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
  --seed 42 \
  --gradient_checkpointing

echo "[sft-rft] DONE. Saved to $OUT"
ls -la "$OUT"
