#!/usr/bin/env bash
# run_exp1_offpolicy_eval.sh — eval-only re-run for the §8.1 arm (job 80388 trained the
# checkpoints fine but the eval step OOM'd in _mean_token_entropy at gen_batch_size=32).
# Fix: gen_batch_size=8 + expandable_segments. No regeneration / retraining — the 4 SFT
# checkpoints + the 2 OPD checkpoints already exist on disk.
#
#   sbatch /project/inniang/research/harness/run_exp1_offpolicy_eval.sh
#
#SBATCH --job-name=exp1-offpolicy-eval
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:h100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
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
export PYTORCH_ALLOC_CONF=expandable_segments:True   # mitigate the entropy-pass fragmentation OOM

PY=/project/inniang/.venv/bin/python
ROOT=/project/inniang/research
CKPT_DIR=$ROOT/harness/checkpoints
RES_DIR=$ROOT/results/exp1_offpolicy_sft_${SLURM_JOB_ID:-local}
mkdir -p "$RES_DIR"

echo "[offpolicy-eval] node=$(hostname) job=${SLURM_JOB_ID:-local$$}  res=$RES_DIR"
echo "[offpolicy-eval] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

eval_ckpt () {  # $1=ckpt  $2=label
  local ckpt="$1" label="$2"
  [ -s "$ckpt/model.safetensors" ] || { echo "[offpolicy-eval] MISSING ckpt: $ckpt"; return 1; }
  echo "[offpolicy-eval] eval $label  ($ckpt)"
  $PY -m harness.eval_passk \
    --ckpt "$ckpt" --task gsm_symbolic \
    --n-prompts 64 --n-samples 16 --k 1,2,4,8,16 --temps 0.6 \
    --max-new-tokens 1024 --gen-batch-size 8 --eval-seed 1000000 --no-self-bleu \
    --out "$RES_DIR/eval_${label}.json"
}

eval_ckpt "$CKPT_DIR/sft-rft-correct-s42" "sft_correct_s42"
eval_ckpt "$CKPT_DIR/sft-rft-correct-s43" "sft_correct_s43"
eval_ckpt "$CKPT_DIR/sft-rft-unfilt-s42"  "sft_unfilt_s42"
eval_ckpt "$CKPT_DIR/sft-rft-unfilt-s43"  "sft_unfilt_s43"
eval_ckpt "$CKPT_DIR/opd-instruct7b-s42"  "opd_instruct_s42"
eval_ckpt "$CKPT_DIR/opd-instruct7b-s43"  "opd_instruct_s43"

echo "[offpolicy-eval] DONE. results in $RES_DIR"
ls -la "$RES_DIR"
