#!/usr/bin/env bash
# run_exp7c_arm.sh — Exp 7c (scale vs capacity-gap disambiguator), ONE arm on TWO H100s:
# 7B-SFT student (card 0, bnb 8-bit Adam) <- 13B-Instruct frozen teacher (card 1).
# See harness/configs/exp7c_scale_13b_teacher.yaml for the design + read.
#
#   ARM=A sbatch /project/inniang/research/harness/run_exp7c_arm.sh   # onpolicy-opd-noclip
#   ARM=C sbatch /project/inniang/research/harness/run_exp7c_arm.sh   # onpolicy-opd-clip (KEY)
#   ARM=D sbatch /project/inniang/research/harness/run_exp7c_arm.sh   # onpolicy-lowlam-clip
#
#SBATCH --job-name=exp7c-arm
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger01
#SBATCH --gres=gpu:h100_80gb:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --output=/project/inniang/research/harness/logs/%x-%j.out

set -euo pipefail
RESEARCH_ROOT=/project/inniang/research
PYTHON=/project/inniang/.venv/bin/python
cd "$RESEARCH_ROOT"
export PYTHONPATH="$RESEARCH_ROOT:${PYTHONPATH:-}"
export HF_HOME=/project/inniang/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline WANDB_PROJECT="${WANDB_PROJECT:-distill-harness}" WANDB_DIR="$RESEARCH_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARM="${ARM:?set ARM=A|C|D}"
CONFIG="harness/configs/exp7c_scale_13b_teacher.yaml"
NUM_STEPS="${NUM_STEPS:-250}"
SEED="${SEED:-42}"
JOBID="${SLURM_JOB_ID:-local$$}"

case "$ARM" in
  A) NAME=onpolicy-opd-noclip-13bt;  ALPHA=1.0; LAM=1.0;  CLIP=null ;;
  C) NAME=onpolicy-opd-clip-13bt;    ALPHA=1.0; LAM=1.0;  CLIP=1.0  ;;
  D) NAME=onpolicy-lowlam-clip-13bt; ALPHA=1.0; LAM=0.10; CLIP=1.0  ;;
  *) echo "bad ARM=$ARM"; exit 2 ;;
esac

# fail-fast: the 13B teacher snapshot must be fully on disk (compute nodes are offline)
SNAP=$(ls -d /project/inniang/hf-cache/hub/models--allenai--OLMo-2-1124-13B-Instruct/snapshots/*/ 2>/dev/null | head -1)
[ -n "$SNAP" ] && ls "$SNAP"/*.safetensors >/dev/null 2>&1 \
  || { echo "[exp7c-$ARM] FATAL: 13B-Instruct snapshot incomplete under hf-cache"; exit 1; }

LABEL="${NAME}-s${SEED}-${JOBID}"
CKPT="$RESEARCH_ROOT/harness/checkpoints/$LABEL"
RES_DIR="$RESEARCH_ROOT/results/exp7c_${ARM}_${JOBID}"
LOGF="harness/logs/exp7c_${ARM}_${NAME}-s${SEED}_${JOBID}.log"
ELOGF="harness/logs/exp7c_${ARM}_eval_${NAME}-s${SEED}_${JOBID}.log"
mkdir -p harness/logs "$RES_DIR"

echo "[exp7c-$ARM] node=$(hostname) job=$JOBID arm=$ARM ($NAME alpha=$ALPHA lam=$LAM clip=$CLIP teacher=13B-Instruct) steps=$NUM_STEPS -> $CKPT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

CUDA_VISIBLE_DEVICES="0,1" "$PYTHON" -m harness.unified_trainer \
    --config "$CONFIG" \
    --set model_device_id=0 --set teacher.device_id=1 \
    --set seed="$SEED" --set num_steps="$NUM_STEPS" \
    --set alpha="$ALPHA" --set lam="$LAM" --set per_token_kl_clip="$CLIP" \
    --set save_ckpt=true --set ckpt_dir="$CKPT" \
    --set wandb_run_name="$LABEL" \
    > "$LOGF" 2>&1
echo "[exp7c-$ARM] TRAIN done -> $CKPT"

echo "[exp7c-$ARM] evaluating (T=0.6, 64x16, k 1..16, eval-seed 1e6)"
CUDA_VISIBLE_DEVICES="0" "$PYTHON" -m harness.eval_passk \
    --ckpt "$CKPT" --task gsm_symbolic \
    --n-prompts 64 --n-samples 16 --k 1,2,4,8,16 --temps 0.6 \
    --max-new-tokens 1024 --eval-seed 1000000 --no-self-bleu --gen-batch-size 8 \
    --out "$RES_DIR/eval_${NAME}_s${SEED}.json" \
    > "$ELOGF" 2>&1
echo "[exp7c-$ARM] EVAL done"

"$PYTHON" - "$RES_DIR/eval_${NAME}_s${SEED}.json" "$ARM" <<'PYEOF'
import json, sys
t = json.load(open(sys.argv[1])).get("metrics_by_temp", {}).get("T=0.6", {})
print(f"[exp7c-{sys.argv[2]}] HEADLINE  p@1={t.get('eval/pass@1')}  p@16={t.get('eval/pass@16')}  tok_ent={t.get('eval/token_entropy')}")
print(f"[exp7c-{sys.argv[2]}] exp7 same-size refs: A 0.0088 / C 0.3232 / D 0.6611")
PYEOF

echo "[exp7c-$ARM] DONE. results in $RES_DIR"
ls -la "$RES_DIR"
