#!/usr/bin/env bash
# run_exp7_arm_1gpu.sh — Exp 7, ONE arm on ONE H100 with the frozen teacher co-resident on the student's
# card (teacher.device_id=0). Lets the 4 arms schedule independently on the congested single H100 node
# (itiger01) instead of needing an 8-GPU exclusive block. Pick the arm with ARM={A,B,C,D}:
#   A onpolicy-opd-noclip   alpha=1 lam=1   clip=null          -- DEAD at 1B (0.008): revive at 7B? (KEY)
#   B offpolicy-revkd       alpha=0 lam=1   clip=null +buffer  -- ALIVE at 1B (0.298): off-policy control
#   C onpolicy-opd-clip     alpha=1 lam=1   clip=1.0           -- does the clip rescue on-policy at 7B?
#   D onpolicy-lowlam-clip  alpha=1 lam=0.1 clip=1.0 grpo      -- best-1B recipe (0.709): hold at 7B?
#
#   ARM=B sbatch /project/inniang/research/harness/run_exp7_arm_1gpu.sh
#
#SBATCH --job-name=exp7arm
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger01
#SBATCH --gres=gpu:h100_80gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
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

ARM="${ARM:?set ARM=A|B|C|D}"
CONFIG="harness/configs/exp7_scale_7b.yaml"
NUM_STEPS="${NUM_STEPS:-250}"
SEED="${SEED:-42}"
JOBID="${SLURM_JOB_ID:-local$$}"
OFFP_FILE="rft_data/gsm_symbolic_from_OLMo-2-1124-7B-Instruct_seed4242_ALL.jsonl"

case "$ARM" in
  A) NAME=onpolicy-opd-noclip;  ALPHA=1.0; LAM=1.0;  CLIP=null; OFFP="" ;;
  B) NAME=offpolicy-revkd;      ALPHA=0.0; LAM=1.0;  CLIP=null; OFFP="$OFFP_FILE" ;;
  C) NAME=onpolicy-opd-clip;    ALPHA=1.0; LAM=1.0;  CLIP=1.0;  OFFP="" ;;
  D) NAME=onpolicy-lowlam-clip; ALPHA=1.0; LAM=0.10; CLIP=1.0;  OFFP="" ;;
  *) echo "bad ARM=$ARM"; exit 2 ;;
esac

LABEL="${NAME}-s${SEED}-1g-${JOBID}"
CKPT="$RESEARCH_ROOT/harness/checkpoints/$LABEL"
RES_DIR="$RESEARCH_ROOT/results/exp7_1gpu_${ARM}_${JOBID}"
LOGF="harness/logs/exp7_${ARM}_${NAME}-s${SEED}_${JOBID}.log"
ELOGF="harness/logs/exp7_${ARM}_eval_${NAME}-s${SEED}_${JOBID}.log"
mkdir -p harness/logs "$RES_DIR"

echo "[exp7-$ARM] node=$(hostname) job=$JOBID arm=$ARM ($NAME alpha=$ALPHA lam=$LAM clip=$CLIP offp=${OFFP:-none}) steps=$NUM_STEPS -> $CKPT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

extra=()
[ -n "$OFFP" ] && extra+=(--set "offpolicy_teacher_states=$OFFP")

CUDA_VISIBLE_DEVICES="0" "$PYTHON" -m harness.unified_trainer \
    --config "$CONFIG" \
    --set model_device_id=0 --set teacher.device_id=0 \
    --set seed="$SEED" --set num_steps="$NUM_STEPS" \
    --set alpha="$ALPHA" --set lam="$LAM" --set per_token_kl_clip="$CLIP" \
    "${extra[@]}" \
    --set save_ckpt=true --set ckpt_dir="$CKPT" \
    --set wandb_run_name="$LABEL" \
    > "$LOGF" 2>&1
echo "[exp7-$ARM] TRAIN done -> $CKPT"

echo "[exp7-$ARM] evaluating (T=0.6, 64x16, k 1..16, eval-seed 1e6)"
CUDA_VISIBLE_DEVICES="0" "$PYTHON" -m harness.eval_passk \
    --ckpt "$CKPT" --task gsm_symbolic \
    --n-prompts 64 --n-samples 16 --k 1,2,4,8,16 --temps 0.6 \
    --max-new-tokens 1024 --eval-seed 1000000 --no-self-bleu --gen-batch-size 8 \
    --out "$RES_DIR/eval_${NAME}_s${SEED}.json" \
    > "$ELOGF" 2>&1
echo "[exp7-$ARM] EVAL done"

"$PYTHON" - "$RES_DIR/eval_${NAME}_s${SEED}.json" "$ARM" <<'PYEOF'
import json, sys
t = json.load(open(sys.argv[1])).get("metrics_by_temp", {}).get("T=0.6", {})
print(f"[exp7-{sys.argv[2]}] HEADLINE  p@1={t.get('eval/pass@1')}  p@16={t.get('eval/pass@16')}  tok_ent={t.get('eval/token_entropy')}")
PYEOF

echo "[exp7-$ARM] DONE. results in $RES_DIR"
ls -la "$RES_DIR"
