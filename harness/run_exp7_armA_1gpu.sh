#!/usr/bin/env bash
# run_exp7_armA_1gpu.sh — Exp 7 FAST PATH, 1-GPU variant: arm A (on-policy pure-OPD) with the frozen
# teacher CO-RESIDENT on the student's card (teacher.device_id=0 == model_device_id=0 — the harness
# default, the same layout the 1B runs used). Fits the single free H100 on a congested itiger01 so the
# scale headline lands NOW instead of waiting ~2 days for an 8-GPU exclusive slot. Identical training to
# arm A of run_exp7_scale_7b.sh otherwise: alpha=1 lam=1 clip=null, 250 steps, seed 42; then eval at the
# §8.1/Exp-5 protocol (T=0.6, 64x16, eval-seed 1e6).
#
# Memory budget on one 80 GB H100: 7B student bf16 (14) + bf16 grads (14) + bnb Adam8bit states (~14) +
# activations w/ grad-checkpointing (~6) ≈ 48-52 GB student; frozen 7B teacher bf16 inference ≈ 14 GB.
# Total ≈ 62-66 GB → ~14 GB headroom. If it OOMs it does so at step 1 (cheap); fall back to the 2-GPU run.
#
#   sbatch /project/inniang/research/harness/run_exp7_armA_1gpu.sh
#
#SBATCH --job-name=exp7-armA1g
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

CONFIG="harness/configs/exp7_scale_7b.yaml"
NUM_STEPS="${NUM_STEPS:-250}"
SEED="${SEED:-42}"
JOBID="${SLURM_JOB_ID:-local$$}"
NAME="onpolicy-opd-noclip"
LABEL="${NAME}-s${SEED}-armA1g-${JOBID}"
CKPT="$RESEARCH_ROOT/harness/checkpoints/$LABEL"
RES_DIR="$RESEARCH_ROOT/results/exp7_armA_${JOBID}"
LOGF="harness/logs/exp7_armA1g_${NAME}-s${SEED}_${JOBID}.log"
ELOGF="harness/logs/exp7_armA1g_eval_${NAME}-s${SEED}_${JOBID}.log"
mkdir -p harness/logs "$RES_DIR"

echo "[exp7-armA1g] node=$(hostname) job=$JOBID steps=$NUM_STEPS seed=$SEED -> $CKPT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# --- train arm A: alpha=1 lam=1 clip=null, student+teacher BOTH on card 0 (teacher.device_id=0) ---
CUDA_VISIBLE_DEVICES="0" "$PYTHON" -m harness.unified_trainer \
    --config "$CONFIG" \
    --set model_device_id=0 --set teacher.device_id=0 \
    --set seed="$SEED" --set num_steps="$NUM_STEPS" \
    --set alpha=1.0 --set lam=1.0 --set per_token_kl_clip=null \
    --set save_ckpt=true --set ckpt_dir="$CKPT" \
    --set wandb_run_name="$LABEL" \
    > "$LOGF" 2>&1
echo "[exp7-armA1g] TRAIN done -> $CKPT"

# --- eval at the §8.1/Exp-5 matched protocol ---
echo "[exp7-armA1g] evaluating (T=0.6, 64x16, k 1..16, eval-seed 1e6)"
CUDA_VISIBLE_DEVICES="0" "$PYTHON" -m harness.eval_passk \
    --ckpt "$CKPT" --task gsm_symbolic \
    --n-prompts 64 --n-samples 16 --k 1,2,4,8,16 --temps 0.6 \
    --max-new-tokens 1024 --eval-seed 1000000 --no-self-bleu --gen-batch-size 8 \
    --out "$RES_DIR/eval_${NAME}_s${SEED}.json" \
    > "$ELOGF" 2>&1
echo "[exp7-armA1g] EVAL done"

# --- headline ---
"$PYTHON" - "$RES_DIR/eval_${NAME}_s${SEED}.json" <<'PYEOF'
import json, sys
t = json.load(open(sys.argv[1])).get("metrics_by_temp", {}).get("T=0.6", {})
p1, p16, ent = t.get("eval/pass@1"), t.get("eval/pass@16"), t.get("eval/token_entropy")
print(f"[exp7-armA1g] HEADLINE  p@1={p1}  p@16={p16}  tok_ent={ent}")
print("  1B reference: on-policy pure-OPD was 0.008 (dead zone). off-policy revKD control was 0.298.")
print("  * p@1 >> 0.01 -> on-policy pure-OPD REVIVES at native 7B (1B collapse = capacity artifact).")
print("  * p@1 ~  0.01 -> on-policy reverse-KL instability is SCALE-ROBUST (generalizes §8.1).")
PYEOF

echo "[exp7-armA1g] DONE. results in $RES_DIR"
ls -la "$RES_DIR"
