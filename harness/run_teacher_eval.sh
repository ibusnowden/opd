#!/usr/bin/env bash
# run_teacher_eval.sh — diagnostic: evaluate the three frozen 7B OPD teachers (SFT / DPO / Instruct)
# DIRECTLY on the same held-out gsm_symbolic prompts used for Exp-1's pass@k eval.  This tells us
# the *teachers' own* pass@1/@k ceiling on the task — i.e. whether OPD's ~1% pass@1 on the 1B-SFT
# student reflects OPD-failing or just-the-teacher-can't-do-this.  Same eval_seed as the trained
# checkpoints' eval (`harness/run_passk_eval_sweep.sh`) so prompts are identical.
#
# Each 7B model in bf16 takes ~14 GB; with gen_batch=8 + 1024 max_new_tokens + vocab 100352 entropy
# logits (~6 GB peak), 80 GB H100 has lots of room.  3 evals on 3 GPUs (GPUs 0/1/2); 8-GPU node
# remains --exclusive so the other 5 sit idle — wasteful, but `bigTiger` doesn't do partial.
#
#   sbatch /project/inniang/research/harness/run_teacher_eval.sh
#
#SBATCH --job-name=teacher-eval
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

TASK="${TASK:-gsm_symbolic}"
N_PROMPTS="${N_PROMPTS:-128}"
N_SAMPLES="${N_SAMPLES:-64}"
K_VALUES="${K_VALUES:-1,2,4,8,16,32,64}"
TEMPS="${TEMPS:-0.6,1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
GEN_BATCH="${GEN_BATCH:-8}"
EVAL_SEED="${EVAL_SEED:-1000000}"           # ← SAME as run_passk_eval_sweep.sh so prompts match
JOBID="${SLURM_JOB_ID:-local$$}"
OUT_DIR="${OUT_DIR:-results/teacher_eval_${JOBID}}"
mkdir -p harness/logs "$OUT_DIR"

declare -a CKPTS=(
  "teacher-7b-sft|allenai/OLMo-2-1124-7B-SFT"
  "teacher-7b-dpo|allenai/OLMo-2-1124-7B-DPO"
  "teacher-7b-instruct|allenai/OLMo-2-1124-7B-Instruct"
)

echo "[teacher-eval] node=$(hostname) job=$JOBID task=$TASK n_prompts=$N_PROMPTS n_samples=$N_SAMPLES k=$K_VALUES temps=$TEMPS"
echo "[teacher-eval] eval_seed=$EVAL_SEED (matches run_passk_eval_sweep.sh -> prompts identical to the trained-checkpoint eval)"

declare -a pids=() labels=()
for i in "${!CKPTS[@]}"; do
  label="${CKPTS[$i]%%|*}"; ckpt="${CKPTS[$i]#*|}"
  gpu="$i"
  logf="harness/logs/teval_${i}_${label}_${JOBID}.log"
  outjson="$OUT_DIR/${label}.json"
  echo "[teacher-eval] GPU $gpu  ->  $label  ($ckpt)  -> $outjson"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" -m harness.eval_passk \
      --ckpt "$ckpt" --task "$TASK" \
      --n-prompts "$N_PROMPTS" --n-samples "$N_SAMPLES" \
      --k "$K_VALUES" --temps "$TEMPS" \
      --max-new-tokens "$MAX_NEW_TOKENS" --gen-batch-size "$GEN_BATCH" \
      --eval-seed "$EVAL_SEED" --no-self-bleu \
      --out "$outjson" \
      > "$logf" 2>&1 &
  pids+=("$!"); labels+=("$label")
  sleep 2
done

echo "[teacher-eval] 3 teachers launched on GPUs 0,1,2; waiting..."
rc=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then echo "[teacher-eval] OK    ${labels[$j]}"
  else code=$?; echo "[teacher-eval] FAIL  ${labels[$j]}  (exit $code) — see harness/logs/teval_${j}_${labels[$j]}_${JOBID}.log"; rc=1; fi
done
echo "[teacher-eval] done (rc=$rc); JSON in $OUT_DIR/"
exit "$rc"
