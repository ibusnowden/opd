#!/usr/bin/env bash
# run_passk_eval_sweep.sh — post-hoc pass@k eval on the 8 saved checkpoints from a finished
# opd-teachers job + the init student baseline.  9 processes pinned to 8 H100s (one extra job sits
# behind whichever finishes first).  Wider k (1..64) than the in-loop eval, at T={0.6, 1.0}.
#
#   sbatch /project/inniang/research/harness/run_passk_eval_sweep.sh                 # default settings
#   N_SAMPLES=128 N_PROMPTS=256 sbatch /project/inniang/research/harness/run_passk_eval_sweep.sh
#
#SBATCH --job-name=passk-eval
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
# eval_passk's entropy pass allocates a (gen_batch × seqlen × vocab) logits tensor; vocab=100352 +
# seqlen up to (prompt + max_new_tokens) ~ 2048 → at gen_batch=32 that's ~26 GB just for fp32 logits.
# 8 is comfortable on 80 GB H100s alongside the 1B student.
GEN_BATCH="${GEN_BATCH:-8}"
EVAL_SEED="${EVAL_SEED:-1000000}"  # disjoint from training seeds 42/43
JOBID="${SLURM_JOB_ID:-local$$}"
OUT_DIR="${OUT_DIR:-results/exp1_passk_${JOBID}}"
mkdir -p harness/logs "$OUT_DIR"

# 9 checkpoints: 8 trained + the init student
declare -a CKPTS=(
  "init|allenai/OLMo-2-0425-1B-SFT"
  "opd-sft7b-s42|harness/checkpoints/opd-sft7b-s42"
  "opd-dpo7b-s42|harness/checkpoints/opd-dpo7b-s42"
  "opd-instruct7b-s42|harness/checkpoints/opd-instruct7b-s42"
  "rl-baseline-s42|harness/checkpoints/rl-baseline-s42"
  "opd-sft7b-s43|harness/checkpoints/opd-sft7b-s43"
  "opd-dpo7b-s43|harness/checkpoints/opd-dpo7b-s43"
  "opd-instruct7b-s43|harness/checkpoints/opd-instruct7b-s43"
  "rl-baseline-s43|harness/checkpoints/rl-baseline-s43"
)

echo "[passk-eval] node=$(hostname) job=$JOBID task=$TASK n_prompts=$N_PROMPTS n_samples=$N_SAMPLES k=$K_VALUES temps=$TEMPS out_dir=$OUT_DIR"
echo "[passk-eval] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"

# Launch in two waves: first the 8 trained checkpoints (one per GPU, parallel), then the init
# (on GPU 0, after the others) — keeps things simple and avoids `wait -n` + `set -e` getting in
# each other's way (which killed the previous attempt when one OOM'd).
# Two waves: wave 1 = the 8 trained checkpoints in parallel (one per GPU); wave 2 = the init
# baseline on GPU 0.  Launches are INLINE (not via $(func) — that runs in a subshell and the
# bg-PID would not be a child of this shell, breaking `wait`).
declare -a pids=() labels=()
for i in "${!CKPTS[@]}"; do
  [ "$i" = "0" ] && continue
  label="${CKPTS[$i]%%|*}"; ckpt="${CKPTS[$i]#*|}"
  gpu=$(( (i - 1) % 8 ))
  logf="harness/logs/passk_${i}_${label}_${JOBID}.log"
  outjson="$OUT_DIR/${label}.json"
  echo "[passk-eval] GPU $gpu  ->  $label  ($ckpt)  -> $outjson"
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

echo "[passk-eval] wave 1 launched (${#pids[@]} jobs); waiting..."
rc=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then echo "[passk-eval] OK    ${labels[$j]}"
  else code=$?; echo "[passk-eval] FAIL  ${labels[$j]}  (exit $code) — see harness/logs/passk_*_${labels[$j]}_${JOBID}.log"; rc=1; fi
done

# Wave 2: the init baseline on GPU 0.
ilab="${CKPTS[0]%%|*}"; ickpt="${CKPTS[0]#*|}"
ilogf="harness/logs/passk_0_${ilab}_${JOBID}.log"
ioutjson="$OUT_DIR/${ilab}.json"
echo "[passk-eval] GPU 0  ->  $ilab  ($ickpt)  -> $ioutjson"
CUDA_VISIBLE_DEVICES=0 nohup "$PYTHON" -m harness.eval_passk \
    --ckpt "$ickpt" --task "$TASK" \
    --n-prompts "$N_PROMPTS" --n-samples "$N_SAMPLES" \
    --k "$K_VALUES" --temps "$TEMPS" \
    --max-new-tokens "$MAX_NEW_TOKENS" --gen-batch-size "$GEN_BATCH" \
    --eval-seed "$EVAL_SEED" --no-self-bleu \
    --out "$ioutjson" \
    > "$ilogf" 2>&1 &
init_pid=$!
echo "[passk-eval] wave 2 launched (init on GPU 0, pid $init_pid); waiting..."
if wait "$init_pid"; then echo "[passk-eval] OK    $ilab"
else code=$?; echo "[passk-eval] FAIL  $ilab  (exit $code) — see $ilogf"; rc=1; fi

echo "[passk-eval] done (rc=$rc); JSON in $OUT_DIR/"
exit "$rc"
