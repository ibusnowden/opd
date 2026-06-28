#!/usr/bin/env bash
# run_passk_v21_eval.sh — RESULTS.md §7.3 bullets 2 & 4: wide pass@k + cross-task generalisation
# eval over the v2.1 specialized-teacher λ-sweep seed-42 checkpoints (+ the GRPO baseline).
#
#   Bullet 2 (diversity-on-Pareto): default invocation, TASK=gsm_symbolic (held-out same task as
#     training).  Tests whether λ=0.05's 7× entropy advantage converts into wider pass@64 coverage.
#       sbatch /project/inniang/research/harness/run_passk_v21_eval.sh
#
#   Bullet 4 (generalisation gap): TASK=simple_equations (different reasoning_gym task — same math
#     skill, different word-problem surface).  Tests whether the 5–10× train-rollout-acc vs
#     held-out-pass@1 gap from §7.1 is template memorisation or genuine distribution shift.
#       OUT_DIR=results/passk_v21_genshift_simple_eq TASK=simple_equations \
#         sbatch /project/inniang/research/harness/run_passk_v21_eval.sh
#
# 9 checkpoints: 8 v2.1 λ-sweep arms + GRPO baseline (rl-baseline-s42) → wave 1 = 8 on 8 GPUs in
# parallel, wave 2 = GRPO baseline on GPU 0. Wider k (1..64) at T={0.6, 1.0}; otherwise identical
# eval shape to run_passk_eval_sweep.sh so results compare apples-to-apples to §7.1.
#
#SBATCH --job-name=passk-v21
#SBATCH --partition=bigTiger
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
EVAL_SEED="${EVAL_SEED:-1000000}"
JOBID="${SLURM_JOB_ID:-local$$}"
OUT_DIR="${OUT_DIR:-results/passk_v21_${TASK}_${JOBID}}"
mkdir -p harness/logs "$OUT_DIR"

# 9 checkpoints: 8 v2.1 λ-sweep arms + GRPO baseline (Exp 4 v2 lam=0 source-of-truth).
declare -a CKPTS=(
  "rl-baseline-s42|harness/checkpoints/rl-baseline-s42"
  "v21_lam0.05_s42|harness/checkpoints/exp4gsm_lam0.05_seed42"
  "v21_lam0.10_s42|harness/checkpoints/exp4gsm_lam0.1_seed42"
  "v21_lam0.20_s42|harness/checkpoints/exp4gsm_lam0.2_seed42"
  "v21_lam0.35_s42|harness/checkpoints/exp4gsm_lam0.35_seed42"
  "v21_lam0.50_s42|harness/checkpoints/exp4gsm_lam0.5_seed42"
  "v21_lam0.70_s42|harness/checkpoints/exp4gsm_lam0.7_seed42"
  "v21_lam0.85_s42|harness/checkpoints/exp4gsm_lam0.85_seed42"
  "v21_lam1.00_s42|harness/checkpoints/exp4gsm_lam1.0_seed42"
)

echo "[passk-v21] node=$(hostname) job=$JOBID task=$TASK n_prompts=$N_PROMPTS n_samples=$N_SAMPLES k=$K_VALUES temps=$TEMPS out_dir=$OUT_DIR"
echo "[passk-v21] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"

# Wave 1: 8 v2.1 checkpoints (skip index 0 = GRPO baseline) on 8 GPUs in parallel.
declare -a pids=() labels=()
for i in "${!CKPTS[@]}"; do
  [ "$i" = "0" ] && continue
  label="${CKPTS[$i]%%|*}"; ckpt="${CKPTS[$i]#*|}"
  gpu=$(( (i - 1) % 8 ))
  logf="harness/logs/passk_v21_${i}_${label}_${JOBID}.log"
  outjson="$OUT_DIR/${label}.json"
  echo "[passk-v21] GPU $gpu  ->  $label  ($ckpt)  -> $outjson"
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

echo "[passk-v21] wave 1 launched (${#pids[@]} jobs); waiting..."
rc=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then echo "[passk-v21] OK    ${labels[$j]}"
  else code=$?; echo "[passk-v21] FAIL  ${labels[$j]}  (exit $code)"; rc=1; fi
done

# Wave 2: GRPO baseline on GPU 0.
ilab="${CKPTS[0]%%|*}"; ickpt="${CKPTS[0]#*|}"
ilogf="harness/logs/passk_v21_0_${ilab}_${JOBID}.log"
ioutjson="$OUT_DIR/${ilab}.json"
echo "[passk-v21] GPU 0  ->  $ilab  ($ickpt)  -> $ioutjson"
CUDA_VISIBLE_DEVICES=0 nohup "$PYTHON" -m harness.eval_passk \
    --ckpt "$ickpt" --task "$TASK" \
    --n-prompts "$N_PROMPTS" --n-samples "$N_SAMPLES" \
    --k "$K_VALUES" --temps "$TEMPS" \
    --max-new-tokens "$MAX_NEW_TOKENS" --gen-batch-size "$GEN_BATCH" \
    --eval-seed "$EVAL_SEED" --no-self-bleu \
    --out "$ioutjson" \
    > "$ilogf" 2>&1 &
init_pid=$!
echo "[passk-v21] wave 2 launched (GRPO baseline on GPU 0, pid $init_pid); waiting..."
if wait "$init_pid"; then echo "[passk-v21] OK    $ilab"
else code=$?; echo "[passk-v21] FAIL  $ilab  (exit $code) — see $ilogf"; rc=1; fi

echo "[passk-v21] done (rc=$rc); JSON in $OUT_DIR/"
exit "$rc"
