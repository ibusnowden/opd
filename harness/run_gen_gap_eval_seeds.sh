#!/usr/bin/env bash
# run_gen_gap_eval_seeds.sh — generalization-gap direct test.
#
# §7.7 open follow-up: "re-eval the v2.1 λ=0.05 checkpoint on a gsm_symbolic
# variant with the same template family but redrawn seed pool". This script
# does that by varying --eval-seed.
#
# What we want to know:
#   - Is the v2.1 λ=0.05 s42 "bimodal-breakthrough" (0.415 p@1 on eval-seed
#     1_000_000 in §7.4) stable across eval seed pools, or does it vary?
#   - For contrast: a v2.1 λ=0.05 s43 ckpt (which the 71208 kl_signal traces
#     show collapsed); a clipped λ=0.05 s42 ckpt (recovered, ~0.66 p@1).
#
# 3 ckpts × 3 eval seeds = 9 evals, one node, ~30 min wall.
#
# Usage:
#   sbatch /project/inniang/research/harness/run_gen_gap_eval_seeds.sh
#
#SBATCH --job-name=gengap
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger01
#SBATCH --gres=gpu:h100_80gb:7
#SBATCH --cpus-per-task=56
#SBATCH --mem=448G
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
JOBID="${SLURM_JOB_ID:-local$$}"
OUT_DIR="${OUT_DIR:-results/gengap_eval_seeds_${JOBID}}"
mkdir -p harness/logs "$OUT_DIR"

# 3 ckpts × 3 eval seeds = 9 evals
declare -a CKPTS=(
  "v21_lam005_s42|harness/checkpoints/exp4gsm_lam0.05_seed42"
  "v21_lam005_s43|harness/checkpoints/exp4gsm_lam0.05_seed43"
  "clip1_lam005_s42|harness/checkpoints/clip1.0-lam0.05-s42-71271"
)
declare -a SEEDS=( 1000000 2000000 3000000 )

echo "[gengap] node=$(hostname) job=$JOBID task=$TASK n_prompts=$N_PROMPTS n_samples=$N_SAMPLES k=$K_VALUES temps=$TEMPS out_dir=$OUT_DIR"
echo "[gengap] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"

# Build a flat list of all (ckpt, seed) jobs
declare -a JOBS=()
for ck in "${CKPTS[@]}"; do
  for s in "${SEEDS[@]}"; do
    JOBS+=("${ck}|${s}")
  done
done
echo "[gengap] total jobs: ${#JOBS[@]}"

run_wave() {
  local start="$1"
  local end="$2"
  local pids=()
  local labels=()
  local i entry label ckpt seed gpu logf outjson

  for i in $(seq "$start" "$end"); do
    [ "$i" -ge "${#JOBS[@]}" ] && continue
    entry="${JOBS[$i]}"                       # label|ckpt|seed
    label="${entry%%|*}"                      # v21_lam005_s42
    rest="${entry#*|}"                        # ckpt|seed
    ckpt="${rest%|*}"
    seed="${rest##*|}"
    gpu=$(( i - start ))
    label_with_seed="${label}_es${seed}"
    logf="harness/logs/gengap_${i}_${label_with_seed}_${JOBID}.log"
    outjson="$OUT_DIR/${label_with_seed}.json"
    echo "[gengap] GPU $gpu -> $label_with_seed ($ckpt, eval_seed=$seed) -> $outjson"
    CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" -m harness.eval_passk \
        --ckpt "$ckpt" --task "$TASK" \
        --n-prompts "$N_PROMPTS" --n-samples "$N_SAMPLES" \
        --k "$K_VALUES" --temps "$TEMPS" \
        --max-new-tokens "$MAX_NEW_TOKENS" --gen-batch-size "$GEN_BATCH" \
        --eval-seed "$seed" --no-self-bleu \
        --out "$outjson" \
        > "$logf" 2>&1 &
    pids+=("$!")
    labels+=("$label_with_seed")
    sleep 2
  done

  echo "[gengap] wave ${start}-${end} launched (${#pids[@]} jobs); waiting..."
  local rc=0
  local j code
  for j in "${!pids[@]}"; do
    if wait "${pids[$j]}"; then echo "[gengap] OK   ${labels[$j]}"
    else code=$?; echo "[gengap] FAIL ${labels[$j]} (exit $code)"; rc=1; fi
  done
  return "$rc"
}

rc=0
# 9 evals across 7 GPUs: 7 + 2
run_wave 0 6 || rc=1
run_wave 7 8 || rc=1
echo "[gengap] done (rc=$rc); JSON in $OUT_DIR/"
exit "$rc"
