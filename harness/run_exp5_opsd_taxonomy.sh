#!/usr/bin/env bash
# run_exp5_opsd_taxonomy.sh — §7.10 mechanism test: re-run the §5 per-token KL taxonomy on
# the 3 OPSD checkpoints from job 75338. Crucially we use the SAME LOGIT TEACHER as §5
# (allenai/OLMo-2-1124-7B-SFT, NO answer-conditioning at diagnosis time) so the §5 numbers
# are apples-to-apples comparable.
#
# Mechanism prediction (§7.10): training under answer-conditioned OPSD should shift the
# student's KL-vs-logit-teacher mass away from the uncertain bucket and toward content
# (because the answer-conditioned teacher pulled the student into answer-aimed trajectories
# where the regular teacher's signal is content-aligned). If true: the OPSD students should
# have higher content mass / lower uncertain mass than the §5 logit-trained ckpts at matched λ.
#
#SBATCH --job-name=exp5-opsd-tax
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger01
#SBATCH --gres=gpu:h100_80gb:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --output=/project/inniang/research/harness/logs/%x-%j.out

set -euo pipefail

RESEARCH_ROOT=/project/inniang/research
PYTHON=/project/inniang/.venv/bin/python
cd "$RESEARCH_ROOT"

export PYTHONPATH="$RESEARCH_ROOT:${PYTHONPATH:-}"
export HF_HOME=/project/inniang/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

TEACHER="${TEACHER:-allenai/OLMo-2-1124-7B-SFT}"
TASK="${TASK:-gsm_symbolic}"
N_PROMPTS="${N_PROMPTS:-64}"
N_SAMPLES="${N_SAMPLES:-4}"
MAX_NEW="${MAX_NEW:-1024}"
TEMP="${TEMP:-0.6}"
EVAL_SEED="${EVAL_SEED:-2000000}"     # match the §5 seed
GEN_BATCH="${GEN_BATCH:-2}"
CLIP_THRESH="${CLIP_THRESH:-1.0}"
JOBID="${SLURM_JOB_ID:-local$$}"
OUT_DIR="${OUT_DIR:-results/exp5_taxonomy_${JOBID}}"
mkdir -p harness/logs "$OUT_DIR"

declare -a ARMS=(
  "opsd_lam010_s42|harness/checkpoints/opsd-clip1.0-lam0.10-s42-75338"
  "opsd_lam050_s42|harness/checkpoints/opsd-clip1.0-lam0.50-s42-75338"
  "opsd_lam100_s42|harness/checkpoints/opsd-clip1.0-lam1.0-s42-75338"
)

echo "[exp5-opsd-tax] node=$(hostname) job=$JOBID teacher=$TEACHER task=$TASK arms=${#ARMS[@]}"
echo "[exp5-opsd-tax] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"

pids=(); labels=()
for i in "${!ARMS[@]}"; do
  entry="${ARMS[$i]}"
  label="${entry%%|*}"
  student="${entry#*|}"
  outjson="$OUT_DIR/diag_${label}.json"
  toksjson="$OUT_DIR/tokens_${label}.jsonl"
  logf="harness/logs/exp5tax_${label}_${JOBID}.log"
  echo "[exp5-opsd-tax] GPU $i -> $label (student=$student) -> $outjson"
  CUDA_VISIBLE_DEVICES="$i" nohup "$PYTHON" -m harness.diagnose_per_token_kl \
      --student "$student" --teacher "$TEACHER" \
      --task "$TASK" --n-prompts "$N_PROMPTS" --n-samples "$N_SAMPLES" \
      --max-new-tokens "$MAX_NEW" --temperature "$TEMP" \
      --eval-seed "$EVAL_SEED" --gen-batch-size "$GEN_BATCH" \
      --clip-thresh "$CLIP_THRESH" \
      --out "$outjson" --tokens-out "$toksjson" \
      > "$logf" 2>&1 &
  pids+=("$!"); labels+=("$label")
  sleep 2
done

echo "[exp5-opsd-tax] launched ${#pids[@]} runs; waiting..."
rc=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then echo "[exp5-opsd-tax] OK   ${labels[$j]}"
  else code=$?; echo "[exp5-opsd-tax] FAIL ${labels[$j]} (exit $code)"; rc=1; fi
done
echo "[exp5-opsd-tax] done (rc=$rc); JSONs in $OUT_DIR/"
ls -la "$OUT_DIR/"
exit "$rc"
