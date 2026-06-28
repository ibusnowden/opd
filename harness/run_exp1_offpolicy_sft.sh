#!/usr/bin/env bash
# run_exp1_offpolicy_sft.sh — §8.1 "what's load-bearing" arm (L2: confound-controlled).
#
# Closes the deferred off-policy-SFT arm of Exp 1 (RESULTS line 109) with the correctness
# filter ISOLATED. The existing §6.5 SFT control (sft-student-rft-gsm-s42, p@1 0.377) trained
# only on verifier-ACCEPTED teacher traces, so "off-policy beats on-policy OPD (0.013)" is
# confounded by the correctness filter. Here we regenerate the SAME teacher rollouts WITHOUT
# the filter (--keep_all) and SFT two size-matched arms from one generation draw:
#   - CORRECT : acc>=1.0 subset           (= the RFT/STaR recipe; cross-checks 0.377)
#   - UNFILT  : random same-size subset    (mixed correct+incorrect; the disambiguator)
# Both arms: identical model/init/hp/#steps; ONLY the correctness filter differs.
#
# Then eval pass@k (T=0.6, 64x16, eval-seed 1_000_000 — the §6.5 / Exp-1 protocol) on the new
# SFT arms AND re-eval the on-policy OPD-Instruct checkpoints (opd-instruct7b-s42/s43) at the
# SAME seed, so every number in the §8.1 table is apples-to-apples.
#
#   sbatch /project/inniang/research/harness/run_exp1_offpolicy_sft.sh
#
#SBATCH --job-name=exp1-offpolicy-sft
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:h100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
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

PY=/project/inniang/.venv/bin/python
ROOT=/project/inniang/research
TEACHER=allenai/OLMo-2-1124-7B-Instruct
STUDENT=allenai/OLMo-2-0425-1B-SFT
DATA_DIR=$ROOT/rft_data
CKPT_DIR=$ROOT/harness/checkpoints
RES_DIR=$ROOT/results/exp1_offpolicy_sft_${SLURM_JOB_ID:-local}
mkdir -p "$RES_DIR"

ALL=$DATA_DIR/gsm_symbolic_from_OLMo-2-1124-7B-Instruct_seed4242_ALL.jsonl
CORRECT=$DATA_DIR/gsm_symbolic_from_OLMo-2-1124-7B-Instruct_seed4242_CORRECT.jsonl
UNFILT=$DATA_DIR/gsm_symbolic_from_OLMo-2-1124-7B-Instruct_seed4242_UNFILT.jsonl

echo "[offpolicy-sft] node=$(hostname) job=${SLURM_JOB_ID:-local$$}"
echo "[offpolicy-sft] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# ---------------------------------------------------------------------------
# Step 1 — generate UNFILTERED teacher rollouts (same params as the original
# 2550-accepted draw: 1500 prompts x 4 samples, temp 1.0, seed 4242).
# ---------------------------------------------------------------------------
if [ ! -s "$ALL" ]; then
  echo "[offpolicy-sft] STEP 1: generating unfiltered rollouts -> $ALL"
  $PY -m harness.rft_generate \
    --teacher "$TEACHER" --task gsm_symbolic \
    --n_prompts 1500 --n_samples 4 --temperature 1.0 --max_new_tokens 1024 \
    --seed 4242 --gen_batch_size 16 --keep_all \
    --output "$ALL"
else
  echo "[offpolicy-sft] STEP 1: reusing existing $ALL"
fi

# ---------------------------------------------------------------------------
# Step 2 — fail-fast sanity check + build the two size-matched datasets from
# the SAME draw (CORRECT = acc>=1.0; UNFILT = random same-size mixed subset).
# ---------------------------------------------------------------------------
echo "[offpolicy-sft] STEP 2: sanity check + build CORRECT / UNFILT subsets"
$PY - "$ALL" "$CORRECT" "$UNFILT" <<'PYEOF'
import json, random, sys
all_path, correct_path, unfilt_path = sys.argv[1:4]
rows = [json.loads(l) for l in open(all_path) if l.strip()]
n = len(rows)
correct = [r for r in rows if r["accuracy"] >= 1.0]
frac = len(correct) / max(n, 1)
print(f"[build] total={n} correct={len(correct)} ({100*frac:.1f}%)")
# Fail fast if generation is degenerate (so we don't waste train/eval).
assert n >= 5000, f"too few completions ({n}); expected ~6000"
assert 0.20 <= frac <= 0.75, f"correct fraction {frac:.2f} outside sane [0.20,0.75] — bad generation"
# Size-matched: UNFILT has EXACTLY as many rows as CORRECT, drawn at random from ALL.
rng = random.Random(4242)
unfilt = rng.sample(rows, len(correct))
uf_correct = sum(1 for r in unfilt if r["accuracy"] >= 1.0)
print(f"[build] UNFILT size={len(unfilt)} of which {uf_correct} ({100*uf_correct/len(unfilt):.1f}%) happen to be correct")
with open(correct_path, "w") as f:
    for r in correct: f.write(json.dumps(r) + "\n")
with open(unfilt_path, "w") as f:
    for r in unfilt: f.write(json.dumps(r) + "\n")
print(f"[build] wrote {correct_path} ({len(correct)}) and {unfilt_path} ({len(unfilt)})")
PYEOF

# ---------------------------------------------------------------------------
# Step 3 — SFT both arms x 2 seeds. Identical hp to run_exp3_sft_student.sh
# (epochs 2, lr 5e-6, batch 4, grad_accum 4, max_len 2048).
# ---------------------------------------------------------------------------
train_sft () {  # $1=data  $2=outdir  $3=seed
  local data="$1" out="$2" seed="$3"
  echo "[offpolicy-sft] STEP 3: SFT seed=$seed data=$(basename "$data") -> $(basename "$out")"
  $PY -m harness.train_sft_rft \
    --model_name "$STUDENT" --data_path "$data" --output_dir "$out" \
    --num_epochs 2 --per_device_batch_size 4 --grad_accum 4 --lr 5e-6 \
    --warmup_ratio 0.05 --max_length 2048 --seed "$seed"
}

train_sft "$CORRECT" "$CKPT_DIR/sft-rft-correct-s42" 42
train_sft "$CORRECT" "$CKPT_DIR/sft-rft-correct-s43" 43
train_sft "$UNFILT"  "$CKPT_DIR/sft-rft-unfilt-s42"  42
train_sft "$UNFILT"  "$CKPT_DIR/sft-rft-unfilt-s43"  43

# ---------------------------------------------------------------------------
# Step 4 — eval pass@k at the matched protocol on every §8.1 arm.
# ---------------------------------------------------------------------------
eval_ckpt () {  # $1=ckpt  $2=label
  local ckpt="$1" label="$2"
  echo "[offpolicy-sft] STEP 4: eval $label  ($ckpt)"
  $PY -m harness.eval_passk \
    --ckpt "$ckpt" --task gsm_symbolic \
    --n-prompts 64 --n-samples 16 --k 1,2,4,8,16 --temps 0.6 \
    --max-new-tokens 1024 --eval-seed 1000000 --no-self-bleu \
    --out "$RES_DIR/eval_${label}.json"
}

eval_ckpt "$CKPT_DIR/sft-rft-correct-s42" "sft_correct_s42"
eval_ckpt "$CKPT_DIR/sft-rft-correct-s43" "sft_correct_s43"
eval_ckpt "$CKPT_DIR/sft-rft-unfilt-s42"  "sft_unfilt_s42"
eval_ckpt "$CKPT_DIR/sft-rft-unfilt-s43"  "sft_unfilt_s43"
eval_ckpt "$CKPT_DIR/opd-instruct7b-s42"  "opd_instruct_s42"
eval_ckpt "$CKPT_DIR/opd-instruct7b-s43"  "opd_instruct_s43"

echo "[offpolicy-sft] DONE. results in $RES_DIR"
ls -la "$RES_DIR"
