#!/usr/bin/env bash
# run_exp7b_1b_clip_instruct.sh — Exp 7b: the §7.12 finding-2 cross-teacher hedge closer.
#
# Exp 7 (§7.12) showed the per-token clip converts dead on-policy pure-OPD into a live recipe at 7B
# (A 0.009 -> C 0.323, same 7B-Instruct teacher). The 1B reference for "clipped pure-OPD is still dead"
# (~0.05) came from §7.7 with the 7B-**SFT** teacher — a cross-teacher comparison. This run closes the
# hedge: 1B-SFT student <- 7B-**Instruct** teacher, alpha=1 lam=1 clip=1.0, otherwise verbatim the
# opd_diff_teachers.yaml protocol that produced opd-instruct7b-s42/s43 (the unclipped 0.005/0.008 arms).
#
# Read:
#   p@1 stays ~0.05 (dead)  -> clip's accuracy payoff on pure OPD GROWS with scale (finding 2 confirmed
#                              with a same-teacher 1B ref; the 7B-SFT-teacher ~0.05 was not a teacher artifact).
#   p@1 jumps to ~0.3       -> the §7.7 "clip doesn't rescue pure OPD at 1B" was a TEACHER effect, not a
#                              scale effect; finding 2's scale story needs a rewrite.
#
#   sbatch /project/inniang/research/harness/run_exp7b_1b_clip_instruct.sh
#
#SBATCH --job-name=exp7b-1b-clip-instruct
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:h100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/project/inniang/.venv/bin/python
ROOT=/project/inniang/research
CFG=harness/configs/opd_diff_teachers.yaml
CKPT_DIR=$ROOT/harness/checkpoints
RES_DIR=$ROOT/results/exp7b_1b_clip_instruct_${SLURM_JOB_ID:-local}
mkdir -p "$RES_DIR"

echo "[exp7b] node=$(hostname) job=${SLURM_JOB_ID:-local$$}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# fail-fast: config validates with the overrides applied
$PY - <<'PYEOF'
from harness.config import load_config
cfg = load_config("harness/configs/opd_diff_teachers.yaml")
assert cfg.alpha == 1.0 and cfg.lam == 1.0
print(f"[exp7b] base config OK: student={cfg.model_name} steps={cfg.num_steps} lr={cfg.lr}")
PYEOF

train_arm () {  # $1=seed
  local seed="$1" run="opd-instruct7b-clip1-s$1"
  echo "[exp7b] TRAIN seed=$seed -> $run"
  $PY -m harness.unified_trainer --config "$CFG" \
    --set teacher.model_name=allenai/OLMo-2-1124-7B-Instruct \
    --set per_token_kl_clip=1.0 \
    --set seed="$seed" \
    --set eval_every=0 \
    --set wandb_run_name="$run" \
    --set wandb_group=exp7b-1b-clip-instruct \
    --set ckpt_dir="$CKPT_DIR/$run"
}

eval_ckpt () {  # $1=ckpt  $2=label
  echo "[exp7b] EVAL $2 ($1)"
  $PY -m harness.eval_passk \
    --ckpt "$1" --task gsm_symbolic \
    --n-prompts 64 --n-samples 16 --k 1,2,4,8,16 --temps 0.6 \
    --max-new-tokens 1024 --eval-seed 1000000 --no-self-bleu \
    --gen-batch-size 8 \
    --out "$RES_DIR/eval_$2.json"
}

train_arm 42
eval_ckpt "$CKPT_DIR/opd-instruct7b-clip1-s42" "opd_clip1_instruct_s42"
train_arm 43
eval_ckpt "$CKPT_DIR/opd-instruct7b-clip1-s43" "opd_clip1_instruct_s43"

echo "[exp7b] SUMMARY"
$PY - "$RES_DIR" <<'PYEOF'
import json, os, sys
res = sys.argv[1]
print(f"{'arm':<28}{'p@1':>8}{'p@16':>8}{'tok_ent':>9}")
for label in ["opd_clip1_instruct_s42", "opd_clip1_instruct_s43"]:
    p = os.path.join(res, f"eval_{label}.json")
    if not os.path.exists(p):
        print(f"{label:<28}{'MISSING':>8}"); continue
    t = json.load(open(p)).get("metrics_by_temp", {}).get("T=0.6", {})
    print(f"{label:<28}{t.get('eval/pass@1', float('nan')):>8.4f}"
          f"{t.get('eval/pass@16', float('nan')):>8.4f}{t.get('eval/token_entropy', float('nan')):>9.4f}")
print("refs: unclipped on-policy <- Instruct (s42/s43) p@1 0.005/0.008 | clip <- 7B-SFT (s42, S7.7) ~0.05 | 7B arm C 0.323")
PYEOF

echo "[exp7b] DONE. results in $RES_DIR"
ls -la "$RES_DIR"
