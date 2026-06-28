#!/usr/bin/env bash
# run_exp9_prm_teacher_b.sh — PRM-as-teacher variant (b): a SEPARATELY-TRAINED step-level PRM
# scoring process correctness (prms-as-teachers.md variant (b); roadmap §8.4 item 1).
#
# The roadmap's headline test: does a trained PRM (scoring process correctness, not answer-
# dependence) REPLACE the blunt per_token_kl_clip? Variant (c) (Exp 6) used a self-referential
# answer-info-gain and FAILED — a mass-preserving reweight CONCENTRATES the per-token KL tail the
# clip exists to BOUND (§7.11: no-clip+reweight 0.006, clip+reweight 0.157 vs 0.204 baseline).
# Variant (b)'s trained PRM scores PROCESS CORRECTNESS — a different signal that may have bounded
# mass. 4 arms at lam=1, seed 42, 4 GPUs in parallel (everything else == Exp 6):
#   A opsd-baseline       clip=1.0  prm=off   — Exp-6 arm A (should reproduce 0.204)
#   B opsd-noclip         clip=null prm=off   — Exp-6 arm B (should reproduce 0.007)
#   C prm-trained-noclip  clip=null prm=on    — THE CONJECTURE: trained-PRM replaces the clip
#   D prm-trained-clip    clip=1.0  prm=on    — do trained-PRM + clip stack?
#
# Headline (prms-as-teachers.md line 45, variant b): does C >= A — can a trained process-
# correctness PRM replace the blunt per-token KL clip while keeping OPSD's gains? And does D > A
# (does trained-PRM + clip beat the clipped logit-teacher baseline)?
#
#   sbatch /project/inniang/research/harness/run_exp9_prm_teacher_b.sh
#
# Prerequisite: the trained PRM checkpoint must exist at $PRM_MODEL_PATH (run
#   sbatch harness/run_exp9_train_prm.sh
# first). The launcher fails fast if it's missing.
#
# Note: the PRM arms (C/D) run TWO frozen-teacher forwards per rollout batch (the 7B-SFT
# answer-conditioned teacher for logits + the 1B PRM for step scores) vs one for A/B — extra cost
# is at rollout time only (cached), training inner loop is unchanged.
#
#SBATCH --job-name=exp9-prm-teacher-b
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger04
#SBATCH --gres=gpu:rtx_6000:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=16:00:00
#SBATCH --output=/project/inniang/research/harness/logs/%x-%j.out

set -euo pipefail

RESEARCH_ROOT=/project/inniang/research
PYTHON=/project/inniang/.venv/bin/python
cd "$RESEARCH_ROOT"

export PYTHONPATH="$RESEARCH_ROOT:${PYTHONPATH:-}"
export HF_HOME=/project/inniang/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_PROJECT="${WANDB_PROJECT:-distill-harness}"
export WANDB_DIR="$RESEARCH_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="${CONFIG:-harness/configs/exp9_prm_teacher_b.yaml}"
NUM_STEPS="${NUM_STEPS:-500}"
SEED="${SEED:-42}"
JOBID="${SLURM_JOB_ID:-local$$}"
CKPT="$RESEARCH_ROOT/harness/checkpoints"
RES_DIR="$RESEARCH_ROOT/results/exp9_prm_teacher_b_${JOBID}"
PRM_MODEL_PATH="${PRM_MODEL_PATH:-$CKPT/prm_step_level_s4242}"
mkdir -p harness/logs "$RES_DIR"

# Fail fast if the trained PRM checkpoint is missing.
if [[ ! -f "$PRM_MODEL_PATH/prm_head.pt" ]]; then
    echo "[exp9] FAIL: trained PRM checkpoint not found at $PRM_MODEL_PATH/prm_head.pt"
    echo "[exp9] Run sbatch harness/run_exp9_train_prm.sh first (Stage 1+2 trains the PRM)."
    exit 1
fi
echo "[exp9] using trained PRM: $PRM_MODEL_PATH"

# arm i:  name | per_token_kl_clip | prm_reweight | prm_source | prm_model_path
NAMES=(opsd-baseline opsd-noclip prm-trained-noclip prm-trained-clip)
CLIPS=(1.0          null        null              1.0)
PRMS=(false         false       true              true)
SRCS=(answer_info_gain answer_info_gain trained trained)
MPATHS=(""          ""          "$PRM_MODEL_PATH" "$PRM_MODEL_PATH")

echo "[exp9] node=$(hostname) job=$JOBID config=$CONFIG steps=$NUM_STEPS seed=$SEED res_dir=$RES_DIR"
echo "[exp9] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"

# ---------------------------------------------------------------------------
# Step 0 — fail-fast: base config + all 4 arm overrides validate; trainer imports.
# ---------------------------------------------------------------------------
$PYTHON - <<'PYEOF'
import yaml
from harness.config import ResearchConfig
raw = yaml.safe_load(open("harness/configs/exp9_prm_teacher_b.yaml"))
assert raw["lam"] == 1.0 and raw["teacher"]["kind"] == "self" and raw["teacher"]["condition_on"] == "answer"
PRM_PATH = __import__("os").environ.get("PRM_MODEL_PATH", "harness/checkpoints/prm_step_level_s4242")
for clip, prm, src, mp in [(1.0, False, "answer_info_gain", None), (None, False, "answer_info_gain", None),
                           (None, True, "trained", PRM_PATH), (1.0, True, "trained", PRM_PATH)]:
    ResearchConfig(**{**raw, "per_token_kl_clip": clip, "prm_reweight": prm,
                       "prm_source": src, "prm_model_path": mp})  # raises if an arm is mis-specified
from harness import unified_trainer  # import smoke
from harness.teachers import PRMTeacher  # variant (b) teacher
print("[exp9] base config + all 4 arm overrides validate OK; trainer + PRMTeacher import OK")
PYEOF

# ---------------------------------------------------------------------------
# Step 1 — train all 4 arms in parallel (one per GPU).
# ---------------------------------------------------------------------------
pids=(); labels=()
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"; clip="${CLIPS[$i]}"; prm="${PRMS[$i]}"; src="${SRCS[$i]}"; mpath="${MPATHS[$i]}"
  label="${name}-s${SEED}-${JOBID}"
  logf="harness/logs/exp9_${i}_${name}-s${SEED}_${JOBID}.log"
  echo "[exp9] GPU $i -> $name (clip=$clip prm_reweight=$prm prm_source=$src) -> $CKPT/$label ($logf)"
  CUDA_VISIBLE_DEVICES="$i" nohup "$PYTHON" -m harness.unified_trainer \
      --config "$CONFIG" \
      --set model_device_id=0 \
      --set seed="$SEED" \
      --set num_steps="$NUM_STEPS" \
      --set per_token_kl_clip="$clip" \
      --set prm_reweight="$prm" \
      --set prm_source="$src" \
      --set prm_model_path="$mpath" \
      --set ckpt_dir="$CKPT/$label" \
      --set wandb_run_name="$label" \
      > "$logf" 2>&1 &
  pids+=("$!"); labels+=("$name")
  sleep 3
done

echo "[exp9] launched ${#pids[@]} training runs; waiting..."
rc=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then echo "[exp9] TRAIN OK   GPU $j ${labels[$j]}"
  else code=$?; echo "[exp9] TRAIN FAIL GPU $j ${labels[$j]} (exit $code)"; rc=1; fi
done
[ "$rc" -eq 0 ] || { echo "[exp9] a training arm failed (rc=$rc); skipping eval"; exit "$rc"; }

# ---------------------------------------------------------------------------
# Step 2 — eval the 4 ckpts at the §8.1/Exp-5 protocol (T=0.6, 64x16, eval-seed 1e6), parallel.
# ---------------------------------------------------------------------------
echo "[exp9] training done; evaluating 4 ckpts (T=0.6, 64x16, k 1..16, eval-seed 1e6)"
epids=(); elabels=()
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"; label="${name}-s${SEED}-${JOBID}"
  elogf="harness/logs/exp9_eval_${i}_${name}-s${SEED}_${JOBID}.log"
  CUDA_VISIBLE_DEVICES="$i" nohup "$PYTHON" -m harness.eval_passk \
      --ckpt "$CKPT/$label" --task gsm_symbolic \
      --n-prompts 64 --n-samples 16 --k 1,2,4,8,16 --temps 0.6 \
      --max-new-tokens 1024 --eval-seed 1000000 --no-self-bleu --gen-batch-size 8 \
      --out "$RES_DIR/eval_${name}_s${SEED}.json" \
      > "$elogf" 2>&1 &
  epids+=("$!"); elabels+=("$name")
  sleep 3
done
for j in "${!epids[@]}"; do
  if wait "${epids[$j]}"; then echo "[exp9] EVAL OK   ${elabels[$j]}"
  else code=$?; echo "[exp9] EVAL FAIL ${elabels[$j]} (exit $code)"; rc=1; fi
done

# ---------------------------------------------------------------------------
# Step 3 — summary table.
# ---------------------------------------------------------------------------
echo "[exp9] SUMMARY"
$PYTHON - "$RES_DIR" "$SEED" <<'PYEOF'
import json, os, sys
res, seed = sys.argv[1], sys.argv[2]
def m(name):
    p = os.path.join(res, f"eval_{name}_s{seed}.json")
    if not os.path.exists(p): return None
    return json.load(open(p)).get("metrics_by_temp", {}).get("T=0.6")
rows = [
    ("A opsd-baseline       (clip=1.0 ,prm=off)", "opsd-baseline"),
    ("B opsd-noclip         (clip=None,prm=off)", "opsd-noclip"),
    ("C prm-trained-noclip  (clip=None,prm=on )", "prm-trained-noclip"),
    ("D prm-trained-clip    (clip=1.0 ,prm=on )", "prm-trained-clip"),
]
print(f"{'arm':<40}{'p@1':>8}{'p@16':>8}{'tok_ent':>9}")
for desc, name in rows:
    t = m(name)
    if t is None: print(f"{desc:<40}{'MISSING':>8}"); continue
    print(f"{desc:<40}{t.get('eval/pass@1',float('nan')):>8.4f}{t.get('eval/pass@16',float('nan')):>8.4f}{t.get('eval/token_entropy',float('nan')):>9.4f}")
print()
print("reference: Exp-6 variant (c) 4-seed p@1: A clip/no-rw 0.204, B noclip/no-rw 0.007,")
print("           C noclip+answer-info-gain-rw 0.006 (collapsed harder), D clip+rw 0.157 (reweight hurts).")
print("HEADLINE (variant b): does C (trained-PRM, no clip) >= A (clip, no reweight) = 0.204?")
print("           does D (trained-PRM + clip) > A = 0.204 (trained-PRM + clip beats logit-teacher)?")
print("           roadmap §8.4 item 1 prediction: C collapses like (c); D matches or beats A.")
PYEOF

echo "[exp9] DONE (rc=$rc). results in $RES_DIR"
ls -la "$RES_DIR"
exit "$rc"
