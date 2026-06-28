#!/usr/bin/env bash
# run_exp6_prm_reweighted_opsd.sh — PRM-reweighted OPSD (prms-as-teachers.md variant (c); RESULTS §8.4/§9).
#
# The "natural next experiment": reweight the OPSD teacher reverse-KL by a per-token process-importance
# (self-referential answer-info-gain g_t = log pi_T^answer - log pi_T^no-answer from the SAME 7B-SFT
# teacher), redistributing mass toward content/pivot tokens instead of bluntly clipping uncertain
# outliers. 4 arms at lam=1, seed 42, 4 GPUs in parallel (everything else == Exp 5):
#   A opsd-baseline    clip=1.0  prm=off  — OPSD baseline (Exp-5 4-seed p@1 ~0.252)
#   B opsd-noclip      clip=null prm=off  — does unclipped OPSD collapse? (§7.7 says yes)
#   C opsd-prm-noclip  clip=null prm=on   — THE CONJECTURE: reweighting REPLACES the blunt clip
#   D opsd-prm-clip    clip=1.0  prm=on   — do reweight + clip stack?
# Headline (prms-as-teachers.md line 45): does C >= A — can learned per-token importance replace the
# blunt per-token KL clip while keeping OPSD's gains?
#
#   sbatch /project/inniang/research/harness/run_exp6_prm_reweighted_opsd.sh
#
# Note: the PRM arms (C/D) run TWO frozen-teacher forwards per rollout batch (answer + no-answer) vs
# one for A/B — extra cost is at rollout time only (cached), training inner loop is unchanged.
#
#SBATCH --job-name=exp6-prm-opsd
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger01
#SBATCH --gres=gpu:h100_80gb:4
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

CONFIG="${CONFIG:-harness/configs/exp6_prm_reweighted_opsd.yaml}"
NUM_STEPS="${NUM_STEPS:-500}"
SEED="${SEED:-42}"
JOBID="${SLURM_JOB_ID:-local$$}"
CKPT="$RESEARCH_ROOT/harness/checkpoints"
RES_DIR="$RESEARCH_ROOT/results/exp6_prm_reweighted_opsd_${JOBID}"
mkdir -p harness/logs "$RES_DIR"

# arm i:  name | per_token_kl_clip | prm_reweight
NAMES=(opsd-baseline opsd-noclip opsd-prm-noclip opsd-prm-clip)
CLIPS=(1.0          null         null            1.0)
PRMS=(false         false        true            true)

echo "[exp6] node=$(hostname) job=$JOBID config=$CONFIG steps=$NUM_STEPS seed=$SEED res_dir=$RES_DIR"
echo "[exp6] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"

# ---------------------------------------------------------------------------
# Step 0 — fail-fast: base config + all 4 arm overrides validate; trainer imports.
# ---------------------------------------------------------------------------
$PYTHON - <<'PYEOF'
import yaml
from harness.config import ResearchConfig
raw = yaml.safe_load(open("harness/configs/exp6_prm_reweighted_opsd.yaml"))
assert raw["lam"] == 1.0 and raw["teacher"]["kind"] == "self" and raw["teacher"]["condition_on"] == "answer"
for clip, prm in [(1.0, False), (None, False), (None, True), (1.0, True)]:
    ResearchConfig(**{**raw, "per_token_kl_clip": clip, "prm_reweight": prm})  # raises if an arm is mis-specified
from harness import unified_trainer  # import smoke
print("[exp6] base config + all 4 arm overrides validate OK; trainer imports OK")
PYEOF

# ---------------------------------------------------------------------------
# Step 1 — train all 4 arms in parallel (one per GPU).
# ---------------------------------------------------------------------------
pids=(); labels=()
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"; clip="${CLIPS[$i]}"; prm="${PRMS[$i]}"
  label="${name}-s${SEED}-${JOBID}"
  logf="harness/logs/exp6_${i}_${name}-s${SEED}_${JOBID}.log"
  echo "[exp6] GPU $i -> $name (clip=$clip prm_reweight=$prm) -> $CKPT/$label ($logf)"
  CUDA_VISIBLE_DEVICES="$i" nohup "$PYTHON" -m harness.unified_trainer \
      --config "$CONFIG" \
      --set model_device_id=0 \
      --set seed="$SEED" \
      --set num_steps="$NUM_STEPS" \
      --set per_token_kl_clip="$clip" \
      --set prm_reweight="$prm" \
      --set ckpt_dir="$CKPT/$label" \
      --set wandb_run_name="$label" \
      > "$logf" 2>&1 &
  pids+=("$!"); labels+=("$name")
  sleep 3
done

echo "[exp6] launched ${#pids[@]} training runs; waiting..."
rc=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then echo "[exp6] TRAIN OK   GPU $j ${labels[$j]}"
  else code=$?; echo "[exp6] TRAIN FAIL GPU $j ${labels[$j]} (exit $code)"; rc=1; fi
done
[ "$rc" -eq 0 ] || { echo "[exp6] a training arm failed (rc=$rc); skipping eval"; exit "$rc"; }

# ---------------------------------------------------------------------------
# Step 2 — eval the 4 ckpts at the §8.1/Exp-5 protocol (T=0.6, 64x16, eval-seed 1e6), parallel.
# ---------------------------------------------------------------------------
echo "[exp6] training done; evaluating 4 ckpts (T=0.6, 64x16, k 1..16, eval-seed 1e6)"
epids=(); elabels=()
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"; label="${name}-s${SEED}-${JOBID}"
  elogf="harness/logs/exp6_eval_${i}_${name}-s${SEED}_${JOBID}.log"
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
  if wait "${epids[$j]}"; then echo "[exp6] EVAL OK   ${elabels[$j]}"
  else code=$?; echo "[exp6] EVAL FAIL ${elabels[$j]} (exit $code)"; rc=1; fi
done

# ---------------------------------------------------------------------------
# Step 3 — summary table.
# ---------------------------------------------------------------------------
echo "[exp6] SUMMARY"
$PYTHON - "$RES_DIR" "$SEED" <<'PYEOF'
import json, os, sys
res, seed = sys.argv[1], sys.argv[2]
def m(name):
    p = os.path.join(res, f"eval_{name}_s{seed}.json")
    if not os.path.exists(p): return None
    return json.load(open(p)).get("metrics_by_temp", {}).get("T=0.6")
rows = [
    ("A opsd-baseline  (clip=1.0 ,prm=off)", "opsd-baseline"),
    ("B opsd-noclip    (clip=None,prm=off)", "opsd-noclip"),
    ("C opsd-prm-noclip(clip=None,prm=on )", "opsd-prm-noclip"),
    ("D opsd-prm-clip  (clip=1.0 ,prm=on )", "opsd-prm-clip"),
]
print(f"{'arm':<40}{'p@1':>8}{'p@16':>8}{'tok_ent':>9}")
for desc, name in rows:
    t = m(name)
    if t is None: print(f"{desc:<40}{'MISSING':>8}"); continue
    print(f"{desc:<40}{t.get('eval/pass@1',float('nan')):>8.4f}{t.get('eval/pass@16',float('nan')):>8.4f}{t.get('eval/token_entropy',float('nan')):>9.4f}")
print()
print("reference: Exp-5 OPSD lam=1 clip=1.0 4-seed p@1 0.252+/-0.032 (§7.10); unclipped low-lam collapses (§7.7).")
print("HEADLINE: does C (reweight, no clip) >= A (clip, no reweight)? i.e. can learned per-token importance")
print("REPLACE the blunt per-token KL clip while keeping OPSD's gains? (prms-as-teachers.md line 45)")
PYEOF

echo "[exp6] DONE (rc=$rc). results in $RES_DIR"
ls -la "$RES_DIR"
exit "$rc"
