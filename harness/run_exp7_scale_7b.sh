#!/usr/bin/env bash
# run_exp7_scale_7b.sh — Experiment 7: the SCALE test (RESULTS §8.4/§9 "what genuinely stays open").
#
# NATIVE 7B-SFT student <- 7B-Instruct teacher. Does on-policy pure-OPD finally REVIVE at scale, or does
# the §7.6 dead-zone attractor persist? (§8.1 follow-up 2 ruled out competence-at-1B; this is true scale.)
# Full-FT 7B via bnb 8-bit Adam (cfg.fit.optimizer_8bit; embeddings 32-bit) -> fits one 80 GB H100, with
# the frozen teacher on a SECOND card. 2 H100s/arm; 4 arms fill itiger01's 8 H100s, all in parallel:
#   A onpolicy-opd-noclip   alpha=1 lam=1   clip=null            -- DEAD at 1B (0.008): revive at 7B? (KEY)
#   B offpolicy-revkd       alpha=0 lam=1   clip=null +buffer    -- ALIVE at 1B (0.298): off-policy control
#   C onpolicy-opd-clip     alpha=1 lam=1   clip=1.0             -- does the clip rescue on-policy at 7B?
#   D onpolicy-lowlam-clip  alpha=1 lam=0.10 clip=1.0 grpo       -- best-1B recipe (0.709): hold at 7B?
#
#   sbatch /project/inniang/research/harness/run_exp7_scale_7b.sh
#   NUM_STEPS=4 sbatch /project/inniang/research/harness/run_exp7_scale_7b.sh   # short test
#
#SBATCH --job-name=exp7-scale-7b
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger01
#SBATCH --gres=gpu:h100_80gb:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=36:00:00
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

CONFIG="${CONFIG:-harness/configs/exp7_scale_7b.yaml}"
NUM_STEPS="${NUM_STEPS:-250}"
SEED="${SEED:-42}"
OFFP_FILE="rft_data/gsm_symbolic_from_OLMo-2-1124-7B-Instruct_seed4242_ALL.jsonl"
JOBID="${SLURM_JOB_ID:-local$$}"
CKPT="$RESEARCH_ROOT/harness/checkpoints"
RES_DIR="$RESEARCH_ROOT/results/exp7_scale_7b_${JOBID}"
mkdir -p harness/logs "$RES_DIR"

# arm i:  name | alpha | lam | per_token_kl_clip | offpolicy buffer ("" = on-policy)
NAMES=(onpolicy-opd-noclip offpolicy-revkd      onpolicy-opd-clip onpolicy-lowlam-clip)
ALPHAS=(1.0                0.0                  1.0               1.0)
LAMS=(1.0                  1.0                  1.0               0.10)
CLIPS=(null                null                 1.0               1.0)
OFFP=(""                   "$OFFP_FILE"         ""                "")

echo "[exp7] node=$(hostname) job=$JOBID config=$CONFIG steps=$NUM_STEPS seed=$SEED res_dir=$RES_DIR"
echo "[exp7] $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' | ')"

# ---------------------------------------------------------------------------
# Step 0 — fail-fast: base config + all 4 arm overrides validate; trainer imports; bnb present.
# ---------------------------------------------------------------------------
$PYTHON - "$OFFP_FILE" <<'PYEOF'
import os, sys, yaml
import bitsandbytes as bnb  # noqa: F401  (8-bit Adam must be importable in-job)
from harness.config import ResearchConfig
offp = sys.argv[1]
assert os.path.exists(offp), f"off-policy buffer missing: {offp}"
raw = yaml.safe_load(open("harness/configs/exp7_scale_7b.yaml"))
assert raw["model_name"].endswith("7B-SFT"), raw["model_name"]
assert raw["teacher"]["model_name"].endswith("7B-Instruct"), raw["teacher"]["model_name"]
assert raw["fit"]["optimizer_8bit"] is True
arms = [
    dict(alpha=1.0, lam=1.0,  per_token_kl_clip=None),                                       # A
    dict(alpha=0.0, lam=1.0,  per_token_kl_clip=None, offpolicy_teacher_states=offp),        # B
    dict(alpha=1.0, lam=1.0,  per_token_kl_clip=1.0),                                         # C
    dict(alpha=1.0, lam=0.10, per_token_kl_clip=1.0),                                         # D
]
for a in arms:
    ResearchConfig(**{**raw, **a})  # raises if an arm is mis-specified
from harness import unified_trainer  # import smoke
print("[exp7] base config + all 4 arm overrides validate OK; bnb + trainer import OK")
PYEOF

# ---------------------------------------------------------------------------
# Step 1 — train all 4 arms in parallel; each arm = student (card 0) + teacher (card 1) of its GPU pair.
# ---------------------------------------------------------------------------
pids=(); labels=()
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"; alpha="${ALPHAS[$i]}"; lam="${LAMS[$i]}"; clip="${CLIPS[$i]}"; offp="${OFFP[$i]}"
  g0=$((2*i)); g1=$((2*i+1))                       # this arm's dedicated GPU pair
  label="${name}-s${SEED}-${JOBID}"
  logf="harness/logs/exp7_${i}_${name}-s${SEED}_${JOBID}.log"
  extra=()
  [ -n "$offp" ] && extra+=(--set "offpolicy_teacher_states=$offp")
  echo "[exp7] GPUs $g0,$g1 -> $name (alpha=$alpha lam=$lam clip=$clip offp=${offp:-none}) -> $CKPT/$label ($logf)"
  CUDA_VISIBLE_DEVICES="$g0,$g1" nohup "$PYTHON" -m harness.unified_trainer \
      --config "$CONFIG" \
      --set model_device_id=0 \
      --set teacher.device_id=1 \
      --set seed="$SEED" \
      --set num_steps="$NUM_STEPS" \
      --set alpha="$alpha" \
      --set lam="$lam" \
      --set per_token_kl_clip="$clip" \
      "${extra[@]}" \
      --set ckpt_dir="$CKPT/$label" \
      --set wandb_run_name="$label" \
      > "$logf" 2>&1 &
  pids+=("$!"); labels+=("$name")
  sleep 15                                          # stagger HF snapshot reads / CUDA init across arms
done

echo "[exp7] launched ${#pids[@]} training runs; waiting..."
rc=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then echo "[exp7] TRAIN OK   pair $j ${labels[$j]}"
  else code=$?; echo "[exp7] TRAIN FAIL pair $j ${labels[$j]} (exit $code)"; rc=1; fi
done
[ "$rc" -eq 0 ] || { echo "[exp7] a training arm failed (rc=$rc); skipping eval"; exit "$rc"; }

# ---------------------------------------------------------------------------
# Step 2 — eval the 4 ckpts at the §8.1/Exp-5 protocol (T=0.6, 64x16, eval-seed 1e6), parallel.
# Eval uses the first card of each arm's pair (one model on one GPU, no teacher needed).
# ---------------------------------------------------------------------------
echo "[exp7] training done; evaluating 4 ckpts (T=0.6, 64x16, k 1..16, eval-seed 1e6)"
epids=(); elabels=()
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"; label="${name}-s${SEED}-${JOBID}"; g0=$((2*i))
  elogf="harness/logs/exp7_eval_${i}_${name}-s${SEED}_${JOBID}.log"
  CUDA_VISIBLE_DEVICES="$g0" nohup "$PYTHON" -m harness.eval_passk \
      --ckpt "$CKPT/$label" --task gsm_symbolic \
      --n-prompts 64 --n-samples 16 --k 1,2,4,8,16 --temps 0.6 \
      --max-new-tokens 1024 --eval-seed 1000000 --no-self-bleu --gen-batch-size 8 \
      --out "$RES_DIR/eval_${name}_s${SEED}.json" \
      > "$elogf" 2>&1 &
  epids+=("$!"); elabels+=("$name")
  sleep 5
done
for j in "${!epids[@]}"; do
  if wait "${epids[$j]}"; then echo "[exp7] EVAL OK   ${elabels[$j]}"
  else code=$?; echo "[exp7] EVAL FAIL ${elabels[$j]} (exit $code)"; rc=1; fi
done

# ---------------------------------------------------------------------------
# Step 3 — summary table + the scale headline.
# ---------------------------------------------------------------------------
echo "[exp7] SUMMARY"
$PYTHON - "$RES_DIR" "$SEED" <<'PYEOF'
import json, os, sys
res, seed = sys.argv[1], sys.argv[2]
def m(name):
    p = os.path.join(res, f"eval_{name}_s{seed}.json")
    if not os.path.exists(p): return None
    return json.load(open(p)).get("metrics_by_temp", {}).get("T=0.6")
rows = [
    ("A onpolicy-opd-noclip  (a=1 lam=1   clip=null)", "onpolicy-opd-noclip",  "1B: 0.008 (dead)"),
    ("B offpolicy-revkd      (a=0 lam=1   clip=null)", "offpolicy-revkd",      "1B: 0.298 (alive)"),
    ("C onpolicy-opd-clip    (a=1 lam=1   clip=1.0 )", "onpolicy-opd-clip",    "1B: ~0.05 (dead)"),
    ("D onpolicy-lowlam-clip (a=1 lam=0.1 clip=1.0 )", "onpolicy-lowlam-clip", "1B: 0.709 (best)"),
]
print(f"{'arm':<48}{'p@1':>8}{'p@16':>8}{'tok_ent':>9}   {'1B reference'}")
for desc, name, ref in rows:
    t = m(name)
    if t is None: print(f"{desc:<48}{'MISSING':>8}   {ref}"); continue
    print(f"{desc:<48}{t.get('eval/pass@1',float('nan')):>8.4f}{t.get('eval/pass@16',float('nan')):>8.4f}"
          f"{t.get('eval/token_entropy',float('nan')):>9.4f}   {ref}")
print()
print("HEADLINE (scale): does on-policy pure-OPD (A) revive at native 7B, or stay in the §7.6 dead zone?")
print("  * A >> 0.01 (near B/C/D)  -> 1B collapse was a small-student/capacity artifact; on-policy rescued at scale.")
print("  * A ~ 0.01 (B still wins) -> on-policy reverse-KL instability is SCALE-ROBUST; strongly generalizes §8.1.")
print("Secondary: does the clip alone (C) rescue on-policy at 7B where it didn't at 1B? does the best-1B recipe (D) hold?")
PYEOF

echo "[exp7] DONE (rc=$rc). results in $RES_DIR"
ls -la "$RES_DIR"
exit "$rc"
