#!/usr/bin/env bash
# run_research.sh — launcher for the unified (alpha, lambda, pi_T) trainer (SCAFFOLD).
#
# Patterned on /project/inniang/vibe/code/scripts/run_all_policy_gradients.sh (that one has the
# multi-run loop + offline-spool fallback); this is the single-config research version. The whole
# stack is self-contained under research/ (harness/ + the vendored policy_gradients/). Logging is
# Weights & Biases (`pip install wandb` + `wandb login` once; set WANDB_PROJECT). NOT MLRunX.
#
# Hardware: RTX 6000 Ada, 48 GB. OLMo-2-1B full fine-tune fits with gradient checkpointing; OLMo-2-1B
# student ← OLMo-2-7B-Instruct teacher (bf16, frozen) also fits (~32 GB). For a 13B teacher: 8-bit it
# (needs `pip install bitsandbytes`); 32B teacher: 4-bit, or its own card / a vLLM endpoint. See
# harness/README.md "Fitting bigger OLMo-2 on the RTX GPU".
#
# Usage (run from research/):
#   python -m harness.unified_trainer --config harness/configs/rl_grpo.yaml      # local, 1 GPU
#   sbatch harness/run_research.sh harness/configs/opd.yaml                       # SLURM
#   NPROC=4 sbatch --gres=gpu:4 harness/run_research.sh harness/configs/opd.yaml  # multi-GPU
#
#SBATCH --job-name=distill-harness
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=harness/logs/%x-%j.out

set -euo pipefail

# ABSOLUTE paths only — ${BASH_SOURCE[0]} resolves to the SLURM spool dir inside a job (Key Lesson).
RESEARCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # -> /project/inniang/research
cd "$RESEARCH_ROOT"

CONFIG="${1:-harness/configs/rl_grpo.yaml}"
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 1; }

# --- Python env (project venv) ---
PYTHON="${PYTHON:-/project/inniang/.venv/bin/python}"

# research/ on the path so `harness.*` and the vendored `policy_gradients` import cleanly even if
# launched from elsewhere (cd above already covers `python -m`, this is belt-and-braces):
export PYTHONPATH="$RESEARCH_ROOT:${PYTHONPATH:-}"

# --- Weights & Biases (replaces MLRunX) ---
export WANDB_PROJECT="${WANDB_PROJECT:-distill-harness}"
# export WANDB_MODE=offline      # uncomment on a node with no internet; `wandb sync wandb/latest-run` later.

# --- HuggingFace cache (compute nodes have no internet — use the local snapshot cache) ---
export HF_HOME="${HF_HOME:-/project/inniang/hf-cache}"
# export TRANSFORMERS_OFFLINE=1  # uncomment once the OLMo-2 snapshots are present under $HF_HOME.

mkdir -p harness/logs

NPROC="${NPROC:-1}"
echo "[run_research] config=$CONFIG  nproc=$NPROC  python=$PYTHON  wandb_project=$WANDB_PROJECT  research_root=$RESEARCH_ROOT"

if [ "$NPROC" -gt 1 ]; then
  exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
       -m harness.unified_trainer --config "$CONFIG"
else
  exec "$PYTHON" -m harness.unified_trainer --config "$CONFIG"
fi
