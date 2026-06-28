# Research — Skills & Project Index

A running log of research projects / experiments ("fun ideas"). Each project gets its own
writeup at `research/<project-name>.md` (or `research/<project-name>/README.md` if it grows
its own files), following the **Writeup Template** at the bottom of this file. Add a row to the
**Index** below whenever you start one.

Keep writeups tight: tables over prose, concrete numbers, ✓/✗ markers — same flavor as the
`*_FINAL_SUMMARY.md` notes under `vibe/autoresearch/`.

> Current batch (#1–#14) is sourced from `ideas.md` — Brown & Claude, *"SFT, RL, and On-Policy
> Distillation Through a Distributional Lens"* (Apr 30 2026). All are research **proposals**
> (status `idea`); their Evaluation sections describe *expected* outcomes, not measured results.
>
> **Conventions for this folder's experiments:**
> - Experiment tracking: **Weights & Biases** (not MLRunX).
> - Model family: **OLMo-2** (`allenai/OLMo-2-*`) — shared tokenizer across 1B/7B/13B/32B, so
>   same-family teacher setups (OPD/OPSD/SDFT) are clean (no tokenizer-mismatch tax). **Start
>   small** (OLMo-2-1B); scale the (teacher, student) pair up only once the loop runs end-to-end.
> - Hardware: one **RTX 6000 Ada (48 GB)**. OLMo-2-1B full-FT + OLMo-2-7B-Instruct bf16 teacher
>   fits (~32 GB); 13B teacher → 8-bit, 32B → 4-bit / vLLM-served. See `harness/README.md`.
> - Code: all under `research/`, self-contained — `research/harness/` (the unified (α, λ, π_T)
>   trainer; the **RL corner λ=0 is implemented** — `unified_trainer.py::_run_rl_loop` — the λ>0
>   corners / SFT / off-policy / LoRA-quant / DDP are stubbed) + `research/policy_gradients/` (the
>   RL/PG reference **vendored** from `vibe/code/policy_gradients/`, `mlrunx` removed). Eager imports
>   only (no sys.path tricks, no lazy imports). Run: `python -m harness.unified_trainer --config
>   harness/configs/rl_grpo.yaml --set model_name=allenai/OLMo-2-0425-1B-Instruct`. See `harness/README.md`.

---

## Index

| # | Project | Status | One-liner | Writeup |
|---|---------|--------|-----------|---------|
| 1 | OPD from different teachers | idea | Does an OPD student depend on its teacher (SFT vs RL), or is on-policy data the load-bearing ingredient? Replicate + extend the post's core experiment. | [md](opd-different-teachers.md) |
| 2 | Why student > teacher in OPD | idea | Disentangle on-policy state coverage vs KL-matching ≠ reward-maximization as causes of OPD students surpassing the teacher. | [md](student-beats-teacher-opd.md) |
| 3 | Entropy collapse: OPD vs RL | idea | Quantify the sharper entropy/diversity collapse under OPD's reverse-KL vs RL; do entropy bonus / per-token clipping recover it? | [md](entropy-collapse-opd-vs-rl.md) |
| 4 | Per-token KL: pivot vs style | idea | Where does the per-token (self-)distillation KL concentrate? Is per-token clipping load-bearing or a band-aid for a bad teacher? | [md](per-token-kl-pivot-vs-style.md) |
| 5 | Sparse (RL) vs dense (SFT) updates | idea | Replicate RL-sparse / SFT-dense-redundant parameter-update geometry; place OPD/OPSD on the sparsity↔concentration axis via pruning curves. | [md](sparse-vs-dense-updates.md) |
| 6 | The (α, λ, π_T) meta-algorithm | idea | Build the unified token-level PG that contains SFT/RL/OPD/OPSD as corners; reproduce the corners and empirically probe the interior. | [md](meta-algorithm-alpha-lambda.md) |
| 7 | Per-task hint search (GEPA-style) | idea | Cheap inner loop: search teacher-conditioning hints over the Lagrangian E[Δreward] − β·KL on the current student — no retraining. | [md](per-task-hint-search-gepa.md) |
| 8 | Distribution-level hint rewriter | idea | Train a model that rewrites a big "bad" hint (answer / demo / whitebox) into a minimal one that's surgical for the teacher; amortizes the per-task search. | [md](hint-rewriter-distillation.md) |
| 9 | Co-evolving hint-writer + student | idea | Self-prompt-optimization online RL: hint-writer and student co-evolve with an adaptive KL-budget controller (smoothed minmax). | [md](co-evolving-hint-writer.md) |
| 10 | Hint-writer trained with RL | idea | Train a standalone, scoped hint-writer directly with RL on `correctness-Δ × (1 − KL-Δ)` — a shippable teacher-construction artifact. | [md](hint-writer-rl.md) |
| 11 | Expert RL + OPD | idea | Layer a teacher-KL term on top of locally-optimal RL (DeepSeek-V4 style); λ sweep/schedule for OPD's speed + RL's verifier-bounded ceiling. | [md](expert-rl-plus-opd.md) |
| 12 | PRMs as teachers | idea | Reframe PRMs as teachers; blend per-rollout self-distillation guidance with outcome rewards; PRM-reweighted OPSD to kill the concentration problem. | [md](prms-as-teachers.md) |
| 13 | SFT → RL tipping point | idea | Locate the rollout-compute split where teacher-SFT stops paying off vs RL; confirm SFT-RS shifts the curve up without reshaping it; predict the handoff from cheap signals. | [md](sft-rl-tipping-point.md) |
| 14 | Cross-family teacher tax | idea | Measure how much of cross-family SFT goes to surface form vs competence; the tokenizer-mismatch cost; why on-policy distillation is same-family-gated. | [md](cross-family-teacher-tax.md) |
| 15 | Pass@k vs pass@1: sharpen or expand? | idea | When the RL corner lifts `reward` (≈ pass@1), does pass@k stay flat / fall vs the un-RL'd init? Make pass@k + distinct-n / token-entropy a first-class eval; place SFT/RL/OPD/OPSD on the sharpening↔coverage axis (Yue et al. 2025 vs ProRL, scaled down). | [md](pass-at-k-vs-pass-at-1.md) |

**Status legend:** `idea` · `in progress` · `paused` · `done`

---

## Writeup Template

> Copy everything in this block into a new `research/<project-name>.md`, delete the prompts,
> fill it in. Five sections, in order.

```markdown
# <Project Name>

_Status: idea · Started: YYYY-MM-DD · Last updated: YYYY-MM-DD_

## Introduction
What is the challenge — and why does it matter?
- What problem / question are we tackling?
- Why is it interesting or worth the time? What would success unlock?
- Prior work / inspiration (papers, repos, internal notes).

## Data
What is the data? Include statistics and examples.
- Source(s), size, splits.
- Key statistics (counts, token/char counts, label distribution, sequence lengths, ...).
- A couple of concrete examples (raw samples — show, don't just describe).
- Preprocessing / filtering applied.

## Method and model
Give an overview of the model, then break down the individual modules.
- High-level picture: inputs → ... → outputs (a diagram or bullet flow).
- Module-by-module breakdown (each component, what it does, key design choices).
- Training setup: objective/loss, optimizer, hyperparameters, hardware, infra
  (e.g. W&B run IDs, SLURM job).
- Ablations / variants tried.

## Evaluation
What are the results?
- Metrics + a results table (model/variant · metric(s) · notes).
- Comparison to baselines / prior numbers.
- Plots or curves (link the files).
- Where it works, where it breaks.

## Takeaways
What are the conclusions, and what did we learn from the experiments / project?
- Headline conclusion(s).
- What surprised us; what we'd do differently.
- Open questions / next experiments.
```
