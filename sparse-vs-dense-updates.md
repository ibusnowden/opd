# Parameter-Space Geometry: Sparse (RL) vs Dense (SFT) vs OPD Updates

_Status: idea · Started: 2026-05-11 · Last updated: 2026-05-11_
_Source: `research/ideas.md` — §"RL has sparser parameter updates", §5 "RL: sparse, but saved by destructive interference"._

## Introduction
**Question.** RL fine-tuning has been reported to modify a small full-rank *subnetwork* (Mukherjee et al. 2025), and SFT updates are denser and more *redundant* — prune the updated parameters and RL degrades much faster than SFT (Yuan et al. as cited in the post). Two things to pin down: (1) does this replicate on our setups, and (2) **where does on-policy distillation sit** on the sparse↔dense / redundant↔essential axis? Is OPD "SFT-shaped" (dense, diffuse) because the teacher is family-calibrated, or "OPSD-shaped" (concentrated) when the teacher is shifted?

**Why it matters.** The post's geometric story (§5–§6) is: RL is *sparse but unbiased* (noise vectors cancel — "destructive interference"), SFT is *dense, biased, but diffuse*, OPSD is *dense, biased, and concentrated* (hence needs clipping). If OPD's update geometry really tracks teacher choice, then "what teacher" ≈ "what update shape" — and the meta-algorithm in `meta-algorithm-alpha-lambda.md` is choosing a point in update-geometry space, not just a loss. The *output-distribution* shadow of "RL = a small but *essential* re-weighting (moved mass between modes the base already had, didn't add modes)" is "pass@1 up, pass@k flat/↓ vs. the un-RL'd init" — i.e. this is [[pass-at-k-vs-pass-at-1]] from the parameter side; if the two views agree (sparse-essential ⇔ over-sharpened), that's a strong cross-check.

**Prior work.** Mukherjee et al. 2025 (RL finetunes small subnetworks); Yuan et al. (SFT update redundancy / pruning); Lu & Thinking Machines 2025 (OPD); Zhao et al. 2026 (OPSD); Yue et al. 2025 / ProRL (the output-side version — pass@k vs pass@1; see [[pass-at-k-vs-pass-at-1]]).

## Data
- **Math reasoning** (`reasoning_gym`/AIME-style, already wired into `policy_gradients`) and **Minimal Code Editing** (from `opd-different-teachers.md`) as two domains.
- Held-out general benchmark (LiveCodeBench / a small general eval) for the forgetting axis.
- Logged: per-parameter |Δθ| after training (RL / SFT / OPD-real-teacher / OPSD); singular-value spectra of layer-wise ΔW; performance after masking the bottom-k% / top-k% of |Δθ|.

## Method and model
**Setup.** Train the *same base* to comparable on-task performance with four recipes: RL (GRPO), SFT, OPD (real same-family teacher), OPSD (answer-conditioned self-teacher, with clipping). Snapshot Δθ for each. Then:
1. **Sparsity / rank.** Histogram of |Δθ|; effective rank of ΔW per layer; which submodules move (attn vs MLP, which layers).
2. **Pruning-degradation curves.** Zero out the smallest p% of |Δθ| (and separately the largest p%), re-evaluate; plot performance vs p for each recipe. RL expected to degrade fastest under bottom-out pruning (each update matters); SFT slowest (redundant).
3. **OPD placement.** Does OPD-real-teacher's curve track SFT's? Does OPSD's track... something in between, or its own thing?

**Modules.** RL run = `policy_gradients` harness as-is. New: SFT loop, OPD/OPSD losses + teacher-logprob pass, a Δθ snapshot + masking utility, spectrum analysis script. LoRA reference: `vibe/autoresearch/chal/ablations/train_ablation.py::AttentionLoRA` (if we also want a low-rank-constrained arm). W&B logging + SLURM as usual.

**Ablations.** Matched vs unmatched compute; with/without KL-to-base penalty on the RL run; LoRA-constrained vs full fine-tune for each recipe; teacher size for OPD.

## Evaluation *(proposed — no results yet)*
| Recipe | |Δθ| sparsity | ΔW effective rank | Perf @ prune bottom-50% | Perf @ prune bottom-90% | Forgetting (held-out Δ) |
|---|---|---|---|---|---|---|
| RL (GRPO) | sparse (expected) | full-rank | ≈ kept | drops fast (expected) | small |
| SFT | dense | full-rank | ≈ kept | ≈ kept (redundant, expected) | large |
| OPD (real teacher) | ? (expect SFT-like, diffuse) | ? | ? | ? | small (expected) |
| OPSD (clipped) | ? (expect more concentrated) | ? | ? | ? | moderate |

- **Expected:** replicate RL-sparse / SFT-dense-redundant; OPD-real-teacher closer to SFT in *diffuseness* but with less forgetting (on-policy state coverage); OPSD's geometry more concentrated, consistent with why it needs clipping.
- **Where it breaks:** "subnetwork" claims are sensitive to optimizer / lr / how you threshold; matched-performance across recipes is hard; LoRA arms confound rank with the recipe.

## Takeaways *(predictions)*
- Likely conclusion: update *geometry* (sparsity, concentration, redundancy) is largely a function of the teacher/target, not the algorithm name — which is exactly the meta-algorithm's premise.
- Risk: results may not generalize beyond the tested model sizes / domains (the post itself flags this skepticism).
- Open: can you *induce* RL-like sparse-but-essential updates in a distillation method by reweighting the per-token signal toward causally-important tokens? (links to `per-token-kl-pivot-vs-style.md`, `prms-as-teachers.md`.)
