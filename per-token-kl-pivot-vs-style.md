# Per-Token KL in (Self-)Distillation: Pivot Tokens vs Style Tokens

_Status: idea · Started: 2026-05-11 · Last updated: 2026-05-11_
_Source: `research/ideas.md` — §"On-Policy Distillation: Pseudo RL" (OPSD per-token KL analysis), §5 "OPSD: dense, biased, and concentrated"._

## Introduction
**Question.** In on-policy (self-)distillation, where does the per-token reverse-KL signal actually concentrate? The OPSD paper found *style / pivot* tokens ("wait", "alright") can have higher KL than content tokens ("exponent", "logarithm") — and that updating too aggressively on them causes collapse within ~100 steps unless you add per-token point-wise KL clipping. Does this hold across teachers (self-with-answer-hint vs self-with-demo vs a real bigger teacher)? Is the clipping load-bearing, or is it a band-aid for a badly-chosen teacher?

**Why it matters.** The post's §5 argument is that OPSD is the unique "dense + biased + concentrated" method, and concentration is the failure mode. If we can *characterize* the concentration (which tokens, how heavy-tailed, how it depends on the privileged info given to the teacher) we know what an "ideal teacher" must avoid — directly feeding `hint-rewriter-distillation.md` and `meta-algorithm-alpha-lambda.md`.

**Prior work.** Zhao et al. 2026 (OPSD + per-token clipping); Shenfeld et al. 2026 (SDFT, demonstration-conditioned teacher); Lu & Thinking Machines 2025 (OPD); Diao et al. (SFT has many low-prob low-entropy tokens) and Lai et al. 2025 (data-dependent regularization) as cited in the post.

## Data
- **Long-CoT math** (`reasoning_gym` / AIME-style) — long rollouts where a single substitution/observation is the pivot; ideal for measuring KL concentration along a chain.
- A tagged sample of rollouts where the pivot span is hand-labeled (or proxy-labeled: the token whose intervention most changes final correctness).
- Logged: per-token reverse-KL teacher→student over student rollouts; token POS / lexical class (content vs discourse marker); position in rollout; whether on a labeled pivot span; per-vocab-entry KL (for the clipping analysis).

## Method and model
**Setup.** Run on-policy distillation with several teachers, *no clipping*, for a fixed number of steps; log the per-token KL field; then add clipping and compare. Teachers:
1. **OPSD teacher** = same model conditioned on the ground-truth answer.
2. **SDFT teacher** = same model conditioned on an expert demonstration (possibly off-task).
3. **Real same-family teacher** = a moderately bigger checkpoint, no privileged info (≈ plain OPD).

**Modules.** Reuse `policy_gradients/loss.py::approx_kl`/`masked_mean`, `train.py::rollout()`/`compute_log_probs()`. New: teacher-logprob pass with optional privileged-info conditioning; per-token KL logger; per-vocab-entry point-wise KL clipper (the OPSD fix); a pivot-span labeler.

**Analyses.**
- KL distribution by token class (content vs discourse) and by teacher → is OPSD's "style tokens dominate" specific to the answer-conditioned teacher?
- Heavy-tail statistics (what fraction of total KL comes from the top 1% of tokens).
- Collapse timing without clipping (does it reproduce "~100 steps"?), and how clipping threshold trades collapse-avoidance vs learning speed.
- Correlation: KL magnitude vs causal importance to correctness (do high-KL tokens *deserve* the update?).

**Ablations.** Clip threshold sweep; per-token vs per-sequence vs per-batch KL normalization; fixing teacher to the initial policy vs the current student.

## Evaluation *(proposed — no results yet)*
| Teacher | Top-1% tokens' share of total KL | High-KL tokens that are discourse markers | Steps-to-collapse (no clip) | pass@1 w/ clip |
|---|---|---|---|---|
| OPSD (answer-conditioned) | **high** (expected) | **high** (expected) | ~100 (replicate) | ↑ |
| SDFT (demo-conditioned) | moderate (expected) | moderate | longer | ↑ |
| real bigger teacher (≈OPD) | **low** (expected) | low | none / very long | ↑, no clip needed |

- **Expected:** concentration ↑ with how aggressively the privileged info shifts the teacher; clipping is *necessary* for OPSD, *helpful* for SDFT, *unnecessary* for a well-matched real teacher → "diffuse teacher" is the goal, clipping is a fallback.
- **Where it breaks:** pivot-span labeling is noisy; "discourse marker" classification is fuzzy; collapse dynamics may be very seed-sensitive.

## Takeaways *(predictions)*
- Likely conclusion: the per-token KL field is heavy-tailed for self-with-hint teachers, dominated by discourse/style tokens that aren't causally important; clipping works but is treating a symptom — the real fix is a teacher whose KL is naturally diffuse and content-aligned.
- Risk: results may be model-family-specific (discourse-token behavior varies a lot).
- Open: can you *learn* a per-token weighting (importance ≈ causal effect on correctness) instead of clipping? That's a bridge to PRMs-as-teachers.
