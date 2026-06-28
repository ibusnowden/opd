# Entropy Collapse: On-Policy Distillation vs RL

_Status: idea · Started: 2026-05-11 · Last updated: 2026-05-11_
_Source: `research/ideas.md` — §"A Note on Distributional Shaping" (entropy collapse is "significantly more drastic with OPD")._

## Introduction
**Question.** How much does policy entropy / output diversity collapse during on-policy distillation (reverse-KL, mode-seeking) compared to RL on the same task and budget — and does that collapse cause measurable downstream harm (pass@k, sample diversity, robustness)?

**Hypothesis.** OPD's reverse-KL objective is mode-seeking (Gu et al. 2023), so entropy collapses faster and further than under RL with a comparable KL budget. Some collapse is the *mechanism* of acquiring the new capability ("mode collapse around the new capability"), but past a point it costs diversity you needed for pass@k and exploration.

**Why it matters.** Expert→OPD-merge is becoming the final stage of post-training pipelines (GLM-5, DeepSeek-V4). If OPD systematically narrows the distribution, the merged model could be brittle in ways an RL'd model isn't — and you'd want entropy regularization / clipping as a default, not an afterthought. Also informs `per-token-kl-pivot-vs-style.md` (which tokens drive the collapse) and [[pass-at-k-vs-pass-at-1]] — the downstream-effect side: pass@k on a held-out set as the *lagging* confirmation that the collapse cost real coverage (Yue et al. 2025's "RL ≤ base at large k"; ProRL on whether the right recipe — KL control, ref-policy resets — buys it back), where the entropy curves here are the *leading* indicator.

**Prior work.** Lu & Thinking Machines 2025 (OPD); Gu et al. 2023 (reverse-KL mode collapse, GKD); Zhao et al. 2026 (OPSD per-token clipping prevents collapse within ~100 steps); Schulman et al. 2025 (O(1) bits / episode for outcome RL); Yue et al. 2025 / ProRL (pass@k vs pass@1 — see [[pass-at-k-vs-pass-at-1]]).

## Data
- **Math reasoning** (`reasoning_gym` tasks already in `policy_gradients`; AIME/GSM8K-style) — pass@1 and pass@k both meaningful.
- **Minimal Code Editing** env (from `opd-different-teachers.md`) for a second domain.
- Logged per-step: token-level entropy (mean, and distribution), sequence-level entropy estimate, unique-completion rate at temp=1, n-gram diversity, KL-to-base.

## Method and model
**Setup.** Same base, same task, matched rollout/compute budget; three runs: RL (GRPO), OPD (reverse-KL to a fixed same-family teacher), and SFT (cross-entropy, as a "maximally collapsing" reference). Track entropy trajectories side by side.

**Modules.** Reuse `policy_gradients/loss.py::approx_kl`/`masked_mean` and `train.py::rollout()`/`compute_log_probs()` for the instrumentation; RL run is the existing harness verbatim. New: OPD loss + teacher-logprob pass; an entropy/diversity logging callback (W&B metrics, mirroring the plotting in `gpt_from_scratch/run.py`).

**Manipulations.**
- Entropy bonus coefficient on the OPD run (0, small, larger) — does it recover diversity without killing the capability?
- Per-token KL clipping (à la OPSD) on/off — does it slow the collapse?
- Teacher temperature (sharper vs softer teacher targets).

**Ablations.** Student vs teacher size gap; train-set size; reverse-KL vs forward-KL distillation target (forward-KL = mode-covering — should collapse less).

## Evaluation *(proposed — no results yet)*
| Run | Entropy @ end (vs start) | Unique completions @ temp 1 | pass@1 | pass@8 | KL-to-base |
|---|---|---|---|---|---|
| RL (GRPO) | mild ↓ | moderate | ↑ | ↑ | small |
| OPD (reverse-KL) | **steep ↓** (expected) | **low** (expected) | ↑ | flat/↓ (expected) | moderate |
| OPD + entropy bonus | ↓ less | higher | ≈ | ≈/↑ | similar |
| SFT (ref) | steep ↓ | low | ↑ on-task | ↓ off-task | large |

- **Expected:** OPD entropy curve well below RL's at matched pass@1; pass@8 gap (RL − OPD) widens over training; entropy bonus / KL-clipping partially recover diversity at small capability cost.
- **Where it breaks:** if the task has little legitimate output diversity (then collapse is harmless and uninteresting); if matched-budget is hard to define fairly across RL and OPD.

## Takeaways *(predictions)*
- Likely conclusion: OPD trades diversity for sample-efficiency; some collapse is intrinsic to acquiring the capability, but a chunk is avoidable with cheap regularization → make entropy bonus / per-token clipping a default for OPD-merge stages.
- Risk: entropy is a coarse proxy; "diversity that matters" (correct alternative solutions) may not track raw entropy.
- Open: does the *merged* model (multiple OPD experts merged) collapse worse than any single expert? Is there a diversity-preserving merge?
