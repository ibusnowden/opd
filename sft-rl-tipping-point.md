# The SFT → RL Tipping Point: When Does Teacher-SFT Stop Being Worth It?

_Status: idea · Started: 2026-05-11 · Last updated: 2026-05-11_
_Source: `research/ideas.md` — §1 "the standard pipeline and the compounding argument" (and the SFT-RS discussion)._

## Introduction
**Question.** The "SFT first, then RL" ordering is usually treated as convention, but the post gives it a real mechanism: which *sampling distribution* your method gets to compound with, and where the ceiling sits. Teacher-SFT has a sampling distribution fixed at dataset-construction time — once the student is close to the teacher, marginal examples just memorize, ceiling ≈ teacher. RL samples its own rollouts and improvements compound back into the sampling distribution — ceiling ≈ whatever the verifier can grade. So there's a *tipping point*: below it, SFT bits are extremely cheap-per-improvement (you're learning capabilities you don't have, from a source that does); above it, the student's own lucky-exploration rollouts produce strategies an RL gradient can extract, and rollout compute is better spent on RL. Can we *locate* that tipping point empirically, and confirm that **rejection-sampled SFT (SFT-RS / RFT)** only shifts the curve up without changing its shape (sampling distribution still pinned to whatever you filter)?

**Why it matters.** This is the load-bearing claim under every "SFT-then-RL" recipe and under the GLM-5 / DeepSeek-V4 "experts-then-merge" pipeline (which experts go through RL vs distillation, and for how long, is exactly a tipping-point question — see `expert-rl-plus-opd.md`). If the tipping point is predictable from cheap signals (student-vs-teacher gap, marginal-SFT-example informativeness, rollout pass-rate), you can *schedule* the SFT→RL handoff instead of guessing it. It also grounds the post's framing that all methods are points on a capability-vs-KL frontier (`meta-algorithm-alpha-lambda.md`): the tipping point is where moving along the SFT direction stops buying capability efficiently.

**Prior work.** The post itself (§1); Qwen3 technical report 2025 / Chu et al. 2025 (SFT needed after pretrain for format + instruction-following before RL is efficient); Lu & Thinking Machines 2025 (OPD); standard RFT / rejection-sampling-SFT (STaR-style) work; Brown 2026 (capability-vs-KL Pareto framing); Schulman et al. 2025 (outcome-RL credit assignment).

## Data
- **A task with a clean verifier and a real capability gap to a teacher:** math (`reasoning_gym` / AIME-style, already in `policy_gradients`) and Minimal Code Editing (from `opd-different-teachers.md`). Need a *teacher* meaningfully stronger than the student so "below the tipping point" is a real regime.
- **Teacher-SFT data:** teacher completions on the task (also a rejection-sampled / correctness-filtered version for the SFT-RS arm; and a student-rejection-sampled version).
- **Held-out test pool** per task + a general benchmark for forgetting.
- Logged at each rung: pass@1 / pass@k on test, KL-to-base, KL-to-teacher, entropy, "marginal-example informativeness" (loss the model still has on fresh teacher examples / how much a new SFT batch moves it), student rollout pass-rate, and — for each amount of rollout compute spent — the *marginal* improvement from spending it on more SFT-RS vs on RL.

## Method and model
**Design — a "rollout-compute allocation" sweep.** Fix a base, a teacher, a task, a total rollout-compute budget `C`. For a grid of split points `k`, spend the first `k·C` on teacher-SFT (or SFT-RS) and the remaining `(1−k)·C` on RL; also pure-SFT (`k=1`), pure-RL (`k=0`), and SFT→RL with the handoff *triggered by a signal* (e.g. switch when marginal-SFT-informativeness drops below a threshold, or when student rollout pass-rate crosses a level). Plot final performance vs `k`; the argmax is the empirical tipping point. Repeat with SFT-RS in place of vanilla SFT (expect the whole curve shifted up, same shape, same-ish argmax).

**Modules.** RL legs = `policy_gradients` harness as-is (`loss.py` GRPO/RLOO, `train.py::rollout/compute_rewards/compute_advantages/apply_reward_kl`, `buffer.py`, `config.py`). New: an SFT loop (and a rejection-sampling filter for SFT-RS), a "marginal-informativeness" probe, a triggerable SFT→RL scheduler, and the compute-accounting bookkeeping (rollouts are the common currency: SFT consumes teacher-generation rollouts, RL consumes student-generation rollouts). W&B logging + SLURM driver à la `scripts/run_all_policy_gradients.sh` (`bigTiger`). Tiny-scale plumbing sanity on `gpt_from_scratch/run.py`.

**Predictors of the tipping point.** Regress the empirical argmax-`k` against cheap early signals (initial student-vs-teacher gap, slope of SFT loss curve, student rollout pass-rate at small `k`) across tasks/teachers — can you *predict* where to hand off without running the full sweep?

**Ablations.** Teacher strength (weak / moderate / strong); SFT vs SFT-RS (teacher-filtered) vs SFT-RS (student-filtered); RL algo (GRPO vs RLOO); with/without KL-to-base on the RL leg; trigger signal for the scheduled handoff; task difficulty.

## Evaluation *(proposed — no results yet)*
| Allocation (k = fraction on SFT first) | final pass@1 | pass@k | KL-to-base | marginal gain: more SFT-RS vs RL (at the margin) | notes |
|---|---|---|---|---|---|
| k = 1 (pure SFT) | ≈ teacher (capped) | ↓ off-task | large | — | ceiling = teacher |
| k = 1, SFT-RS | a bit above pure SFT (expected) | ↓ less | large | — | curve shifted up, **same shape** (expected) |
| k = 0.75 | ↑ over pure SFT | ? | mod-large | RL ≳ more-SFT-RS past here (expected) | |
| **k\* (empirical argmax)** | **highest (expected)** | ? | mod | crossover point | the tipping point |
| k = 0 (pure RL) | high, slow to get there | ↑ | small | — | ceiling = verifier |
| scheduled handoff (trigger) | ≈ k\* (expected, no grid search) | ? | mod | — | the practical recipe |

- **Headline metrics:** location of the empirical tipping point `k\*`; whether SFT-RS shifts-but-doesn't-reshape the curve; whether a cheap signal predicts `k\*`; whether the *scheduled* handoff matches the best fixed `k` without a sweep.
- **Expected:** `k\*` strictly interior (some SFT then RL beats both extremes), `k\*` smaller (less SFT) when the student starts closer to the teacher and larger when it starts far; SFT-RS curve = SFT curve shifted up, ~same argmax; rollout-pass-rate / marginal-informativeness are decent predictors → scheduled handoff ≈ optimal.
- **Where it breaks:** "rollout compute" isn't a clean common currency (teacher generation ≠ student generation in cost); long-run RL ceilings are expensive to actually reach so "final" performance may be under-converged; with a *weak* teacher, "below the tipping point" barely exists and the curve is uninteresting; verifier noise (GSM8K-style parsing) confounds the RL legs.

## Takeaways *(predictions)*
- Likely conclusion: the SFT→RL ordering is justified *and quantifiable* — there's a real interior tipping point, predictable from cheap signals, and SFT-RS is a strict-improvement-but-not-a-fix (shifts the curve, doesn't change its shape). Practical upshot: schedule the handoff on a rollout-pass-rate / marginal-informativeness trigger rather than a fixed step count.
- Risk: compute-accounting fairness; under-converged RL ceilings; teacher-strength dependence.
- Open: where does *OPD* sit relative to this curve — does adding an OPD phase between SFT and RL move the tipping point (cheaper than RL, on-policy unlike SFT)? That's the bridge to `expert-rl-plus-opd.md` and `meta-algorithm-alpha-lambda.md`.
