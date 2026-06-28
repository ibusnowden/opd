# Per-Task Prompt/Hint Search Over the Lagrangian (GEPA-style Inner Loop)

_Status: **running as Exp 8** (stage 1 = `harness/hint_search.py`, jobs `115141` smoke → `115142` search; stage 2 config `harness/configs/exp8_fixed_hint_opd.yaml` ready) · Started: 2026-05-11 · Last updated: 2026-06-11_
_Source: `research/ideas.md` — §8 "toward an optimal teacher", bullet 1 ("per-task prompt optimization over the Lagrangian")._

## Introduction
**Idea.** The "optimal teacher" question becomes an optimization problem: find a teacher policy `π_T` that yields a large reward improvement on the *current* student per step, subject to a hard KL constraint that keeps updates stable — `max_{π_T} E[Δreward] − β·KL(π_θ ‖ π_T-induced-update)`. The cheapest version: don't train anything; *search the space of teacher prompts/hints* (a conditioning string `c_T` given to the same-family teacher) with a GEPA-style optimizer, using teacher sampling to estimate the objective on the current student and current task. Output: a per-task hint that makes the teacher a good OPD/OPSD teacher *right now*.

**Why it matters.** It's the lowest-effort point on the §8 ladder — a pure inner loop, no extra training, no new model artifact — and it directly tests the post's claim that you can construct a "locally optimal" teacher (high reward, low KL, surgical) instead of relying on a fixed external one. If it works even partially, the more expensive options (`hint-rewriter-distillation.md`, `co-evolving-hint-writer.md`, `hint-writer-rl.md`) are worth pursuing.

**Prior work.** GEPA (reflective prompt evolution / Pareto prompt optimization); Shenfeld et al. 2026 (SDFT — demonstration as the privileged-info hint; formalizes the ICL assumption); Zhao et al. 2026 (OPSD — the answer as the hint); Lu & Thinking Machines 2025 (OPD); Brown 2026 (capability-vs-KL Pareto framing). DSPy-style prompt optimizers as the tooling reference.

## Data
- A handful of held-out *tasks/task-families* (each: a set of problems + a verifier): math sub-skills from `reasoning_gym` (already in `policy_gradients`), Minimal Code Editing corruption types, a couple of knowledge/format tasks.
- Per task, a small dev pool (e.g. 32–128 problems) to estimate `E[Δreward]` and `KL` for a candidate hint; a disjoint test pool for the final OPD/OPSD run.
- Logged: candidate hint text, estimated Δreward, estimated KL-shift of the teacher (and of the resulting student update), Pareto front of hints, final downstream pass@1.

## Method and model
**Outer search (GEPA-style).** Maintain a population of candidate conditioning strings `c_T` (seeds: empty, "think step by step", a worked demo, the answer, partial answers, "be terse", domain hints). Reflectively mutate/crossover. Score each on the Lagrangian via teacher sampling on the dev pool against the current student. Keep the Pareto front (Δreward vs KL); pick by the target β.

**Inner objective estimate (no training in the loop).**
- Sample teacher completions / logprobs with `c_T` over student rollouts.
- `KL` term: per-token reverse-KL the OPD update *would* apply, averaged (a proxy for "how far this teacher pulls the student") — uses `policy_gradients/loss.py::approx_kl`.
- `Δreward` term: either (a) one or a few cheap OPD steps with that teacher and measure reward change on dev, or (b) a no-train proxy (does the teacher, conditioned on `c_T`, assign higher mass to high-reward continuations of student prefixes?).

**Downstream use.** Take the selected hint → run a real OPD/OPSD pass with that teacher (the unified trainer from `meta-algorithm-alpha-lambda.md`) → evaluate on the test pool + a forgetting benchmark.

**Modules.** New: the GEPA-ish hint optimizer, the Lagrangian estimator, the no-train Δreward proxy. Reuse: `policy_gradients/train.py::rollout()`/`compute_log_probs()`/`compute_rewards()`/`approx_kl`; W&B logging + SLURM driver pattern from `scripts/run_all_policy_gradients.sh`.

**Ablations.** β sweep (how surgical); proxy-Δreward vs few-step-Δreward; population size / # generations; hint length cap; teacher = self vs bigger same-family.

## Evaluation *(proposed — no results yet)*
| Teacher conditioning | est. Δreward | est. KL-shift | downstream pass@1 (test) | forgetting (held-out Δ) |
|---|---|---|---|---|
| none (≈ plain OPD) | ref | ref | ref | small |
| answer (≈ OPSD) | high | **high** | ↑ but needs clip | moderate |
| GEPA-found hint (β tuned) | high | **low** (the point) | ↑ (expected ≥ OPSD, ≤ ?) | small (expected) |
| random/ablated hint | ~0 | varies | ≈ none | — |

- **Expected:** the searched hint sits well inside the Δreward–KL Pareto front vs the "answer" hint at matched Δreward; gives OPSD-like gains without the concentration/collapse problem; the no-train proxy correlates well enough with few-step Δreward to be the cheap default.
- **Where it breaks:** the Lagrangian estimate is noisy on small dev pools; the no-train proxy may not track real Δreward; per-task search doesn't amortize (every new task pays the search cost — that's what `hint-rewriter-distillation.md` fixes).

## Takeaways *(predictions)*
- Likely conclusion: a cheap per-task hint search *does* produce a usable, more-surgical teacher than the trivial "give it the answer" hint — good enough to motivate amortizing the search into a hint-rewriter / hint-writer model.
- Risk: lots of moving parts (prompt optimizer + estimator + downstream run) for a per-task artifact that doesn't transfer.
- Open: what's the right cheap proxy for `E[Δreward]` that avoids any training in the inner loop? Can the Pareto front itself be reused across related tasks?
