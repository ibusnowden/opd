# Train a Standalone Hint-Writer with RL on `correctness-Δ × (1 − KL-Δ)`

_Status: idea · Started: 2026-05-11 · Last updated: 2026-05-11_
_Source: `research/ideas.md` — §8 "toward an optimal teacher", bullet 4 ("train a hint-writing model directly with RL")._

## Introduction
**Idea.** Train a separate model whose job is to write hints, directly with RL, using the objective `correctness-delta × (1 − KL-delta)` — i.e. reward = (how much the hint raises the student's solve rate) × (how little the hint perturbs the teacher distribution). Scope it the way judge/reward models get scoped: per *model*, per *distribution*, per *task*, or "in general". The trained hint-writer is a shippable artifact you bolt onto any OPD/OPSD pipeline.

**Why it matters.** Same goal as `hint-rewriter-distillation.md` and `co-evolving-hint-writer.md` — construct a surgical, locally-good teacher instead of relying on the trivial "give it the answer" — but this variant (a) is *directly* optimized for the thing you want (no SFT-bootstrap stage, no co-evolution instability), and (b) borrows the *scoping discipline* of judge-model training, which makes "what does this hint-writer apply to" an explicit, controllable property. It's the most "ship it" of the §8 bullets.

**Prior work.** Lu & Thinking Machines 2025 (OPD); Zhao et al. 2026 (OPSD + per-token clipping — the failure this avoids); Shenfeld et al. 2026 (SDFT — fixed demo as hint); Brown 2026 (capability-vs-KL Pareto); Lightman et al. (PRMs) and judge/reward-model training as the scoping analogy; standard RLHF/RLVR machinery (Schulman et al. 2017; the `policy_gradients` algos).

## Data
- Training tasks for the hint-writer: a curated mix matched to the intended *scope* —
  - *task-scoped*: one task family (e.g. one math sub-skill or one corruption type);
  - *distribution-scoped*: a domain (all of math, all of code-editing);
  - *general*: the full battery.
  Sources: `reasoning_gym` (in `policy_gradients`), Minimal Code Editing corruptions, knowledge/format families. Each task carries a verifier (for `correctness-delta`).
- Privileged info the hint-writer may condition on: ground-truth answer / full demo / whitebox RF — declared per run.
- Held-out problems *and* held-out task families (to measure within-scope generalization vs out-of-scope behavior).
- Logged: hint text, length; `correctness-delta` (student solve rate with vs without hint, measured via a few OPD micro-steps or a no-train proxy); `KL-delta` (teacher distribution shift, via `approx_kl`); the product reward; downstream OPD pass@1 and forgetting; behavior when applied out of scope.

## Method and model
**Reward.** `R(hint) = Δacc(student | OPD-with-hint) × (1 − ΔKL(teacher | hint))`, both terms in [0,1] (clip/normalize). Optionally the additive Lagrangian form `Δacc − β·ΔKL` as an ablation. `Δacc` estimated by a small number of OPD steps with the hint on a dev pool (or the no-train proxy from `per-task-hint-search-gepa.md`); `ΔKL` from the per-token reverse-KL the teacher-with-hint induces vs teacher-without.

**Training loop.** Standard policy gradient on the hint-writer: sample hints → run the reward computation → update. Reuse `policy_gradients/`: `loss.py` (GRPO/RLOO/REINFORCE — group-relative is natural since we sample several hints per problem), `train.py::rollout/compute_advantages/apply_reward_kl`, `approx_kl`, `buffer.py`, `config.py` yaml. The student/teacher used inside the reward = the unified OPD trainer from `meta-algorithm-alpha-lambda.md`. New: the reward computer (the inner OPD-micro-step + KL-shift measurement), the scope-config plumbing, out-of-scope eval. W&B logging + SLURM driver mirroring `scripts/run_all_policy_gradients.sh` (`bigTiger`). Tiny-scale plumbing sanity on `gpt_from_scratch/run.py`.

**Scoping experiments.** Train task-scoped vs distribution-scoped vs general hint-writers; measure each on in-scope held-out, near-scope, and far-scope tasks; quantify the generality↔quality tradeoff (a general hint-writer should be weaker per-task but broadly usable — like a general judge).

**Ablations.** Reward form (product vs additive-Lagrangian); `Δacc` via micro-steps vs no-train proxy; # hints sampled per problem; conditioning channel (answer/demo/whitebox-RF); hint length cap; hint-writer size; with/without an explicit anti-leakage penalty (don't just re-encode the answer).

## Evaluation *(proposed — no results yet)*
| Hint-writer | scope | in-scope pass@1 (downstream OPD) | KL-shift | clip needed? | out-of-scope behavior | forgetting Δ |
|---|---|---|---|---|---|---|
| none (answer hint = OPSD) | — | ↑ | high | yes | n/a | moderate |
| task-scoped | one task | **↑↑ (expected best per-task)** | low | no (expected) | degrades / neutral | small |
| distribution-scoped | one domain | ↑ | low | no | ok within domain | small |
| general | all | ↑ (weaker per-task, expected) | low | no | broadly usable | small |
| ablate: product → additive-β | — | ≈ | depends on β | — | — | — |

- **Headline metrics:** does an RL-trained hint-writer dominate "answer hint" (OPSD) on the Δacc–KL frontier? does the scoping tradeoff behave like judge-model scoping (narrow = sharper, general = broader)? does it generalize within scope to held-out tasks?
- **Expected:** yes to the first; scoping behaves as analogized; within-scope generalization is decent, out-of-scope is poor (as intended/expected).
- **Where it breaks:** reward is expensive (inner OPD micro-steps) and non-stationary if the student moves (mitigate: hold student fixed, or refresh occasionally — and then it's basically `co-evolving-hint-writer.md`); leakage (hint-writer learns to smuggle the answer); the product reward can be gamed (drive `KL-delta`→0 with a vacuous hint that also has `correctness-delta`→0 — needs a floor on `Δacc`).

## Takeaways *(predictions)*
- Likely conclusion: a directly-RL-trained, scoped hint-writer is the most practical realization of "construct a locally-optimal teacher" — ship-able, composable with any OPD pipeline, and a concrete down payment on the post's "density + unbiasedness + on-policy" wishlist (the hint-writer adds back unbiased-ish, surgical signal to a dense distillation loss).
- Risk: reward cost; leakage; reward gaming; non-stationarity.
- Open: what's the right *scope granularity* in practice? Can one "general" hint-writer + a thin per-task adapter beat a fleet of task-scoped ones? How does this compose with `expert-rl-plus-opd.md` (hint-writer feeding the OPD term while an outcome reward runs alongside)?
