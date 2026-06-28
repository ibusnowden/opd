# Distribution-Level Hint Rewriter: Big "Bad" Hint → Minimal Hint

_Status: idea · Started: 2026-05-11 · Last updated: 2026-05-11_
_Source: `research/ideas.md` — §8 "toward an optimal teacher", bullet 2 ("distribution-level prompt optimization")._

## Introduction
**Idea.** Per-task hint search (`per-task-hint-search-gepa.md`) doesn't amortize. So: train one model — a **hint rewriter** — that takes a "bad" privileged-info hint (the ground-truth answer, a full worked demonstration, whitebox reward-function access) and rewrites it into a *minimal* hint that, when handed to the same-family teacher in an OPD/OPSD step, moves the teacher's distribution as *little* as possible while still raising the student's reward. The training signal is the same Lagrangian (`E[Δreward] − β·KL`), aggregated across a *distribution* of tasks. Output: a reusable hint-rewriter artifact.

**Why it matters.** The post's §5 diagnosis is that OPSD fails because the answer-conditioned teacher is *too distributionally aggressive* (concentrated KL → collapse → needs clipping). A hint rewriter is the principled fix: keep the privileged info's *reward-relevant content* but strip the part that just yanks the teacher's distribution around. It turns "we have the answer but conditioning on it is dangerous" into "we have a distilled, surgical version of the answer". Generalizes SDFT's "use a demo" by *learning what makes a good demo/hint*.

**Prior work.** Shenfeld et al. 2026 (SDFT — fixed demo as hint; ICL assumption); Zhao et al. 2026 (OPSD — answer as hint, + clipping); Lu & Thinking Machines 2025 (OPD); Brown 2026 (Pareto framing); judge-model / reward-model training as a structural analogy (a small model that scores/conditions another). GEPA / prompt-optimization as the non-amortized baseline this replaces.

## Data
- A **task distribution**, not one task: many (problem-set + verifier) pairs across math (`reasoning_gym`), code editing (Minimal Code Editing corruptions), and a couple of knowledge/format families — so the rewriter learns a transferable skill, not one task's quirks.
- Per training example: `(problem, student rollout(s), bad-hint, candidate minimal-hint) → Lagrangian score`. The "bad hint" channel: ground-truth answer / full demo / whitebox RF signature.
- Held-out *task families* (not just held-out problems) for the transfer test.
- Logged: rewriter outputs, hint length, est. Δreward, est. teacher-KL-shift, downstream OPD pass@1, collapse/clip-needed flag.

## Method and model
**Two-stage.**
1. **Bootstrap targets** with the per-task search from `per-task-hint-search-gepa.md`: for a sample of (task, student-state), find good minimal hints (Pareto front at the target β). Use these as supervised targets *or* as a reward signal.
2. **Train the rewriter** — input: bad-hint (+ problem, maybe student rollout); output: minimal-hint. Either SFT on the bootstrapped good hints, or RL with reward = the Lagrangian (`Δreward × (1 − KL-shift)` or `Δreward − β·KL`), or SFT-then-RL. The student is held fixed during rewriter training (or periodically refreshed).

**Modules.** New: the rewriter model + its training loop (SFT and/or RL on the Lagrangian reward), a Lagrangian-reward computer (calls teacher + a few OPD micro-steps, or the no-train proxy from the GEPA proposal). Reuse: `policy_gradients/` for the RL arm of rewriter training (`loss.py`, `train.py::rollout/compute_advantages/apply_reward_kl`), `approx_kl` for the KL-shift term; `gpt_from_scratch/run.py` for a tiny-scale sanity check of the reward plumbing; W&B logging + SLURM driver à la `scripts/run_all_policy_gradients.sh`. The downstream OPD run uses the unified trainer from `meta-algorithm-alpha-lambda.md`.

**Ablations.** Rewriter trained per-domain vs across-domains; bad-hint channel = answer vs demo vs whitebox-RF; SFT-only vs RL-only vs both; β; with/without the student rollout as rewriter input; rewriter size.

## Evaluation *(proposed — no results yet)*
| Teacher conditioning | hint length | est. teacher-KL-shift | downstream pass@1 | clip needed? | forgetting Δ |
|---|---|---|---|---|---|
| answer (OPSD) | long | high | ↑ | yes | moderate |
| fixed demo (SDFT) | medium | medium | ↑ | sometimes | small–mod |
| rewriter(answer→minimal) | **short** | **low** (the point) | ↑ (expected ≥ SDFT) | no (expected) | small (expected) |
| rewriter on **held-out task family** | short | low (expected) | ↑ (expected — transfers) | no | small |

- **Headline metrics:** does the rewriter generalize to held-out task families? does it dominate "answer" and "fixed demo" on the Δreward–KL frontier? does downstream OPD with rewritten hints avoid the OPSD collapse without clipping?
- **Expected:** yes to all three, with the biggest open question being how far it transfers.
- **Where it breaks:** bootstrapped targets inherit the per-task search's noise; the Lagrangian reward is expensive (teacher calls + micro-steps); rewriter could learn to "leak" the answer in a compressed form (i.e. just re-encode it) rather than genuinely distill — needs a leakage check.

## Takeaways *(predictions)*
- Likely conclusion: a learned hint rewriter is the right amortization of "construct a locally-optimal teacher" — it makes OPSD-strength signal safe and reusable, and is a concrete step toward the post's "density of distillation + unbiasedness of RL + on-policy" wish.
- Risk: leakage; transfer may be domain-bounded; reward cost.
- Open: should the rewriter be conditioned on the *current student* (truly local) or be student-agnostic (more reusable, less optimal)? Where's the sweet spot — and is that just `co-evolving-hint-writer.md`?
