# Why Can an OPD Student Outperform Its Teacher?

_Status: idea · Started: 2026-05-11 · Last updated: 2026-05-11_
_Source: `research/ideas.md` — §"Why Can the Student Outperform the Teacher?" / §3 "on-policy state coverage"._

## Introduction
**Question.** On-policy distillation students sometimes beat their teacher's own sampled performance (Agarwal et al. 2023 on GSM8K; observed again in the post). Why? Two non-exclusive hypotheses:
- **H1 — On-policy state coverage.** The teacher gives advice *on the student's own prefixes*. Off-policy distillation supervises parts of the distribution the student rarely visits; OPD supervises exactly where the student goes wrong. (Ross et al. 2010 / DAGGER: exposure bias.)
- **H2 — KL-matching ≠ reward-maximization.** The teacher *distribution* (not its greedy decode) carries style, uncertainty, alternative continuations, reasoning structure. Matching it reshapes the student's sampling behavior in ways that improve outcomes without reproducing the teacher's greedy trajectory.

**Why it matters.** If H1 dominates, "student > teacher" is mostly an exposure-bias fix and the ceiling is still ~teacher. If H2 contributes, distributional shaping itself adds value and the framing of OPD as "RL with token-level teacher reward" is too narrow — which informs the meta-algorithm in `meta-algorithm-alpha-lambda.md`. Either way, because OPD is reverse-KL (mode-seeking), any "student > teacher" pass@1 win plausibly comes *with* a coverage cost — so whether it also beats the teacher (and the init) at pass@k is the load-bearing follow-up: see [[pass-at-k-vs-pass-at-1]].

**Prior work.** Agarwal et al. 2023; Lu & Thinking Machines 2025; Ross et al. 2010; Qwen3 report 2025; Gu et al. 2023 (reverse-KL mode-seeking / diversity); Yue et al. 2025 / ProRL (RL/distillation sharpen vs. expand — see [[pass-at-k-vs-pass-at-1]]).

## Data
- **Reasoning tasks with clean verifiers:** GSM8K (replicate the known effect first), then AIME-style / `reasoning_gym` math tasks already wired into `policy_gradients`.
- Optionally the Minimal Code Editing env from `opd-different-teachers.md` for cross-domain check.
- Stats to log: teacher greedy pass@1, teacher sample pass@1 (temp=1), student pass@1 before/after OPD, # student-visited states with teacher disagreement.

## Method and model
**Setup.** Fix teacher = a moderately-better same-family model. Distill via OPD onto a weaker student. Then run three diagnostic arms to separate H1 vs H2:
1. **On-policy vs off-policy (isolates H1).** OPD on student rollouts vs distillation on teacher rollouts, same teacher, same compute.
2. **Distribution vs argmax target (isolates H2).** Reverse-KL to the *full* teacher distribution vs SFT on the teacher's *greedy* token (a one-hot collapse of the same teacher). On-policy in both.
3. **Prefix-origin swap.** Train on teacher logprobs over (a) student prefixes, (b) teacher prefixes re-scored by teacher — same target form, different state distribution.

**Modules.** Reuse `policy_gradients/train.py::rollout()`, `compute_log_probs()`; new: teacher-logprob pass, reverse-KL loss, "argmax-teacher" SFT mode. W&B logging + SLURM as in `run_all_policy_gradients.sh`.

**Instrumentation.** Per-token: teacher–student KL, whether the token is on a "pivot" (high-KL) span; bucket gains by prefix novelty (how unlikely the prefix is under the teacher).

**Ablations.** Teacher temperature for trajectory generation; student vs teacher size gap; with/without per-token KL clipping.

## Evaluation *(proposed — no results yet)*
| Arm | Student pass@1 | Beats teacher (sample)? | Beats teacher (greedy)? | Notes |
|---|---|---|---|---|
| OPD (on-policy, full dist) | ? | expect sometimes ✓ | ? | the headline effect |
| Off-policy distill (teacher rollouts) | ? | expect ✗ / weaker | ? | tests H1 |
| On-policy, argmax-teacher SFT | ? | expect ≤ full-dist OPD | ? | tests H2 |
| Prefix-origin: teacher prefixes | ? | expect ✗ | ? | tests H1 |

- **Expected:** on-policy arms > off-policy arms (H1 real); full-distribution target ≥ argmax target (H2 contributes, smaller effect); gains concentrate on student-visited error states.
- **Where it breaks:** if teacher is much stronger (gap dominated by capability, not exposure); if verifier is noisy (GSM8K parsing); if student size is too small to represent the teacher's distribution.

## Takeaways *(predictions)*
- Likely conclusion: "student > teacher" is *mostly* on-policy state coverage, *partly* distributional shaping — neither alone is the full story.
- Risk: H1/H2 may not be cleanly separable; the argmax-teacher arm is itself an SFT run with its own quirks.
- Open: can you *amplify* H1 by deliberately oversampling student error states for teacher scoring? Does that beat plain OPD?
