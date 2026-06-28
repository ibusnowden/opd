# The (α, λ, π_T) Meta-Algorithm: Mapping the Corners and Probing the Interior

_Status: idea · Started: 2026-05-11 · Last updated: 2026-05-11_
_Source: `research/ideas.md` — §6 "the sparse/dense × biased/unbiased family", §7 "an illustrative meta-algorithm", §8._

## Introduction
**Idea.** All of SFT, RL, OPD, OPSD are special cases of one token-level policy gradient with two scalar knobs and a teacher choice:
- **α ∈ [0,1]** — how on-policy the sampling distribution is (1 = sample from current student; 0 = fixed dataset / off-policy).
- **λ ∈ [0,1]** — how much of the per-token advantage comes from a teacher reverse-KL term vs a sequence-level outcome reward.
- **π_T(·| c_T)** — which teacher model, conditioned on what context (nothing / a demo / the answer / a learned hint).

Corners: SFT ≈ (α=0, λ=1, π_T = δ_{y_data}); RL ≈ (α=1, λ=0); OPD ≈ (α=1, λ=1, real same-family teacher); OPSD ≈ (α=1, λ=1, self + answer-hint). The post's own caveat: the *interior* (α, λ ∈ (0,1)) is statistically messy (needs importance-sampling corrections) and "I'm not sure that's even a useful algorithm" — the cleaner axis of variation is the effective KL budget β, and the hard sub-problem is *teacher optimization* (see `per-task-hint-search-gepa.md`, `hint-writer-rl.md`).

**Why it matters.** Even if the interior turns out not to be useful, *establishing that* — by actually building the unified update and probing it — is valuable: it tells you whether the clean corners are clean for a fundamental reason (no IS correction, qualitatively distinct β regimes) or just by convention. And the unified harness is the substrate for everything in §8.

**Prior work.** Brown 2026 (the framing this post responds to: post-training methods as capability-vs-KL tradeoffs on a Pareto frontier); Lu & Thinking Machines 2025 (OPD); Zhao et al. 2026 (OPSD); Shenfeld et al. 2026 (SDFT); Schulman et al. 2017 (PPO clipping), Schulman et al. 2025; Lai et al. 2025 (data-dependent regularization); Lu et al. 2025 (on-policy data).

## Data
- A small battery: math (`reasoning_gym`/AIME-style, already in `policy_gradients`), the Minimal Code Editing env, and a held-out general eval for forgetting — same battery used across the other proposals so results compose.
- Logged: pass@1 / pass@k, KL-to-base over training, entropy, per-token KL field, the "concentration" statistics from `sparse-vs-dense-updates.md`.

## Method and model
**Core artifact — a unified token-level PG trainer.** One loss of the form
`Σ_t [ λ·(log π_T(ŷ_t|ŷ_<t) − log π_θ(ŷ_t|ŷ_<t)) + (1−λ)·Â^{outcome}_t ] · ∇_θ log π_θ(ŷ_t|ŷ_<t)`,
sampling with mix parameter α (importance-weight + clip the off-policy fraction, PPO-style), pluggable π_T.

**Modules.**
1. **Sampler** with α-mix (student rollouts + a fixed buffer of off-policy/teacher data; IS weights with clipping). Build on `policy_gradients/train.py::rollout()` + `buffer.py::ReplayBuffer`/`Experience`.
2. **Outcome-reward branch** (λ<1): reuse `compute_rewards()`, `compute_advantages()`/`compute_gae()`/`compute_loo_advantages()`, `apply_reward_kl()`.
3. **Teacher-KL branch** (λ>0): teacher-logprob pass over student tokens, reverse-KL advantage, optional per-token point-wise clipping (OPSD-style) — *new code*.
4. **Teacher registry**: `none` / bigger-same-family / self+demo / self+answer / learned-hint (the latter ties to `hint-writer-rl.md`).
5. **Logging**: W&B metrics + plots à la `gpt_from_scratch/run.py`; SLURM `bigTiger` driver mirroring `scripts/run_all_policy_gradients.sh`.

**Experiments.**
- **Reproduce the corners** exactly (sanity: (α=1,λ=0) should match the existing GRPO run; (α=0,λ=1,δ) should match a vanilla SFT loop).
- **β sweep along each edge** (vary KL-to-base penalty / teacher temperature): trace capability-vs-KL Pareto points; check whether the corners sit on a smooth frontier.
- **Interior probes**: a few (α,λ) ∈ (0,1) settings with IS correction — does anything beat the nearest corner at matched β? Document failure modes (variance blow-up, IS clipping bias).
- **Teacher swap at fixed (α=1,λ=1)**: real teacher vs self+demo vs self+answer — recovers the OPD/SDFT/OPSD comparison within one codebase.

**Ablations.** IS-clip threshold; per-token KL clip on/off; reverse-KL vs forward-KL teacher term; outcome advantage estimator (GRPO vs RLOO vs GAE).

## Evaluation *(proposed — no results yet)*
| Setting | (α, λ, π_T) | pass@1 | KL-to-base | Variance / stability | On the Pareto frontier? |
|---|---|---|---|---|---|
| GRPO (corner) | (1, 0, —) | ref | ref | stable | yes (RL point) |
| SFT (corner) | (0, 1, δ_data) | ref | large | stable | yes (SFT point) |
| OPD (corner) | (1, 1, real teacher) | ref | mod | stable | yes (OPD point) |
| OPSD (corner) | (1, 1, self+answer) | ref | mod | needs clip | near-OPD |
| interior A | (0.5, 1, real teacher) | ? | ? | ? (expect worse) | ? |
| interior B | (1, 0.5, real teacher) | ? | ? | ? | ? |

- **Expected:** corners reproduce; β sweeps trace a coherent capability/KL frontier; interior points mostly *don't* dominate the nearest corner once IS correction is paid for — confirming the post's hunch that the useful axis is β + teacher choice, not (α,λ) interpolation.
- **Where it breaks:** IS corrections at small α make gradients high-variance; "matched β" across very different update shapes is itself a modeling choice; small models may not show a clean frontier.

## Takeaways *(predictions)*
- Likely conclusion: the meta-algorithm is most valuable as *a single codebase that contains all the corners* (so teacher-optimization experiments are apples-to-apples), and as a way to *empirically rule the interior in or out* — not as a new SOTA recipe.
- Risk: spending a lot of engineering to confirm a negative; mitigate by treating the unified trainer as shared infra for the other §8 proposals.
- Open: is there a *reparametrization* of the interior (different from (α,λ) mixing) where the statistics stay clean? That's the actual "better-than-RL algorithm" question.
