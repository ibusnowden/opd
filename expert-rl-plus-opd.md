# Expert RL + OPD: A Teacher-KL Term On Top of Locally-Optimal RL

_Status: idea · Started: 2026-05-11 · Last updated: 2026-05-11_
_Source: `research/ideas.md` — §8 "toward an optimal teacher", bullet 5 ("borrow from 'expert RL + OPD'", DeepSeek-V4 style); §"The Full Pipeline" (MiMo-V2 Flash expert-then-merge)._

## Introduction
**Idea.** Some recent models (e.g. DeepSeek-V4) layer a *teacher distillation signal on top of* locally-optimal RL — i.e. the meta-algorithm with **both** a per-token teacher reverse-KL term **and** a sequence-level outcome reward live simultaneously (in the (α, λ, π_T) parametrization: α=1, 0<λ<1). The question: does combining a dense teacher signal with an unbiased outcome signal get you the best of both — the convergence speed of OPD plus the verifier-bounded ceiling of RL — and what does it do to the bias/concentration/stability picture from §5–§6?

**Why it matters.** The post argues OPD's ceiling is ~the teacher's, RL's ceiling is ~the verifier's; pure OPD is fast-but-capped, pure RL is slow-but-uncapped. "Expert RL + OPD" is the obvious hedge, and it's apparently already in production recipes — but the post treats it as a pointer ("the lessons from that line are directly relevant here") rather than something it analyzes. Worth making concrete: when does the teacher term *help* RL (early, dense guidance) vs *hurt* it (late, biasing the policy away from verifier-optimal)? Should λ anneal? Does the teacher term need its own clipping (the OPSD concentration problem) even when an outcome reward is also present? λ is exactly the sharpening↔coverage dial — and the clean diagnostic for "is a given λ buying coverage back" is pass@k vs. pass@1 (the ProRL "right recipe" question, parametrized): see [[pass-at-k-vs-pass-at-1]].

**Prior work.** DeepSeek-V4 / GLM-5 technical reports (expert RL + OPD, expert-then-merge final stage); MiMo-V2 Flash technical report (per-domain method choice: RL for math/code, distillation for creative/knowledge — referenced in the post); Lu & Thinking Machines 2025 (OPD); Zhao et al. 2026 (OPSD); Brown 2026 (capability-vs-KL Pareto); Schulman et al. 2017/2025 (PPO, credit assignment); Yue et al. 2025 / ProRL (sharpen vs. expand; the λ-blend as the recovery knob — see [[pass-at-k-vs-pass-at-1]]); Lightman et al. (PRMs — an alternative dense term, cf. `prms-as-teachers.md`).

## Data
- **Math & code** (where RLVR is strong): `reasoning_gym`/AIME-style (in `policy_gradients`), Minimal Code Editing — these are where you'd expect the outcome term to dominate eventually.
- **A noisier-reward domain** (creative writing / a knowledge benchmark with an LLM-judge reward): where you'd expect the teacher term to carry more weight (per the MiMo-V2-Flash observation).
- Held-out general benchmark for forgetting; pass@1 / pass@k for ceiling vs diversity.
- Logged: pass@1 over training for each λ schedule, KL-to-base, KL-to-teacher, entropy, per-token KL concentration, "who's driving the gradient" (teacher term vs outcome term magnitude over time).

## Method and model
**Loss.** The unified token-level PG from `meta-algorithm-alpha-lambda.md` at α=1 with 0<λ<1:
`Σ_t [ λ·(log π_T(ŷ_t|·) − log π_θ(ŷ_t|·)) + (1−λ)·Â^{outcome}_t ] · ∇_θ log π_θ(ŷ_t|·)`,
teacher π_T = a real same-family bigger checkpoint (or a learned hint-writer's teacher from `hint-writer-rl.md`).

**Modules.** Reuse: `policy_gradients/` for the outcome branch (`loss.py` advantage estimators, `train.py::rollout/compute_rewards/compute_advantages/apply_reward_kl`, `buffer.py`), `approx_kl` for both KL terms; the unified OPD trainer for the teacher branch. New: the λ-schedule controller, "gradient attribution" logging, optional per-token clip on the teacher branch only. W&B logging + SLURM driver à la `scripts/run_all_policy_gradients.sh` (`bigTiger`).

**Experiments.**
- **λ sweep, fixed:** {0 (pure RL), 0.25, 0.5, 0.75, 1 (pure OPD)} — capability vs convergence vs ceiling.
- **λ schedule:** anneal λ→0 (teacher as a warm-start curriculum, then let the verifier take over) vs anneal λ→up vs constant. Hypothesis: annealing-down gets OPD's speed early and RL's ceiling late.
- **Teacher quality:** weak / moderate / strong same-family teacher × λ — when does a weak teacher's bias cost more than its guidance is worth?
- **Domain split:** rerun the sweep on the math/code tasks vs the noisy-reward domain — does the best λ shift toward the teacher in the noisy domain (matching MiMo-V2-Flash's per-domain choice)?
- **Clipping:** does the teacher branch still need per-token KL clipping when an outcome reward is co-present (does the outcome term "dilute" the concentration problem)?

**Ablations.** Outcome estimator (GRPO/RLOO/GAE); KL-to-base penalty on/off; teacher = fixed checkpoint vs hint-writer-conditioned; sequence-level vs token-level mixing of the two signals.

## Evaluation *(proposed — no results yet)*
| Recipe | convergence speed | pass@1 (final) | ceiling (long run) | pass@k | KL-to-base | teacher-branch clip needed? | best in domain |
|---|---|---|---|---|---|---|---|
| pure RL (λ=0) | slow | high | **verifier-bounded (highest, expected)** | ↑ | small | n/a | math/code |
| pure OPD (λ=1) | **fast** | high-ish | **teacher-bounded (capped)** | flat/↓ | mod | maybe | noisy-reward |
| fixed λ=0.5 | medium-fast | high | between (expected) | ? | mod | ? | ? |
| anneal λ→0 | **fast early** | high | **→ verifier-bounded (expected best of both)** | ↑ late | mod→small | early-only? | math/code |
| weak teacher × λ=0.5 | fast early | ? | maybe *below* pure RL (bias cost) | ? | mod | ? | — |

- **Headline metrics:** does anneal-λ→0 dominate both pure RL and pure OPD (faster than RL to a comparable ceiling)? does the best λ shift with reward noisiness as the MiMo-V2-Flash data suggests? does co-present outcome reward remove the need for teacher-branch clipping?
- **Expected:** anneal-down is the winner in verifiable domains; teacher-heavy is better in noisy-reward domains; a *weak* teacher with high λ can underperform pure RL (bias > guidance) — so teacher quality and λ must be set together.
- **Where it breaks:** "convergence speed" and "ceiling" need careful matched-compute definitions; long-run RL ceilings are expensive to actually reach; the noisy-reward domain's LLM-judge is itself a biased signal (confounds the comparison).

## Takeaways *(predictions)*
- Likely conclusion: the production recipe (teacher KL + outcome reward, anneal the teacher down) is roughly right and *generalizes* — it's the meta-algorithm's α=1, λ-scheduled slice — and the practical knob is "set λ from how noisy your reward is, and how good your teacher is, then anneal".
- Risk: lots of expensive sweeps; matched-compute accounting; the noisy-domain confound.
- Open: should the teacher term be *replaced* mid-training by a PRM-style dense term (cf. `prms-as-teachers.md`) once the policy is good enough that the teacher is no longer ahead of it? What's the principled λ schedule (e.g. λ ∝ teacher-minus-student advantage)?
