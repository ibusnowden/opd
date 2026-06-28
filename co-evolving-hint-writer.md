# Co-Evolving Hint-Writer and Student (Self-Prompt-Optimization Online RL)

_Status: idea · Started: 2026-05-11 · Last updated: 2026-05-11_
_Source: `research/ideas.md` — §8 "toward an optimal teacher", bullet 3 ("self-prompt-optimization online RL")._

## Introduction
**Idea.** A static hint rewriter (`hint-rewriter-distillation.md`) is trained against a frozen student; but the "best teacher" changes as the student improves. So make the hint-writer a *co-evolving* agent: treat hints as rollouts in a parallel environment that improves alongside the main policy, with the two loss components — reward-delta on the student and KL-shift of the teacher — balanced *adaptively* (e.g. a smoothed minmax / Lagrangian-with-learned-multiplier over the two terms). Hint-writer and student train together.

**Why it matters.** This is the "locally optimal teacher, *tracked over time*" version of §8. The post's whole thesis is that on-policy data is load-bearing and the open problem is dense-but-unbiased credit assignment; a co-evolving hint-writer is a candidate mechanism — at every step the teacher is re-tuned to be surgical *for the current student*, so the per-token signal stays informative and the KL budget stays controlled as the student moves. It's the closest of the §8 bullets to an actual "better-than-RL" training loop.

**Prior work.** Lu & Thinking Machines 2025 (OPD); Zhao et al. 2026 (OPSD); Shenfeld et al. 2026 (SDFT); Brown 2026 (capability-vs-KL Pareto); self-play / co-training literature (e.g. generator–verifier, asymmetric self-play); automatic-curriculum / teacher-student RL. Adaptive Lagrangian-multiplier control as in constrained RL (PPO-Lagrangian, RCPO).

## Data
- An online task stream: math (`reasoning_gym` / AIME-style, already in `policy_gradients`) as the primary, Minimal Code Editing as a secondary; a held-out general benchmark for forgetting.
- "Bad hint" sources available to the hint-writer: ground-truth answer, full demo, whitebox RF — same as the rewriter proposal.
- Logged per step: student pass@1 / pass@k, KL-to-base, entropy; hint-writer reward, hint length, teacher-KL-shift, the adaptive multiplier(s); collapse/instability flags; the joint learning curve.

## Method and model
**Two coupled learners, one outer loop.**
1. **Student** updates via OPD/OPSD using the *current* hint-writer's hint to condition the teacher (the unified trainer from `meta-algorithm-alpha-lambda.md`).
2. **Hint-writer** updates via RL with reward `f(Δreward_student, KL_shift_teacher)` where `f` is an adaptively-weighted combination — start from `Δreward − β·KL`, let `β` (or a softmin/softmax temperature) be driven by a controller that targets a desired KL band (smoothed minmax: maximize worst-case-over-recent-batches reward-delta s.t. KL ≤ budget).
3. **Schedule.** Interleave updates (alternating, or k:1); periodically re-estimate the hint-writer's reward on fresh student states; optional EMA / trust region on both to keep the co-evolution from oscillating.

**Modules.** Reuse: `policy_gradients/` for *both* RL loops (`loss.py` algos, `train.py::rollout/compute_advantages/apply_reward_kl`, `approx_kl`, `buffer.py`), the unified OPD trainer from `meta-algorithm-alpha-lambda.md` for the student arm. New: the hint-writer policy + its reward computer; the adaptive-multiplier controller; the interleaving/EMA scheduler; co-evolution diagnostics. Tiny-scale sanity (does the coupled loop even stay stable?) on `gpt_from_scratch/run.py`. W&B logging + SLURM driver à la `scripts/run_all_policy_gradients.sh` (`bigTiger`).

**Ablations.** Co-evolving hint-writer vs frozen rewriter vs per-task search vs plain OPD vs plain RL (all matched compute); update ratio (k:1); fixed-β vs adaptive-β controller; with/without trust region; hint-writer = self-LoRA vs separate small model; refresh frequency of the hint-writer's reward estimate.

## Evaluation *(proposed — no results yet)*
| Setup | student pass@1 | KL-to-base (stays in band?) | pass@k | stability (oscillation?) | forgetting Δ | compute vs RL |
|---|---|---|---|---|---|---|
| plain RL (GRPO) | ref | small | ↑ | stable | small | 1× (baseline) |
| plain OPD (real teacher) | ↑ faster | mod | flat/↓ | stable | small | <1× |
| frozen rewriter | ↑ | mod, fixed-β drift | ? | stable | small | <1× |
| **co-evolving hint-writer** | **↑ (expected ≥ all above at matched compute)** | **in band (controller)** | ? (entropy bonus may help) | risk: oscillation | small (expected) | target <1× of RL |
| ablate adaptive-β → fixed | ↑ | drifts out of band | ? | less stable | varies | — |

- **Headline metrics:** does co-evolution beat the frozen-rewriter and plain-OPD baselines at matched compute? does the adaptive controller actually keep KL in band without manual tuning? does it stay *stable* (the main risk)?
- **Expected:** modest win over frozen rewriter; the controller earns its keep; stability requires the trust-region / EMA pieces.
- **Where it breaks:** classic two-learner instability (chasing / collapse); the hint-writer's reward is expensive *and* non-stationary; "matched compute" accounting is tricky with two models.

## Takeaways *(predictions)*
- Likely conclusion: co-evolution is the most promising §8 bullet *in principle* but the most engineering-heavy and the most prone to instability; worth doing only after the frozen-rewriter result is positive.
- Risk: a lot of complexity; could spend all the time fighting oscillation.
- Open: is there a *non-adversarial* formulation (hint-writer and student optimizing a shared objective, not a minmax) that's stabler — and does it still track the moving optimum?
