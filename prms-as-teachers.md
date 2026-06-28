# PRMs as Teachers: Blending Per-Rollout Self-Distillation Guidance With Outcome Rewards

_Status: idea · Started: 2026-05-11 · Last updated: 2026-05-11_
_Source: `research/ideas.md` — closing notes (lines ~85, ~89): "maybe we can just think of PRMs as teachers?"; "better ways of blending per-rollout SD guidance with outcome rewards"._

## Introduction
**Idea.** RL's sparsity comes from outcome-only rewards spread over many noisy tokens; PRMs (process reward models) promise dense per-step signal but "don't train efficiently at scale" (Lightman-style PRMs). OPSD already does a *form* of process supervision — it isolates pivot tokens via per-token teacher KL — so the question is whether you can reframe a PRM **as a teacher** in the (α, λ, π_T) sense, and *blend* per-rollout self-distillation guidance with outcome rewards so you get the density of process supervision, the unbiasedness of outcome RL, and the on-policy property of both. The post's own framing of the open problem: "you want something with the density of distillation, the unbiasedness of RL, and the on-policy property of both."

**Why it matters.** This is the most direct stab at the post's stated open problem (the dense-but-unbiased credit-assignment knob). It connects three threads the post leaves dangling: (1) OPSD ≈ self-reflective process supervision *already*, but pivot tokens are sparse and the teacher-ideal updates aren't on-policy; (2) PRMs are dense but biased and train poorly; (3) "expert RL + OPD" (`expert-rl-plus-opd.md`) shows production recipes already layer a dense teacher term on outcome RL. Treating "PRM" and "teacher" as the same kind of object lets you reuse the (α, λ) machinery to interpolate between them and study the bias/density/on-policy tradeoff explicitly. The success test for "PRM-reweighting fixed OPSD's concentration without blunt clipping" is, downstream, **pass@k**: did the recovered diversity actually buy back coverage vs. the un-RL'd init, or just shuffle which trick the policy collapses onto? — see [[pass-at-k-vs-pass-at-1]].

**Prior work.** Lightman et al. (PRMs / "Let's Verify Step by Step"); Schulman et al. 2025 (O(1) bits per episode for outcome RL — the sparsity argument); Zhao et al. 2026 (OPSD — per-token teacher KL ≈ self-reflective process signal, + clipping); Lu & Thinking Machines 2025 (OPD); DeepSeek-V4 / GLM-5 reports (dense teacher term + outcome RL); Brown 2026 (capability-vs-KL Pareto); Yue et al. 2025 / ProRL (sharpen vs. expand — the downstream pass@k test; see [[pass-at-k-vs-pass-at-1]]); GRPO/PPO/RLOO (`policy_gradients`) as the outcome-RL substrate.

## Data
- **Long-CoT math** (`reasoning_gym`/AIME-style, in `policy_gradients`) — the canonical PRM domain; long rollouts where step-level credit matters.
- Optionally a step-labeled subset (PRM800K-style human/model step labels, or proxy labels: a step is "good" if intervening there most improves final correctness) to *train/calibrate* the PRM-as-teacher and to evaluate per-token credit quality.
- Minimal Code Editing as a secondary domain (steps = edit decisions).
- Held-out general benchmark for forgetting; pass@1 / pass@k.
- Logged: per-token credit from each source (PRM-teacher KL term vs broadcast outcome advantage vs self-distillation KL), how concentrated/heavy-tailed each is, correlation of each with the proxy causal-importance labels; downstream pass@1, entropy, KL-to-base.

## Method and model
**Reframe.** A "PRM teacher" π_T is any object that produces a per-token (or per-step) preference signal you can write as `log π_T(ŷ_t|ŷ_<t) − log π_θ(ŷ_t|ŷ_<t)` — concretely: (a) a same-family model conditioned on partial-progress info (an OPSD-ish self-teacher), or (b) a small PRM that scores prefixes and is converted to a token-level advantage, or (c) the post's note: *use the PRM's step scores to reweight the OPSD per-token KL toward causally-important tokens* (so high-KL-but-stylistic tokens get down-weighted, high-KL-and-important tokens get up-weighted — addresses the concentration problem from `per-token-kl-pivot-vs-style.md`).

**Blend.** Use the unified token-level PG from `meta-algorithm-alpha-lambda.md` at α=1 with a three-way per-token advantage: `λ_PRM · Â^{PRM-teacher}_t + λ_SD · Â^{self-distill}_t + (1−λ_PRM−λ_SD) · Â^{outcome}_t`. Sweep / schedule the λ's.

**Modules.** Reuse: `policy_gradients/` outcome branch (`loss.py`, `train.py::rollout/compute_rewards/compute_advantages/apply_reward_kl`, `approx_kl`, `buffer.py`), the unified OPD trainer for the teacher/SD branches. New: a small PRM (or a prefix-conditioned self-teacher) + its training/calibration, a PRM-score → token-advantage converter, the per-token reweighting of the SD-KL by PRM importance, the three-way λ scheduler, per-token-credit diagnostics. Tiny-scale plumbing sanity on `gpt_from_scratch/run.py`. W&B logging + SLURM driver à la `scripts/run_all_policy_gradients.sh` (`bigTiger`).

**Experiments.**
- **PRM-as-teacher vs PRM-as-reward:** does converting PRM scores into a *teacher KL term* (on-policy, reverse-KL) behave better than adding them as a *dense reward* (the classic, badly-scaling way)?
- **PRM-reweighted OPSD:** does using PRM importance to reweight the per-token self-distillation KL remove the need for blunt per-token clipping while keeping the gains? (direct test of the line-89 conjecture)
- **Three-way λ sweep / schedule:** find where on the density↔bias↔on-policy simplex you get RL's ceiling at OPD-ish speed.
- **Credit-quality probe:** correlate each source's per-token signal with proxy causal-importance labels — which source assigns credit best, and does blending beat each alone?

**Ablations.** PRM = trained model vs prefix-conditioned self-teacher vs answer-conditioned self-teacher; PRM size / amount of step-label data; reweighting function (linear / softmax-over-importance / hard top-k); λ schedules; with/without per-token clipping on top of reweighting.

## Evaluation *(proposed — no results yet)*
| Recipe | per-token credit quality (corr. w/ importance) | concentration / clip needed? | pass@1 (final) | ceiling (long run) | pass@k | convergence speed | forgetting Δ |
|---|---|---|---|---|---|---|---|
| outcome-only RL (GRPO) | low (broadcast) | n/a | high | **verifier-bounded** | ↑ | slow | small |
| OPSD (answer self-teacher) | medium (pivot-ish, + style noise) | concentrated → **clip needed** | high-ish | teacher-bounded | flat/↓ | fast | moderate |
| PRM-as-dense-reward (classic) | medium (biased) | n/a | ? | biased ceiling | ? | medium | ? |
| PRM-as-teacher (KL term) | medium-high (expected) | less concentrated (expected) | high | between (expected) | ? | fast-ish | small (expected) |
| PRM-reweighted OPSD | **high (expected — the conjecture)** | **diffuse, no clip (expected)** | high | teacher/PRM-bounded | better than OPSD (expected) | fast | small |
| blended (3-way λ, scheduled) | high | manageable | **high (target: ≥ RL, faster)** | **→ verifier-bounded (target)** | ↑ | **fast (target)** | small |

- **Headline metrics:** does PRM-as-teacher beat PRM-as-reward? does PRM-reweighting make OPSD safe without clipping (line-89 conjecture)? does the blended recipe get RL's ceiling at meaningfully less compute? which source assigns per-token credit best?
- **Expected:** PRM-as-teacher ≥ PRM-as-reward; reweighting works (the conjecture holds, at least partially); blending beats each ingredient alone on the speed/ceiling tradeoff; outcome term still needed for the unbiased tail (hardest problems).
- **Where it breaks:** PRMs are exactly the thing that "don't train efficiently at scale" — a weak PRM injects bias and the whole thing degrades to a worse OPSD; proxy causal-importance labels are noisy; the three-way λ space is large and expensive to sweep; this is engineering-heavy (PRM + converter + reweighting + 3-way scheduler on top of the unified trainer).

## Takeaways *(predictions)*
- Likely conclusion: "PRM ≈ teacher" is a useful unification — it lets you fix OPSD's concentration problem with *learned* importance instead of blunt clipping, and it slots cleanly into the (α, λ) framework as another teacher choice; the blended recipe is a plausible (partial) answer to the post's open problem, with outcome RL still doing the unbiased heavy-tail work.
- Risk: PRM training cost/instability is the load-bearing weakness — if PRMs don't scale, this doesn't either; high engineering surface.
- Open: can the PRM itself be the *same model* (a prefix-conditioned self-teacher) so there's no separate-PRM-training problem — i.e. is "PRM-reweighted OPSD" just "OPSD with a smarter per-token weight, computed self-referentially"? And if so, where does that self-referential weight come from without circularity?
