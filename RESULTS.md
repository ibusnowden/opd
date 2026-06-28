# On-Policy Distillation Through a Distributional Lens — Results

_Status: in progress · Started: 2026-05-12 · Last updated: 2026-06-06_

> A **results writeup**, blog-post style (cf. [Thinking Machines, "On-Policy Distillation"](https://thinkingmachines.ai/blog/on-policy-distillation/)) — narrative + measured findings + plots. Distinct from the per-proposal `.md`s in this folder, which are *proposals* (status `idea`, with *expected* outcomes). Sections **§4 (Exp 1 + D1–D4)**, **§5 (Exp 2)**, **§6 (Exp 3)**, **§7 (Exp 4 + follow-ups)**, **§7.10 (Exp 5)**, **§7.11 (Exp 6)**, **§7.12 (Exp 7)**, **§7.13 (Exp 8)**, **§8 (Discussion)**, and **§9 (Conclusion)** are real measured results / synthesis, and **§1 (framing / intro)** is the narrative wrapper. The writeup is complete.
>
> Code: everything lives under `research/` — `harness/` (the unified (α, λ, π_T) trainer + `eval_passk.py`) + the vendored `policy_gradients/`. Logging: W&B project `distill-harness` (run offline on the cluster, `wandb sync research/wandb/offline-run-*` afterwards). Hardware: 8×H100 80 GB (`itiger01`). Source/inspiration: Brown & Claude, *"SFT, RL, and On-Policy Distillation Through a Distributional Lens"* (Apr 30 2026); the literature is woven in per-section rather than in a dedicated related-work block (see `roadmap.md` for the full lit assessment, and the per-proposal `.md`s).

---

## Status / progress tracker

| Experiment | Proposal | Status | Runs / artifacts |
|---|---|---|---|
| **Exp 1 — does the teacher's recipe matter?** | [[opd-different-teachers]] (#1) | **done + diagnosed + off-policy-SFT arm (§8.1)** (gsm_symbolic, 500 steps; D1–D4 landed) | SLURM `71148` ✓ (7h28m); D1 smoke `71162` ✓; D2 offline `harness/diagnose_per_token_kl.py`; D3 teacher-eval `71161` ✓; D4 wider pass@k `71158` ✓ — `results/{exp1_opd_teachers,exp1_passk_71158,teacher_eval_71161,diag_per_token_kl}/`. **Off-policy-SFT arm `80388`(train)+`80389`(eval) ✓ (2026-06-04, §8.1)**: off-policy SFT p@1 **0.388** (filtered) / **0.297** (unfiltered) vs on-policy pure-OPD ← Instruct **0.005** — ~55–72×; correctness filter adds only +0.09, so **on-policy state coverage is NOT load-bearing at the 1B/7B gap** (post's central claim reversed). 4th teacher *interface* still deferred. |
| **Exp 2 — where does the per-token teacher signal go?** | [[per-token-kl-pivot-vs-style]] (#4) | **done (offline taxonomy, 2026-05-25)** | offline `harness/diagnose_per_token_kl.py` (extended with `categorize_token` + `taxonomy_summary`) + launcher `run_exp2_taxonomy.sh`; SLURM `75121` ✓ (5 arms × 64 prompts × 4 samples). **§5 reading**: pure OPD λ=1 puts **67% of |KL| mass on uncertain (high-entropy) tokens**, only 17% on content; GRPO inverts this (**60% content, 13% uncertain**) without ever seeing a teacher term; clipped λ=0.10 winner sits in between (45% uncertain, 32% content). **Per-token clip selectively disarms the uncertain bucket**: 54-79% of clip-removed mass is uncertain across α=1 arms, only 15-46% from content. SFT control is uniquely format-heavy (38% format mass, 60% of its clip-removed mass is format) — matches §6.5's third regime. Closes the §7.7 mechanism story at the bucket level: clipping is a *structural reweighting* that lifts content's relative weight by killing uncertain outliers. |
| **Exp 3 — what does OPD do to the weights?** | [[sparse-vs-dense-updates]] (#5) | **static + dynamic + spectral + SFT control done (2026-05-22/25)** | `harness/delta_theta_snapshot.py` + `prune_dtheta_eval.py` + `effrank_all_tensors.py` + `plot_*.py`. **§6 static** (sparsity, 11 ckpts): all α=1 arms RL-sparse (top-1% mass 0.57-0.63); "SFT-dense for OPD" prediction falsified. **§6.1 + §6.2 dynamic** (prune curves, 4 ckpts × 10 prune levels = 40 evals via SLURM 72097 + 72122): bottom 50% of moves is dead weight; broader-tier arms have a sharp cliff in p∈[80%, 90%]; GRPO degrades **gradually** (retains 50% of p@1 at p=95%); pure OPD *improves monotonically* with pruning (5× unpruned p@1 at p=95%). **§6.3 spectral (top-10 slice, partially corrected by §6.4)** + **§6.4 spectral (full pass, 11 ckpts × all 114 2D weights, ~2h CPU)**: tier separation is small & submodule-specific — sharper tier adds ~6% rank at attn_qkv and *compresses* embed by ~3%; broader tier preserves rank everywhere; MLP is unchanged across both tiers. **§6.5 SFT-from-rollouts control (jobs 74735 + 74736, 2026-05-25)**: closes the (α=0, λ=1, π_T=δ_data) corner — SFT is the **sparsest** arm (top-1% mass 0.729, only 1.6% changed>1e-4, p99 |Δθ| 3-13× smaller than any α=1 arm); spectral signature is a *more extreme* sharper tier (attn_qkv rank ratio 1.075 above pure OPD's 1.069; embed rank ratio **0.846** vs sharper-tier mean 0.970, 5× larger compression); prune curve is gradual with no cliff and p@16 *peaks* at p=70%. Three-regime picture (SFT > sharper > broader) replaces the prior two-tier story; "RL sparse / SFT dense" literature prediction is reversed at this training intensity. |
| **Exp 4 — the (α, λ) interior** | [[meta-algorithm-alpha-lambda]] (#6), [[expert-rl-plus-opd]] (#11) | **done in-dist + cross-task + gen-gap — closed** | SLURM `71174` v1 cancelled; `71177/71188` v2 seeds 42/43 ✓; **§7.2 positive control**: `71200/71201/71202/71208`; **§7.4 pass@k + cross-task** `71210/71211` ✓; **§7.5 GRPO multi-seed** `71242` ✓; **§7.6 collapse-recovery** `71249` ✓; **§7.7 clip sweep** `71250` ✓; **§7.7 clip-λ sweep** `71271` ✓; **§7.7 low-λ replication** `71395` ✓; **§7.8 clipped cross-task eval** `71574` ✓ (2026-05-21); **§7.9 gen-gap eval-seed robustness** `72040` ✓ (2026-05-22). **Reading:** in-dist `gsm_symbolic` clipped λ=0.10 narrowly beats GRPO-v2 (0.693 vs 0.687 p@1, 0.801 vs 0.786 p@16); off-task `simple_equations` GRPO wins p@1–p@16 by ~2× but clipped λ=0.10 crosses over and beats GRPO at p@32–p@64 at T=0.6; eval-seed variance is tiny (sd 0.006-0.014); v2.1 bimodality is a *training-seed* effect not an eval-seed artifact, and clip1 dominates both unclipped seeds (0.685 vs 0.435 / 0.085). λ≥0.50 stays dead. |
| **Cross-cutting eval — pass@k vs pass@1 / entropy collapse** | [[pass-at-k-vs-pass-at-1]] (#15), [[entropy-collapse-opd-vs-rl]] (#3) | **done — crossover landed** | `harness/eval_passk.py`; D4 ran on all 9 Exp-1 ckpts (k up to 64, T∈{0.6, 1.0}); Exp-4 v2.1 pass@k and `simple_equations` cross-task evals ran as `71210/71211`; clipped-band cross-task eval `71574` ✓ — **clean pass@k crossover at k≥32, T=0.6**: clip λ=0.10 beats GRPO at p@32/p@64 on `simple_equations`. |
| **Exp 5 — PRMs as teachers (answer-conditioned OPSD)** | [[prms-as-teachers]] (#13) | **pilot + taxonomy + multi-seed done (2026-05-26)** | `harness/teachers.py` `PrivilegedInfoTeacher` implemented (batched chat-template hint + slice-back); `policy_gradients/buffer.py::Experience.teacher_logprobs` field added; `harness/unified_trainer.py` rollout-time teacher caching via `_teacher_needs_entries`. Smoke `75316` ✓; main sweep `75338` ✓ (3 arms × seed 42 × 7h on itiger01 3×H100); §5-style taxonomy on OPSD ckpts `77371` ✓; multi-seed array `77369` ✓ (seeds 43–45 × 3 λ, all rc=0) — **confirms the single-seed headline** (λ=1.0 0.252±0.032, λ=0.50 0.269±0.055, λ=0.10 0.696±0.085 p@1), unimodal, no seed-bimodality → caveat (i) resolved. **Headline (vs §7.7 logit-teacher at the same λ/clip/seed)**: pure OPD λ=1 **rescued 6.5×** (0.029 → **0.188** p@1); mid-λ λ=0.50 flipped from dead to alive (0.099 → **0.238**, 2.4×); low-λ λ=0.10 slightly *worse* (0.709 → 0.665, −0.044). **Taxonomy reading reverses the proposal-stage mechanism prediction**: OPSD-trained students have MORE uncertain-bucket mass (81-82%) and LESS content mass (5-12%) than logit-trained at matched λ — opposite of the predicted "shift to content". Revised mechanism: answer-conditioning changes what the uncertain-bucket signal *means*, not where the mass lives; §5's bucket labels are decorrelated from per-token signal quality once the teacher's conditioning changes. Multi-seed + OPSD-on-cross-task + answer-conditioned-as-diagnostic-teacher remain open. |

| **Exp 6 — PRM-reweighted OPSD (variant c)** | [[prms-as-teachers]] (#13, variant c) | **done — negative (2026-06-07)** | New code: `harness/config.py` (`prm_reweight`/`prm_source`/`prm_weight_fn`/`prm_temperature`/`prm_weight_ceiling` + validators + `opsd_prm` recipe), `harness/teachers.py` (`_conditioned_logprobs(hint_fn)` + `answer_info_gain()`), `harness/distill_losses.py` (`prm_importance_weights()` + reweighted `reverse_kl_distill_advantage`), `harness/unified_trainer.py` (rollout-time compute/cache into `Experience.prm_weights` + W&B `prm/weight_*`); config/launcher/smokes `exp6_*`. GPU smoke `111266` ✓; main `111363` ✓ (4 arms × seed 42, 4×H100, 9h28m); results `results/exp6_prm_reweighted_opsd_111363/`. **Headline test (does importance reweighting REPLACE the blunt clip?) → NO.** p@1: A clip/no-rw **0.204**, B noclip/no-rw **0.007**, **C noclip+rw 0.006 (collapses *harder*, grad_norm→45, pass@16 0.031<B's 0.078)**, D clip+rw **0.157** (reweight *hurts* vs A even stacked). **Mechanism (§7.11): a mass-preserving reweight *concentrates* per-token KL mass → *sharpens* the heavy tail the clip exists to *bound* (§8.2) → fires the §7.6 dead-zone collapse harder.** Clip confirmed a *structural* stabilizer of the on-policy reverse-KL dynamic, not an ad-hoc instrument awaiting replacement. Trained-PRM variant (b) still open (but predicted to need a bound too). Single seed (C dead → no multi-seed needed). |

| **Exp 7 — the SCALE test** | §8.4/§9 "what genuinely stays open" | **done (2026-06-11) — folded in as §7.12** | Native **7B-SFT student ← 7B-Instruct teacher** (1-GPU/arm, teacher co-resident, bnb 8-bit Adam), `gsm_symbolic`, 250 steps, seed 42; config `harness/configs/exp7_scale_7b.yaml`, launcher `run_exp7_arm_1gpu.sh`; jobs `115090/115092/115093/115094`; results `results/exp7_1gpu_{A,B,C,D}_*/`. **p@1: A onpolicy pure-OPD noclip 0.0088 (DELAYED COLLAPSE at step ~152 — dead-zone attractor is scale-robust), C +clip=1.0 0.3232 (clip alone rescues pure on-policy OPD at 7B, 37× same-teacher), B offpolicy revKD 0.4082, D low-λ clip 0.6611 (best; also highest-entropy — the §7.8 diversity↔precision trade dissolves at 7B).** On↔off-policy gap 38–55× (1B) → 1.26× (7B): §8.1's "on-policy is a liability" is a small-student artifact, conditional on the clip. CONFOUND: scale vs capacity-gap-closure (7B←7B changes both) → Exp 7b/7c below. |
| **Exp 7b — 1B clipped-pure-OPD ← Instruct (hedge closer)** | §7.12 finding-2 cross-teacher hedge | **running (submitted 2026-06-11, job `115135`)** | 1B-SFT ← 7B-**Instruct**, alpha=1 lam=1 **clip=1.0**, seeds 42/43, otherwise verbatim the `opd_diff_teachers.yaml` protocol that produced the unclipped 0.005/0.008 arms; launcher `harness/run_exp7b_1b_clip_instruct.sh`. Closes §7.12 finding 2's hedge: the 1B "clipped pure-OPD ~0.05 dead" reference used the 7B-**SFT** teacher. If still ~0.05 → clip's accuracy payoff on pure OPD grows with scale (same-teacher confirmed); if ~0.3 → it was a teacher effect, finding 2 rewrite. |
| **Exp 7c — 7B ← 13B-Instruct (scale vs capacity-gap)** | §7.12 load-bearing caveat | **done (A/C 2026-06-12 jobs `115136`/`115137`; D 2026-06-20 job `121947`, after orig `115138` save-truncated)** | Same 7B-SFT student, **13B-Instruct teacher on a second card** (2×H100/arm; no co-residency at 13B; teacher downloaded to hf-cache, tokenizer verified identical to 7B), arms A13/C13/D13, 250 steps, seed 42; config `harness/configs/exp7c_scale_13b_teacher.yaml`, launcher `harness/run_exp7c_arm.sh`. **Disambiguation result: the confound resolves toward scale/gap-robust instability, NOT gap-closure.** Re-opening the capacity gap does *not* prevent the collapse — **A13 (noclip) still dives to the §7.6 dead zone** (train acc 0.52 → 0.005–0.009 over steps ~75–200, grad_norm→51), then oscillates violently out late (final-checkpoint eval p@1 0.237 is a snapshot of an *unstable* run, not a stable recovery). **C13 (clip) is stable end-to-end** (acc 0.21–0.68, grad_norm ≤5, never dead; p@1 **0.403** vs 7B←7B 0.323). So the unclipped on-policy reverse-KL collapse now reproduces across 1B←7B, 7B←7B, *and* 7B←13B; clip stays the structural stabilizer. New (n=1, cautious) wrinkle: the larger/more-informative teacher makes the dead zone *escapable-but-unstable* rather than a permanent attractor. **D13 (low-λ clip) is the best arm — p@1 0.746** (job `121947`, stable: acc 0.52→0.95→0.83, grad_norm ≤5, 0 collapse steps; vs 7B←7B 0.661) **and simultaneously the highest-entropy arm** (tok_ent 1.29) → the §7.8 precision↔coverage trade dissolves at 7B←13B too. Two completing reads: (i) the more-capable 13B teacher *lifts the stable clipped recipes* (C 0.323→0.403, D 0.661→0.746) but *cannot rescue the unstable unclipped arm* (A) → teacher capability helps **only once the on-policy signal is stabilized**; (ii) the rank order A ≪ C < D holds at every gap. |
| **Exp 8 — per-task hint search (GEPA-style)** | [[per-task-hint-search-gepa]] (#9) | **stage 1 done (2026-06-11; smoke `115141` → full search `115142`); stage 2 partial → completing (2026-06-23)** | The last unstarted tier-2 roadmap item, sharpened by §7.10+§7.12: does a searched **task-level** (non-privileged, entry-independent) hint rescue pure OPD the way §7.10's per-problem **answer** conditioning does? Stage 1 (`harness/hint_search.py`, no training): GEPA-style search over fixed teacher-conditioning strings, scored on the Lagrangian `teacher_acc − β·kl_p99` against ONE fixed 64×4 student-rollout set (kl_p99 = the §8.2 tail stat); 7B-Instruct as reflective mutator; seeds 8 + 6×3 mutations. **Stage 1 winner (job `115142`):** "Ensure each step logically follows from the previous, showing all calculations clearly; aim for a simple, straightforward solution." (acc=0.547, kl_p99=4.881, score=−0.429; best Lagrangian). Stage 2 (`harness/configs/exp8_fixed_hint_opd.yaml`, `harness/run_exp8_fixed_hint_opd.sh`): pure OPD λ=1 clip=1.0 from the 7B-SFT teacher + `condition_on=fixed_hint`, matched to the §7.10 protocol. Two arms: **best** (the searched winner) and **placebo** (task-irrelevant tone hint "Respond in a calm, friendly, and encouraging tone.") to isolate task-relevant shift from "any appended conditioning string". Read: hint-OPD ≈ 0.03–0.05 → the §7.10 rescue is per-problem *privilege*; ≈ 0.15–0.25 → distribution-shift onto answer-shaped paths generally, keeps the §8.4 unbiasedness leg. **Stage-2 partial results (2026-06-22, T=0.6, 64×16, eval-seed 1e6):** best/s42 p@1 **0.245**, best/s43 p@1 **0.126** (bimodal — the design's warned failure mode), placebo/s42 p@1 **0.210**. The placebo control is the scientifically load-bearing comparison and was **under-seeded** (1 seed vs best's 2): placebo/s42 lands *above* the best arm's mean (0.210 vs 0.186), which — if it holds up — would *flip* the verdict from "distribution-shift suffices" to "the rescue is just any appended conditioning string" (i.e. the §7.10 rescue is per-problem privilege, and task-level hint search is a dead end at λ=1). **Completing the matched 3-seed control (2026-06-23, jobs `126571`/`126572`/`126573`):** best/s44 + placebo/s43 + placebo/s44, bringing both arms to seeds {42,43,44} (the script's prescribed bimodality guard); running sequentially on 1×H100 (per-user H100 cap on `bigTiger`/`normal`), ETA ~06:00 2026-06-24. Verdict pending the placebo seed spread. |

_Legend: `not built` (code TODO) · `running` · `done` · `paused`._

---

## Current headline findings (2026-05-21)

- **Plain OPD loses badly to GRPO on `gsm_symbolic`.** Same-family 7B teachers move the student toward teacher log-probs, but pass@1 remains near zero while GRPO reaches ~0.6-0.7.
- **Teacher recipe is not the main variable.** SFT, DPO, and Instruct 7B teachers all produce similarly weak pure-OPD students when λ=1.0.
- **The teachers are competent.** Teacher pass@1 is ~0.45 and pass@64 is ~0.93-0.96, so pure OPD is not failing because the teacher cannot solve the task.
- **Pure OPD failure is state-space lock-in.** At λ=1.0, the student samples bad trajectories and the teacher's per-token corrections on those trajectories do not compose into globally correct multi-step reasoning. Clipping outlier KL terms does not fix this corner.
- **Interior expert-RL+OPD originally looked bimodal and seed-fragile.** Unclipped λ=0.05 with the task-specialized teacher split into high and low seed clusters because all seeds collapsed around steps 100-150 and only some recovered.
- **The latest mechanism result changes the story.** `per_token_kl_clip` fixes the interior collapse, but not uniformly across λ. At clip=1.0, λ∈{0.05,0.10,0.20} matches or beats GRPO-level pass@1 over 4 seeds; λ=0.10 is the current best point (mean 0.693 p@1 / 0.801 p@16).
- **Outcome reward is still load-bearing, and only in a bounded dose.** λ=1.0 pure OPD remains dead with clip=1.0, while low-λ clipped interior works. High teacher weight remains harmful: λ≥0.50 is still far below GRPO under clipping.
- **Cross-task is the resolving test, and it splits.** On off-task `simple_equations` (`71574`), GRPO wins p@1–p@16 by ~2× — the §7.4 generalisation tax is *not* cured by kl_clip. But clipped λ=0.10 **crosses over and beats GRPO at p@32 (0.565 vs 0.549) and p@64 (0.666 vs 0.607)** at T=0.6. The OPD-blend's high token entropy (~8×) and 2× distinct-2 diversity convert into coverage at high k but not into low-k accuracy off-task.
- **Headline.** The (α=1, λ=0.10, clip=1.0) interior is **crossover-dominant, not strictly dominant**: in-distribution narrowly better than GRPO at all k; off-task worse at low k and better at high k. The right pitch is "ProRL-style diversity-for-pass@k trade", not "free-lunch OPD beats RL".

---

## 1. Introduction

The "distributional lens" (Brown & Claude 2026): SFT, RL, and on-policy distillation aren't three unrelated recipes — they're three points in one space, asking *which objective puts probability mass where*. The post writes them as a single token-level policy gradient with three knobs:

- **α ∈ [0,1]** — how on-policy the data is (1 = sample from the current student; 0 = a fixed dataset / off-policy buffer);
- **λ ∈ [0,1]** — how much of the per-token advantage is the **teacher reverse-KL term** `log π_T − log π_θ` vs. the sequence-level **outcome reward**;
- **π_T** — *which* teacher (a dataset δ-distribution → SFT; none → RL; a bigger same-family checkpoint → OPD; the model conditioned on the answer → OPSD; a learned hint-writer; a PRM; …).

There are three ways to learn to cook. You can read a cookbook — every page is correct, but none of it is about the dish *you* just burned (SFT: dense feedback, off your own states). You can cook and taste only the finished plate — honest, but one bit of signal for an hour of work (RL: sparse feedback, on your states). Or you can cook while a chef tastes every bite as you go (on-policy distillation: dense feedback *and* on your states). OPD looks like the best of both — and that is exactly why it is worth taking apart. This work asks two questions the analogy hides: *which* chef, and what the chef's tasting actually changes about how you cook. It turns out that at a large student–teacher skill gap, a chef correcting every bite of a dish you were never going to plate is worse than re-reading the cookbook — and that a single, blunt rule about how loud the chef is allowed to shout is what separates a recipe that works from one that collapses.

**Why it matters.** Expert-merge by distillation is becoming the *last* stage of frontier post-training pipelines (GLM-5, DeepSeek-V4), so "which OPD objective, and when does it break" is now a production question, not a curiosity. The post's framing makes a falsifiable prediction — a sharpening↔coverage ordering, forward-KL SFT < RL < reverse-KL OPD/OPSD — and, more usefully, says SFT/RL/OPD/OPSD are not four recipes but four corners of *one* `(α, λ, π_T)` space whose interior nobody has mapped. The payoff of taking that literally is a reusable *map* of the space — which knob is load-bearing, which is redundant, where the cliffs are — rather than one more point on it.

**What we contribute.** A single unified `(α, λ, π_T)` trainer that reaches every corner *and* the interior through one code path, plus eight experiments that map the space at a 1B-student / 7B-teacher gap (scaled to 7B←7B and 7B←13B). The headline findings:
1. **Teacher recipe is not the variable.** Same-base SFT / DPO / Instruct 7B teachers all drive pure OPD to p@1 ≈ 0.01 (§4); novelty/recipe is off the critical path.
2. **On-policy state coverage is not load-bearing at this gap — the post's central claim reverses.** Off-policy SFT and off-policy reverse-KD on the *teacher's* traces beat on-policy pure-OPD by ~38–72× (§8.1); what separates winners from losers is whether the per-token signal lands on *coherent, answer-shaped, reachable* trajectories, not whose states they are. The failure is a *dynamical instability* of the on-policy reverse-KL signal, init-independent (§8.1) and scale-robust (§7.12).
3. **One scalar separates collapse from the best recipe.** A per-token KL clip is a *structural stabilizer*, not an ad-hoc instrument (§7.6/§7.7/§8.2): it bounds the heavy tail of per-token pushes that otherwise drives a step-100–150 death. With it, the **clipped low-λ interior `(α=1, λ=0.10, clip=1.0)`** is the study's best recipe — edging GRPO in-distribution and dominating it at high pass@k off-task (§7.8). An importance-reweight cannot replace the clip; it *concentrates* the tail the clip exists to bound (§7.11).
4. **Update geometry tracks loss structure, not the teacher.** Every α=1 trainer writes an RL-shaped *sparse* update regardless of teacher dose — SFT-from-rollouts is the sparsest of all (§6/§8.3); the variation is *which subspace* the update lives in, not sparse-vs-dense.
5. **The teacher interface carries privileged, per-problem information — or nothing.** Answer-conditioning the teacher (OPSD) rescues pure OPD (§7.10), but a non-privileged *task-level* hint does not beat a placebo (§7.13). The open frontier is density + on-policy + outcome-alignment *without* a privileged crutch (§8.4).

---

## 2. The (α, λ, π_T) meta-algorithm

The unified per-token policy gradient (`research/meta-algorithm-alpha-lambda.md`, `research/harness/distill_losses.py`):

```
grad J(θ)  =  E_{x, ŷ ~ π_α(·|x)}  [ Σ_t  A_t · grad log π_θ(ŷ_t | ŷ_<t) ]
A_t        =  λ · ( log π_T(ŷ_t | ŷ_<t) − log π_θ(ŷ_t | ŷ_<t) )   # teacher reverse-KL  (detached, optionally per-token-clipped)
            + (1 − λ) · A^outcome_t                                 # sequence-level verifier reward (GRPO/RLOO/...)
```

Corners: `(α=0, λ=1, π_T=δ_data)` → cross-entropy **SFT**; `(α=1, λ=0)` → outcome **RL** (GRPO/Dr.GRPO/GSPO/CISPO/RLOO/REINFORCE/PPO); `(α=1, λ=1, π_T=bigger same-family)` → **OPD**; `(α=1, λ=1, π_T=self+answer, + per-token KL clip)` → **OPSD**; `0<λ<1` → expert-RL+OPD.

**Implementation** (`research/harness/`): the trainer is `unified_trainer.py` — `_run_rl_loop` (the λ=0 corner; single-GPU or DDP) and `_run_distill_loop` (the λ>0 path: rollout → a frozen-teacher forward per training batch → `UnifiedTokenLoss`). Sketch of the λ>0 step:

```python
# rollout on-policy, score with the reasoning_gym verifier (this is _run_distill_loop, simplified)
seq, action_mask, attn, rewards, _ = rollout(model, entries, dataset, tokenizer, ...)
log_probs_old = compute_log_probs(model, seq, attn)            # the rollout policy
advantages    = compute_advantages(rewards, outcome_loss, ...) # zero-weighted at λ=1
# training updates on the replay buffer
for exp in experience_batches:
    log_probs       = compute_log_probs(model, exp.seq, exp.attn)            # requires grad
    teacher_logprobs = teacher.token_logprobs(exp.seq, exp.attn, exp.action_mask)  # frozen π_T
    loss = UnifiedTokenLoss(lam, per_token_kl_clip)(log_probs, exp, teacher_logprobs)
    (loss / batch_acc).backward(); ...; optimizer.step()
```

Logged each step: `reward` (rollout accuracy + format), `loss`, `grad_norm`, `off_policy/max_level` (drift of the current policy from the rollout policy), `teacher/reverse_kl` (mean `log π_rollout − log π_T` over generated tokens — should fall under OPD), throughput, GPU stats. Held-out eval every `eval_every` steps (`harness/eval_passk.py`): `eval/pass@k`, `eval/accuracy_mean`, `eval/token_entropy`, `eval/distinct_n`.

The four corners and the explored interior of the `(α, λ)` plane: `α=0,λ=1` (SFT), `α=1,λ=0` (RL/GRPO), `α=1,λ=1` (OPD/OPSD), and the clipped low-λ band `α=1, λ∈[0.05,0.20]` that this work finds is the only live interior.

---

## 3. Setup

- **Models — OLMo-2 family** (one tokenizer across 1B/7B/13B/32B → same-family teacher KL is well-defined, no tokenizer-mismatch tax; cf. [[cross-family-teacher-tax]]). Crucially, AllenAI released the *intermediate* post-training checkpoints, so we get **same-base teachers that differ only in recipe** off the shelf:
  | role | checkpoint | recipe |
  |---|---|---|
  | student | `allenai/OLMo-2-0425-1B-SFT` | SFT only (chat template; room to improve) |
  | teacher A | `allenai/OLMo-2-1124-7B-SFT` | SFT only |
  | teacher B | `allenai/OLMo-2-1124-7B-DPO` | SFT + DPO |
  | teacher C | `allenai/OLMo-2-1124-7B-Instruct` | SFT + DPO + RLVR/GRPO ("the RL teacher") |
- **Tasks** — `reasoning_gym` (already wired into the harness): `gsm_symbolic` (GSM8K-style math; the primary), plus `simple_equations` as a never-trained **cross-task** probe for the pass@k coverage story (§7.8).
- **Eval** — held-out `reasoning_gym` prompt set (seed disjoint from training): `pass@1..k`, accuracy, token-entropy, distinct-n (`harness/eval_passk.py`); plus the per-step `reward` / `teacher/reverse_kl` / `off_policy` traces.
- **Hardware / infra** — 8×H100 80 GB (`itiger01`, SLURM `bigTiger`); W&B project `distill-harness` (offline → sync); per-arm checkpoints `save_pretrained`'d to `harness/checkpoints/<arm>/`.
- **Held-constant within an experiment** — student init (per seed), prompts, rollout budget, eval — so the only thing that varies across arms is the named knob.

---

## 4. Experiment 1 — Does the teacher's recipe matter? ([[opd-different-teachers]] #1)

**Question.** Once the teacher is same-base / tokenizer-matched, how big is the "SFT-teacher vs RL-teacher" gap — and does *any* teacher beat plain RL on the student? `roadmap.md`'s prediction: with teacher compatibility controlled, the SFT-vs-RL gap is much smaller than people expect on easy/medium tasks, larger on frontier tasks where RL changes local reward ordering at crucial decision points.

**Design.** 8 arms = {OPD from teacher A (`-7B-SFT`), B (`-7B-DPO`), C (`-7B-Instruct`); GRPO-on-the-student baseline} × {seed 42, 43}. Plain OPD (λ=1) for the teacher arms; identical hyperparams everywhere (`harness/configs/opd_diff_teachers.yaml`: lr 1e-5, 8 prompts/step × 8 rollouts/prompt × 1024 max_new_tokens, 500 steps, `reasoning_gym/gsm_symbolic`). `reward = accuracy + 0.5·format_score` (format bonuses for `<think>`/`<answer>` tags); pass@k / accuracy / token-entropy / distinct-n on a **held-out** 64-prompt set at T=0.6, run at steps 250 and 500 (`harness/eval_passk.py`). Per-arm checkpoints saved to `harness/checkpoints/<arm>/`. `[Phase 1b: the off-policy-SFT-on-teacher-rollouts arm (isolates on-policy state coverage) is **now done — see §8.1** (jobs 80388/80389): off-policy SFT beats on-policy pure-OPD ~55–72× at p@1, and it's not the correctness filter. Still deferred: a 4th teacher *interface* (a verbal-judge and/or rubric/PRM teacher), which needs `teachers._build_inputs`.]`

**Results.** SLURM `71148` (itiger01, 8×H100, 7h28m, all 8 arms `OK`). Numbers below are mean ± half-range over seeds {42, 43}.

| arm | teacher recipe | `reward` @500 | **pass@1 @500** | pass@8 @500 | pass@16 @500 | `teacher/reverse_kl` @500 | token-entropy @500 |
|---|---|---|---|---|---|---|---|
| **GRPO baseline** | — (RL on student) | **1.16 ± 0.07** | **0.597 ± 0.074** | **0.662 ± 0.068** | **0.672 ± 0.062** | n/a | 0.45 ± 0.03 |
| OPD ← `-7B-SFT` | SFT | 0.47 ± 0.02 | 0.014 ± 0.002 | 0.086 ± 0.003 | 0.141 ± 0.000 | 0.10 ± 0.02 | 0.64 ± 0.06 |
| OPD ← `-7B-DPO` | SFT + DPO | 0.48 ± 0.01 | 0.011 ± 0.002 | 0.065 ± 0.013 | 0.102 ± 0.024 | 0.50 ± 0.03 | 2.38 ± 0.04 |
| OPD ← `-7B-Instruct` | SFT + DPO + RLVR/GRPO | 0.51 ± 0.02 | 0.013 ± 0.000 | 0.073 ± 0.003 | 0.117 ± 0.008 | 0.47 ± 0.01 | 2.47 ± 0.14 |

_(per-arm csv → `results/exp1_opd_teachers/<arm>.csv`; held-out eval json → `results/exp1_opd_teachers/evals.json`.)_

**Three headline findings.**

1. **Teacher-recipe gap: flat.** All three same-base 7B teachers produce essentially the same OPD outcome on `gsm_symbolic`: pass@1 in 0.009–0.015, pass@16 in 0.078–0.141, no monotonic ordering by teacher post-training recipe. Confirms `roadmap.md`'s prediction that, once same-family compatibility is controlled, the SFT-vs-DPO-vs-RL teacher gap is much smaller than the literature implies — on this easy/medium task. The OPD signal does pull the student toward the teacher (`rev_kl` ↓ from ~0.5–1.5 at init to 0.10–0.50 at step 500); the recipe of *which* teacher matters less than the fact of *a* teacher.
2. **OPD decisively loses to plain GRPO on `gsm_symbolic`.** The gap is **35×–75× on pass@1** and **4×–9× on pass@16** (the gap shrinks with k because the OPD students do *some* exploration at higher k, but never close to the verifier-driven GRPO model). The OPD arms' training `reward` saturates near 0.5, which is exactly the format component of `reward = accuracy + 0.5·format` if format=1 alone; the GRPO arms reach ~1.1–1.24 (format ≈ 1 + accuracy ≈ 0.5–0.7), matching their held-out pass@1. **Consistent with the OPD arms learning format / CoT style more than math correctness** — we can't decompose `reward` cleanly until `_pg.compute_rewards` is split into `reward/accuracy` + `reward/format` (small TODO).
3. **Entropy alone is not the right diagnostic.** Three corners of the (entropy × accuracy) plane:
   - **RL baseline**: low entropy (~0.45), high accuracy (~0.60).
   - **OPD ← `-7B-SFT`**: low entropy (~0.65), near-zero accuracy (~0.014).
   - **OPD ← `-7B-DPO` / `-7B-Instruct`**: high entropy (~2.4), near-zero accuracy (~0.012).

   The DPO/Instruct teachers leave the student with ~4× higher per-token entropy than the SFT-teacher arm — but none of that extra spread becomes correct math. So the extra entropy is **non-useful distributional spread** (general-instruction uncertainty inherited from the post-RL teacher's flatter distribution on these prompts), not exploration that converts to coverage. D4 later confirmed this at wider k: higher token entropy did not close the pass@k gap to GRPO.

   **Mechanism (see §4.2 for the full revised version after D3 landed)**: the failure is *not* that the teachers are hedge-distributions — D3 shows they get pass@1 ≈ 0.46. It's that reverse-KL OPD matches the teacher *on the student's own trajectories*, and a pass@1 ≈ 0.001 student's trajectories almost never enter the teacher's high-reward regions of state space. The per-token signal is content-correlated (D2) but on globally off-path trajectories.

**Plots** (`results/exp1_opd_teachers/`):

- `reward_over_steps.png` — the OPD arms plateau at ~0.45–0.55 by ~step 50; the GRPO baselines rise steadily through 500 steps to ~1.1–1.24. The format–accuracy split shows up as a clean ~0.5 step in the OPD curves.
- `revkl_over_steps.png` — `rev_kl` at step 1 ≈ {SFT-teacher 0.4–0.5, DPO 0.9–1.3, Instruct 1.1–1.5}; by step ~50 it's halved; final ≈ {0.08–0.12, 0.47–0.53, 0.46–0.48}. So the *gap* the OPD term closes is bigger for the more-RL'd teachers (expected — the post-RL teachers have moved further from `-1B-SFT`), but the *residual* `rev_kl` plateaus higher for them too.
- `gradnorm_over_steps.png` — OPD-Instruct/DPO grad norms (pre-clip) are ~3× the SFT-teacher arm's; the GRPO baseline runs at ~10× smaller grad norms — a direct read on "the OPD term is doing more work than the verifier-advantage term, even when both target the same student."
- `passk_at_final.png` — pass@k vs k @ step 500 (T=0.6, k≤16): the GRPO baseline curves sit ~30×–60× above all three OPD curves at every k tested. **Wider pass@k (k up to 64) at T=0.6 + 1.0, including the un-RL'd init, ran as SLURM `71158`** — Yue-et-al. crossover results live in D4 below.

### 4.1 Diagnostics — why did OPD lose so badly?

The Exp-1 numbers are interpretable two ways: (A) `roadmap.md`'s prediction held and OPD-from-same-base-teachers is genuinely uncompetitive with verifier RL here, or (B) we set OPD up to fail (the 7B teachers can't do the task, the OPD signal landed on style not content, or the loop has a bug). Three diagnostics run *after* Exp 1 separate these.

**D1 — OPD loop runs cleanly on a different task (`spell_backward` smoke, SLURM `71162`, 32 min, 3 arms × 50 steps on the configs' original placeholder task).** All three teacher arms drive `rev_kl` down sharply (`-7B-SFT`: 2.12 → 0.18; `-7B-DPO`: 3.57 → 0.43; `-7B-Instruct`: 3.89 → 1.08, then climbs back to 1.53 with grad-norm ~70 and loss −14 — destabilizes under the post-RL teacher), `reward` rises from 0.17 to ~0.50 (= the format component, again), and **eval pass@1 = 0 on every arm**. Same failure mode as `gsm_symbolic`: format mastered, task not. This narrowly rules out "the rollout/teacher-logprob/optimizer plumbing is silently broken" — `rev_kl` moves and steps complete — but **doesn't** validate GRPO grouping semantics, the λ-interior loss math (Exp 4 is the actual test of that), reward decomposition, or teacher-state alignment.

**D2 — Per-token KL × correctness (`harness/diagnose_per_token_kl.py`, OLMo-2-0425-1B-SFT student × OLMo-2-1124-7B-SFT teacher, 32 prompts × 4 samples, T=0.6, 512 max-new):**

| metric | **INIT** (`-1B-SFT`, pre-training) | **TRAINED** (`opd-sft7b-s42`, +500 steps) |
|---|---|---|
| n_tokens_total (per 128 completions) | 10,936 | **29,810** (~2.7× longer CoTs) |
| completion accuracy | 0.78 % | 1.56 % |
| `kl_mean` = mean (log π_T − log π_θ) | −0.248 | −0.078 (gap shrank 3×) |
| `kl_mean_on_correct` | −0.163 | −0.165 *(unchanged)* |
| `kl_mean_on_incorrect` | −0.249 | −0.078 *(student now matches teacher's hedging on the wrong tokens too)* |
| **`topq_format_lift`** (format among top-10 % \|KL\|) | **0.57** | **0.53** |
| **`topq_correct_lift`** (top-10 % \|KL\| ∈ correct completions) | **1.09** ≈ random | **2.10** |

Reading the two columns together: OPD training (a) tripled the student↔teacher distributional gap closure (`kl_mean` -0.25 → -0.08), (b) stretched the student's CoT length ~2.7×, (c) **doubled the per-token signal's correlation with downstream correctness** (top-10 % \|KL\| → correct-completion lift `1.09 → 2.10`), and (d) kept the signal **off format/style tokens** (format-lift ~0.55, before and after — substantially below 1.0). All of that is what OPD claims to do. **But `kl_mean_on_correct` is `≈ −0.16` both before and after** — at the answer-bearing tokens, the teacher is *more uncertain than the student*. The teacher isn't an oracle on these prompts; it's a hedge. OPD copied the hedge.

**D3 — Teachers' own pass@k on `gsm_symbolic` (SLURM `71161`, 3h39m, all OK, 128 prompts × 64 samples).** The teachers are *math-competent*, not hedge-distributions:

| teacher | T | pass@1 | pass@8 | pass@16 | pass@32 | pass@64 | token-entropy |
|---|---|---|---|---|---|---|---|
| `-7B-SFT` | 0.6 | 0.454 | 0.787 | 0.850 | 0.903 | **0.938** | 0.48 |
| `-7B-SFT` | 1.0 | 0.321 | 0.754 | 0.837 | 0.898 | **0.945** | 0.57 |
| `-7B-DPO` | 0.6 | 0.470 | 0.794 | 0.858 | 0.905 | **0.930** | 0.33 |
| `-7B-DPO` | 1.0 | 0.437 | 0.818 | 0.883 | 0.928 | **0.961** | 0.34 |
| `-7B-Instruct` | 0.6 | 0.462 | 0.750 | 0.814 | 0.866 | **0.906** | 0.29 |
| `-7B-Instruct` | 1.0 | 0.427 | 0.789 | 0.857 | 0.906 | **0.938** | 0.30 |

(json: `results/teacher_eval_71161/teacher-7b-{sft,dpo,instruct}.json`.) **This overturns the "hedge teacher" mechanism from earlier**: the teachers solve `gsm_symbolic` ~46% of the time at T=0.6 and ~93% at pass@64. The pass@1 gap student-after-OPD vs teacher is **~40×** (0.011 vs 0.46), and at pass@64 is **~4×** (0.24 vs 0.94). OPD failed to transfer math capability that *was there to transfer*.

**D4 — Wider pass@k including the un-RL'd init baseline (SLURM `71158`, k up to 64, T∈{0.6, 1.0}, 128 prompts × 64 samples; 9 ckpts).** Selected arms (seed 42 only shown — seed 43 within within-arm noise):

| arm | T | pass@1 | pass@8 | pass@16 | pass@32 | pass@64 | tok-ent | distinct-4 |
|---|---|---|---|---|---|---|---|---|
| **init** (`-1B-SFT`, no training) | 0.6 | 0.001 | 0.010 | 0.019 | 0.037 | **0.070** | 0.58 | 0.93 |
| **init** | 1.0 | 0.005 | 0.036 | 0.068 | 0.122 | **0.195** | 0.69 | 0.88 |
| OPD ← `-7B-SFT` | 0.6 | 0.011 | 0.070 | 0.116 | 0.176 | **0.242** | 0.62 | 0.64 |
| OPD ← `-7B-SFT` | 1.0 | 0.012 | 0.069 | 0.108 | 0.156 | **0.211** | 1.27 | 0.82 |
| OPD ← `-7B-DPO` | 0.6 | 0.011 | 0.062 | 0.097 | 0.140 | **0.188** | 2.34 | 0.56 |
| OPD ← `-7B-DPO` | 1.0 | 0.011 | 0.063 | 0.102 | 0.158 | **0.227** | 2.96 | 0.70 |
| OPD ← `-7B-Instruct` | 0.6 | 0.011 | 0.064 | 0.099 | 0.139 | **0.180** | 2.47 | 0.51 |
| OPD ← `-7B-Instruct` | 1.0 | 0.011 | 0.066 | 0.105 | 0.155 | **0.211** | 2.99 | 0.73 |
| GRPO baseline | 0.6 | 0.577 | 0.657 | 0.679 | 0.701 | **0.727** | 0.44 | **0.02** |
| GRPO baseline | 1.0 | 0.549 | 0.672 | 0.706 | 0.738 | **0.758** | 1.01 | **0.03** |

Two readings:

- **Yue-style crossover, at high temperature**: at T=1.0 pass@64, the un-RL'd init (0.195) is **essentially tied with** the OPD'd students (0.21–0.23). At T=0.6 OPD beats init 3–4× at pass@64, but the broad-coverage T=1.0 advantage is mostly gone after OPD training. The DPO/Instruct OPD arms at T=1.0 *do* outperform init at pass@64 (0.23 / 0.21 vs 0.20), but the margin is small — distillation-driven "exploration" replaced by hedge-noise that isn't payoff-correlated.
- **GRPO's collapse is real but doesn't hurt pass@k up to k=64**: distinct-4 = **0.02** (vs 0.93 at init) means the GRPO student emits essentially 50× less diverse 4-grams than the un-trained model, yet pass@64 stays at 0.73 — far ahead of every OPD arm. The Yue prediction (RL-collapse hurts pass@k at large k) doesn't materialize by k=64 on `gsm_symbolic` at the 1B scale; whether it shows up at k=256+ or on a harder task is the next step for [[pass-at-k-vs-pass-at-1]].

### 4.2 Revised conclusion + next experiment

Combining the headline table + D1–D4:

- **Teacher recipe gap: flat** (`{SFT, DPO, Instruct}` → pass@1 0.009–0.015, no ordering).
- **OPD vs GRPO: OPD decisively loses on `gsm_symbolic`** (35×–75× on pass@1, 4×–9× on pass@16; ~3× on pass@64).
- **The "teacher is a hedge" mechanism is wrong (D3).** The teachers solve `gsm_symbolic` 46% at pass@1 / 94% at pass@64. They *had* the math. OPD failed to transfer it. The puzzle is now **why the per-token-correlated signal (D2: top_correct_lift 2.10) didn't move pass@1 above 1.1%** when the teacher's own pass@1 is 46%.
- **Revised mechanism — on-policy state-space lock-in + capacity gap.** Reverse-KL OPD matches `π_T` *on the student's own trajectories*, not on the teacher's. When the student starts at pass@1 ≈ 0.001, the trajectories it samples almost never enter the teacher's high-reward regions of state space, so the per-token signal is the teacher's *local* opinion on a *bad* trajectory — content-correlated (D2) but globally off-path. Two pieces of evidence: (a) for the SFT-teacher arm, residual `rev_kl` ≈ 0.10 at step 500 — the student has essentially closed the gap *on its own samples* and still gets pass@1 ≈ 0.01; more steps won't help. (b) trained CoTs are 2.7× longer than init (D2: 29,810 vs 10,936 tokens / 128 completions) — the student is learning the teacher's *verbosity / structure* without absorbing the *math content*. A 1B student also likely lacks the capacity to represent the 7B teacher's joint multi-step reasoning distribution — per-token marginal match doesn't equal joint trajectory transfer.
- **Yue-style crossover lurks at high temperature (D4).** At T=1.0 pass@64, init ≈ 0.20 vs OPD'd students 0.21–0.23 — barely a margin. The OPD training mostly traded T=1.0 coverage for a small T=0.6 pass@k bump, without ever fixing pass@1. GRPO collapses distinct-4 to 0.02 (50× less diverse than init) yet still wins pass@64 (0.73). On `gsm_symbolic` at 1B, GRPO's collapse doesn't cost pass@64 — but [[pass-at-k-vs-pass-at-1]] still has room at k≥256 and on harder tasks.
- **Caveats.** 2 seeds is a small error bar (within-arm spread on GRPO ≈ 0.03 at pass@1). `reward` = `accuracy + 0.5·format` is still mixed in training logs — until `_pg.compute_rewards` is split, the "OPD arms learned format not math" reading is *consistent* with the data, not directly demonstrated. Entropy alone is not the right diagnostic (the three OPD corners of the (entropy, accuracy) plane disprove it).

**Next experiment — the (α, λ) interior.** The D2/D3 reading predicted that a *small* teacher term on top of GRPO should *help* (content-correlated signal from a math-competent teacher) while *large* λ should fail (on-policy lock-in). The Exp 4 results in §7 (2 seeds × 8 λ values, after a load-bearing trainer refactor on the (1-λ) outcome branch) **partially overturn this**: small λ doesn't help (λ ∈ {0.05, 0.10} mean pass@1 ≈ 0.01, dead zone), and the interior does have a peak — but it's **in the *mid*-λ band [0.20, 0.35]**, it's **stochastic** (one breakthrough arm per seed, exact λ location flips between seeds), and **no λ matches pure GRPO** (best interior arm 0.231 vs GRPO 0.577). The headline finding lives in §7 — the rest of this section's §4.2 prediction stands as an honest record of what the D-diagnostics suggested before the experiment ran.

---

## 5. Experiment 2 — Where does the per-token teacher signal concentrate? ([[per-token-kl-pivot-vs-style]] #4)

**Question.** The per-token reverse-KL `log π_T − log π_θ` isn't uniform — it concentrates on a few positions. *Which* positions: high-entropy student tokens (uncertainty), low-entropy/high-divergence tokens (confidently wrong), formatting/style tokens, or true "pivot" tokens whose local decision predicts a large downstream-reward swing? Is OPSD-style per-token clipping helpful *because* it suppresses high-KL/low-leverage tokens while preserving high-leverage pivots?

**Design.** Closed offline (the clip ablation is already in §7.7 / §7.10, so the remaining piece is the taxonomy). Roll each canonical student on 64 held-out `gsm_symbolic` prompts × 4 samples (T=0.6), with the same-base `OLMo-2-1124-7B-SFT` teacher providing per-token log-probs. Bucket every generated token into mutually-exclusive categories: **format** (tag/whitespace/boilerplate heuristic), **uncertain** (student token-entropy > 1.0 nats), **wrong_confident** (entropy < 0.2 nats AND `log π_T − log π_θ < −1.5`), **content** (everything else — the bulk reasoning text). For each bucket per arm: token count, |KL| mass fraction (Σ|kl|_bucket / Σ|kl|_total), mean entropy, mean kl, completion-correctness rate, and — the central question — what fraction of `per_token_kl_clip=1.0`'s removed mass came from that bucket. Code: extended `harness/diagnose_per_token_kl.py` with `categorize_token` + `taxonomy_summary`; launcher `harness/run_exp2_taxonomy.sh`; job `75121` (5 arms × ~5 min on 4×H100).

**Results.** Five canonical (student, teacher) pairs against the same `OLMo-2-1124-7B-SFT` teacher; mean kl, kl_p99, and `kl_heavy_tail_frac` (`|kl|>5`) are headline distribution stats per arm:

| arm | n_tok | accuracy | kl_mean | kl_p99 | heavy_tail_frac | topq_correct_lift |
|---|---:|---:|---:|---:|---:|---:|
| init | 19878 | 0.000 | -0.291 | 0.75 | 0.0137 | n/a |
| sft_student_s42 (§6.5) | 30460 | 0.430 | -0.319 | 0.68 | 0.0138 | 0.75 |
| grpo_v2_s42 (§7.5) | 26619 | 0.668 | -0.344 | 0.18 | 0.0077 | 0.83 |
| clip1, λ=0.10, s42 (§7.7 best) | 32899 | 0.793 | -0.406 | 1.17 | 0.0154 | 0.71 |
| clip1, λ=1.0, s42 (pure OPD dead) | 31842 | 0.047 | -0.370 | 0.77 | 0.0216 | 0.71 |

**Per-bucket KL mass — where the signal *lives* per arm:**

| arm | format mass | uncertain mass | wrong_confident mass | content mass |
|---|---:|---:|---:|---:|
| init | 0.089 | **0.607** | 0.014 | 0.289 |
| sft_student_s42 | **0.376** | 0.331 | 0.007 | 0.286 |
| grpo_v2_s42 | 0.123 | 0.134 | 0.140 | **0.603** |
| clip1, λ=0.10, s42 | 0.173 | **0.445** | 0.067 | 0.315 |
| clip1, λ=1.0, s42 (pure OPD) | 0.153 | **0.669** | 0.012 | 0.166 |

**Per-bucket clip-removal — what `per_token_kl_clip=1.0` actually disarms:**

| arm | total |KL| removed by clip | format clip share | uncertain clip share | wrong_conf clip share | content clip share |
|---|---:|---:|---:|---:|---:|
| init | 46.1% | 0.054 | **0.704** | 0.021 | 0.221 |
| sft_student_s42 | 49.8% | **0.597** | 0.277 | 0.009 | 0.117 |
| grpo_v2_s42 | 39.5% | 0.080 | 0.254 | 0.211 | **0.456** |
| clip1, λ=0.10, s42 | 44.8% | 0.222 | **0.536** | 0.094 | 0.147 |
| clip1, λ=1.0, s42 (pure OPD) | 56.8% | 0.163 | **0.791** | 0.016 | 0.030 |

_(JSONs: `results/exp2_taxonomy_75121/diag_*.json` + per-token JSONL traces `tokens_*.jsonl`.)_

**Four findings that close Exp 2.**

1. **Pure OPD's KL mass is overwhelmingly on uncertain (high-entropy) tokens.** λ=1.0 puts **67%** of its |KL| mass on the uncertain bucket (student entropy mean 2.56 nats in that bucket), only **17%** on content. So when pure OPD "learns from the teacher", the gradient is dominated by positions where the student has no concentrated prediction — exactly the places where the teacher's per-token "correction" is also a hedge over many alternatives. The mass goes where the teacher's information content is lowest. This is the mechanistic reason §7.7 found pure OPD dead even with clipping: the clipped-away mass is also from this bucket (79% of clip-removed mass from uncertain in λ=1.0), so clipping leaves the model with the residual non-uncertain teacher signal — but that residual is too small (only ~33% of mass total) to do useful work without an outcome anchor.
2. **GRPO's natural disagreement-with-teacher profile is content-concentrated.** GRPO never trained with a teacher term, but evaluating its trained policy against the 7B-SFT teacher's distribution gives **60% of |KL| mass on content**, only 13% on uncertain — the inverse of pure OPD. GRPO's policy disagrees with the teacher on the actual reasoning tokens (the body of the CoT), not on uncertainty hedges. And its disagreement is *high-leverage*: the wrong_confident bucket carries 14% of mass on just 2% of tokens. So GRPO ends up at a *content-aligned* (but not teacher-aligned) policy by following the verifier signal — independent confirmation that the outcome reward picks out content-relevant token positions, even without any teacher term.
3. **The clipped λ=0.10 winner sits between the two.** Its KL mass is **45% uncertain, 32% content, 17% format, 7% wrong_confident** — a partial inheritance from both endpoints. Crucially, the per-token clip's removed mass is **54% from uncertain, 15% from content, 22% from format**: the clip *selectively suppresses* the uncertain bucket (where the teacher's signal is noisy hedge) while leaving most of the content-bucket signal intact. This is the proposal's "clipping is precision-targeting the right positions" hypothesis confirmed at the bucket level — the clip is **not** an across-the-board attenuation; it's a structural reweighting that lifts the relative weight of content KL by removing uncertain outliers.
4. **The SFT control has a uniquely format-heavy disagreement profile.** sft_student's |KL| mass is **38% format vs 33% uncertain vs 29% content** — the highest format share of any arm. And the clip-removed mass is **60% format**. This matches §6.5's third-regime finding: SFT-from-rollouts learned the teacher's stylistic structure (CoT format, tag emission, boilerplate vocabulary) first and most, and the lingering disagreement with the teacher concentrates on those same stylistic positions. The 1.6%-of-weights-moved we measured in §6.5 is moving them mostly to match format tokens, not to do new math.

**Mechanism summary (the §7 mechanism becomes quantitative).** §7.7's claim was "per_token_kl_clip = 1.0 fixes the low-λ collapse by bounding outlier per-token KL pushes." §8.2 made this quantitative at the *p99 / heavy_tail* level. **§5 closes the loop at the *bucket* level**: clipping disarms exactly the bucket (uncertain) where the teacher's per-token signal is least informative, which (a) explains why clipping rescues low-λ training (the destabilising signal was concentrated on student-uncertain positions), (b) explains why clipping doesn't rescue λ=1 (without an outcome anchor, even the content-bucket signal is on globally-wrong trajectories per §7.2's state-space lock-in), and (c) reframes the proposal's "pivot vs style" dichotomy: there isn't a meaningful "pivot" set in this data — there's a **content bucket** (~70% of tokens, ~30-60% of mass depending on arm) and a few small auxiliary buckets, and clipping operates by holding the content bucket mostly intact while compressing the uncertain bucket.

**Caveats.** (i) The taxonomy is heuristic — "format" uses a substring list, "uncertain"/"wrong_confident" use fixed entropy thresholds (1.0 nats / 0.2 nats). Robustness checks (sliding thresholds, learned format classifier) would strengthen the result but unlikely change the qualitative picture. (ii) The "pivot" bucket from the proposal (intervene-and-see causal leverage) is not directly measured here; the wrong_confident bucket is the closest proxy and it is small (0.1-2% of tokens). A counterfactual rollout study would be needed to identify true pivots, but the within-arm consistency (uncertain dominates pure OPD; content dominates GRPO; format dominates SFT) suggests this is not the binding constraint. (iii) Eval-seed n=4 samples is small; we kept the 64-prompt budget to match §7.4 / §7.8.

---

## 6. Experiment 3 — What does OPD do to the weights? ([[sparse-vs-dense-updates]] #5)

**Question.** RL fine-tuning modifies a sparse subnetwork (~5–30% of weights, full-rank); SFT updates are dense and redundant (prune them and SFT degrades far slower than RL). Where does OPD sit on the sparse↔dense / redundant↔essential axis — SFT-shaped (because a family-calibrated teacher), or its own thing? Does "what teacher" ≈ "what update geometry"?

**Design.** Snapshot Δθ for each canonical checkpoint (RL baseline; GRPO-v2 multi-seed; the clipped λ-interior winners from §7.7; the dead-zone high-λ + pure-OPD corner; the unclipped v2.1 λ∈{0.05, 1.0} for breakthrough-vs-collapse). Compute, per tensor and aggregated:
- |Δθ| quantiles: mean, p50, p90, p99 of absolute deviation
- top-K% mass concentration: fraction of total Σ|Δθ|² in the top K% of |Δθ| values, for K∈{1, 5, 20} (1.0 = perfectly sparse; K_pct = perfectly dense)
- changed-fraction at thresholds: fraction of weights with |Δθ| > {1e-5, 1e-4, 1e-3}
- by-category aggregation: embed / attn_qkv / attn_o / mlp_in / mlp_down / norm
- (planned) ΔW effective rank and pruning-degradation curves

Utility: `harness/delta_theta_snapshot.py` (`--batch` mode hits 11 ckpts in ~25 min). Outputs: `figs/dtheta/dtheta_<label>.json` and the aggregate summary `figs/dtheta/dtheta_summary.json`.

**Results — sparsity / changed-fraction across arms (2026-05-22):**

| arm                        | top-1% mass | top-5% mass | top-20% mass | changed > 1e-4 | changed > 1e-3 | p99 \|Δθ\| |
|----------------------------|---:|---:|---:|---:|---:|---:|
| rl_baseline_s42            | 0.573 | 0.918 | 1.000 | 0.093 | 0.019 | 1.28e-3 |
| **grpo_v2_s42**            | **0.632** | **0.943** | 1.000 | **0.059** | **0.000** | **3.83e-4** |
| grpo_v2_s43                | 0.569 | 0.925 | 1.000 | 0.085 | 0.013 | 1.03e-3 |
| clip1, λ=0.05, s42         | 0.575 | 0.913 | 1.000 | 0.099 | 0.024 | 1.54e-3 |
| clip1, λ=0.10, s42         | 0.577 | 0.921 | 1.000 | 0.094 | 0.021 | 1.47e-3 |
| clip1, λ=0.20, s42         | 0.570 | 0.922 | 1.000 | 0.096 | 0.021 | 1.49e-3 |
| clip1, λ=0.50, s42         | 0.615 | 0.940 | 1.000 | 0.081 | 0.006 | 7.00e-4 |
| clip1, λ=0.85, s42         | 0.614 | 0.935 | 1.000 | 0.086 | 0.007 | 7.22e-4 |
| **clip1, λ=1.00 (pure OPD), s42** | **0.611** | **0.931** | 1.000 | **0.089** | **0.007** | **7.44e-4** |
| v2.1 unclipped λ=0.05, s42 | 0.573 | 0.918 | 1.000 | 0.101 | 0.024 | 1.57e-3 |
| v2.1 unclipped λ=1.00, s42 | 0.588 | 0.934 | 1.000 | 0.097 | 0.018 | 2.03e-3 |

_Sparsity bar chart: `research/figs/dtheta/exp3_dtheta_sparsity.png`; per-submodule heatmap: `research/figs/dtheta/exp3_dtheta_by_category.png`. Per-checkpoint JSON: `research/figs/dtheta/dtheta_*.json`._

**Reading.**

1. **All α=1 arms are RL-sparse.** Every arm — RL baseline, GRPO, clipped low-λ interior, clipped high-λ, pure OPD, unclipped v2.1 — sits in the same regime: top-1% mass between 0.57 and 0.63, top-5% mass between 0.91 and 0.94, only 6–10% of weights moved by >1e-4. The SFT-dense pattern the literature predicts for distillation does **not** appear. **Pure reverse-KL OPD (λ=1) on a same-base teacher produces an RL-shaped sparse update, not an SFT-shaped dense update.** The §8.3 hypothesis "OPD's update geometry tracks the teacher's recipe (RL-teacher → sparse, SFT-teacher → dense)" is **falsified for this setup** — every α=1 trainer in this sweep ends up sparse regardless of how much teacher signal is mixed in.

2. **Two visible sparsity tiers (within the same "all sparse" regime).**
   - **Sharper tier** (top-1% mass 0.61–0.63, changed>1e-4 = 0.06–0.09, p99|Δθ| ≤ 1e-3): GRPO-v2-s42, clip1 λ=0.50, clip1 λ=0.85, clip1 λ=1.0 (pure OPD).
   - **Broader tier** (top-1% mass 0.57–0.58, changed>1e-4 = 0.08–0.10, p99|Δθ| = 1–2e-3): RL baseline, GRPO-v2-s43, clip1 low-λ winners (λ∈{0.05, 0.10, 0.20}), unclipped v2.1 λ=0.05.
   - The split is **not** "more teacher signal → sparser" cleanly: pure OPD is sharper than the breakthrough clipped low-λ interior, but rl_baseline (no teacher at all) is in the broader tier. The pattern is closer to *how much the loss tells the model to change*: the clipped-trainer at low λ has the broadest update because the outcome branch dominates the loss and the clip leaves a wide thin tail of small-magnitude per-token KL corrections; pure OPD and high-λ converge tighter because the teacher's pointwise corrections are aligned and don't fight each other.

3. **GRPO-v2 has substantial seed-to-seed variance in sparsity.** s42 (top-1%=0.632, changed>1e-4=0.059) is the *sparsest* update in the whole batch; s43 (top-1%=0.569, changed>1e-4=0.085) is in the broader tier alongside the clipped low-λ arms. Same trainer, same hyperparameters, different seed → different sparsity. So single-seed sparsity numbers from the literature should be read cautiously.

4. **Submodule story (`exp3_dtheta_by_category.png`).** Every arm concentrates the changed-fraction in `attn_qkv` (13–18% of weights moved by >1e-4) > `embed` (1–13%) > `attn_o` ≈ `mlp_in` ≈ `mlp_down` (6–10%) > `norm` (~0%). The biggest contrast is in the **embedding**: GRPO-v2-s42 and the clipped high-λ / pure-OPD arms touch only 1–3% of embeddings; the clipped low-λ arms and unclipped v2.1 λ=0.05 touch 11–13% of embeddings. So the "broader" sparsity tier mostly comes from broader embedding updates — the policy is reshaping its token-level prior more, not its attention/MLP backbone more.

**What this means for §8.3.** The post's geometry story — "OPD picks a point in update-geometry space, and the meta-algorithm chooses where" — is supported in spirit (the (α=1, λ, clip) interior does produce *different* sparsity points, with the embed-touched broader tier sitting next to the RL baseline and the embed-untouched sharper tier sitting next to GRPO and pure OPD), but the strong version — "RL-teacher → RL-sparse, SFT-teacher → SFT-dense" — is **not** what we see at α=1 on a same-base, same-tokenizer teacher with this sample size. Every trainer in the α=1 family is RL-sparse, and the variation within is more about which submodules carry the change than how much of the model moves overall.

### 6.1 Prune-degradation curves (job `72097`, 2026-05-23)

The static §6 sparsity reading raised a question: are the two visible tiers (sharper vs broader) *functionally* different, or just two appearances of the same redundancy regime? The canonical RL's-Razor test is dynamic: zero the bottom p% of moved weights (revert them to base init) and re-evaluate pass@1. If the broader-tier arms (RL baseline, clipped low-λ) lose less from bottom-pruning than the sharper-tier arms (GRPO-v2-s42, pure OPD), the broader spread is genuinely redundant ↔ same functional sparsity. If they lose just as much, the tiers are distinct functional updates, not just denser noise.

**Setup.** `harness/prune_dtheta_eval.py` (build pruned state-dict in memory, swap into model, call `evaluate_passk` directly). Prune semantics: among the weights that actually moved (|Δθ|>0), revert the bottom `prune_pct` to base. The "moved" subset matters: bf16 storage makes ~85% of weights bit-identical between base and ckpt — pruning by percentile of all-weights would put almost all prune-pct values inside that zero-spike. SLURM `72097` (4 ckpts × 5 prune levels = 20 evals on 4×H100, ~90 min wall, T=0.6, n_prompts=64, n_samples=16, k∈{1,2,4,8,16}).

**Results (T=0.6, gsm_symbolic, eval_seed=1M):**

| arm | frac moved | p@1 @p=0% | p@1 @p=10% | p@1 @p=30% | p@1 @p=50% | p@1 @p=90% | p@16 @p=90% |
|---|---:|---:|---:|---:|---:|---:|---:|
| rl_baseline_s42 (broader) | 16.6% | 0.552 | 0.544 | 0.556 | 0.558 | **0.003** | 0.047 |
| GRPO-v2-s42 (sharper) | 13.5% | 0.540 | 0.546 | 0.548 | 0.552 | **0.439** | **0.734** |
| clip1, λ=0.10, s42 (broader) | 16.4% | 0.691 | 0.686 | 0.693 | 0.696 | **0.008** | 0.078 |
| clip1, λ=1.00, s42 (pure OPD, dead) | 14.3% | 0.047 | 0.039 | 0.037 | 0.029 | **0.161** | 0.344 |

_Plots: `research/figs/dtheta/exp3_prune_curves_polished.png` (blog-ready headline) + `exp3_prune_curves.png` (raw 2-panel diagnostic). JSON: `research/figs/dtheta/prune_sweep_72097/*.json`. The prune_pct column is fraction of MOVED weights; the corresponding fraction of TOTAL weights is ~1.8-2.2% at p=10%, ~5-6% at p=30%, ~7-8% at p=50%, ~12-15% at p=90%._

**Reading.**

1. **The bottom 50% of moves is functionally dead weight for every healthy arm.** RL baseline, GRPO-v2-s42, and clip1 λ=0.10 all retain pass@1 within ±0.02 of the unpruned baseline up to **p=50%** of moves reverted (= 7-8% of total params). The static §6 top-K%-mass numbers had functional meaning: the 6-10% of weights with the larger |Δθ| do all the work; the small-magnitude tail is genuinely redundant. This is the cleanest empirical confirmation of an RL's-Razor-style pattern in this writeup.

2. **The §6 two tiers are functionally distinct at high-prune.** At **p=90%** (only the top 10% of moves kept):
   - **Sharper-tier (GRPO-v2-s42):** p@1 = 0.439 (~81% of unpruned). Its top 10% of moves carries most of the function. Strikingly, **p@16 *increases*** from 0.625 → 0.734 at p=90%, which is consistent with high-prune acting as a regularizer that bumps back toward the SFT init's broader distribution.
   - **Broader-tier (RL baseline + clip1 λ=0.10):** both collapse to ~0 (rl=0.003, clip1=0.008). The narrow top 10% of their moves is *not* enough; their function depends on the broader spread of mid-magnitude changes. So "broader" in §6 wasn't an artifact — it's a real functional dispersal.

3. **Pure OPD λ=1 is anomalous, in the expected way.** Unpruned p@1 = 0.047 (consistent with §7.5/§7.7 "pure OPD is dead at this gap"). As we prune **more** of its moves, p@1 *improves*: 0.047 → 0.029 (worst at p=50%) → **0.161 at p=90%** (3.4× the unpruned value). The pure-OPD trainer moved 14.3% of weights, but those moves are *misaligned* with the reward — reverting most of them undoes mistargeting. This makes the "pure OPD update is dead and the trainer can't find it" mechanism quantitative: the *useful* fraction of pure OPD's moves is so small that retaining only the top 10% gives a better model than keeping all the moves. The geometry is sparse and *also* poorly aimed.

4. **clip1 λ=0.10 and rl_baseline have nearly indistinguishable prune profiles** — same fraction moved (16.4% vs 16.6%), same flat plateau to p=50%, same collapse at p=90%. The clipped low-λ trainer doesn't reshape the *structure* of the update relative to pure RL; it shifts the *values*. That's a sharp geometric statement: in the (α=1, λ, clip) family, **the teacher dose changes which weights move where, but not which weights move at all**, and the "shape" of the resulting subnetwork tracks the trainer family, not the teacher.

**What this closes about §8.3.** The post's geometry story — "OPD picks a point in update-geometry space" — gets a cleaner version after these curves: at the (α=1, same-base teacher) regime we explored, **all live arms produce the same functional sparsity pattern** (top ~6-8% of total weights = essential subnetwork, bottom 5-8% of moves = redundant). The difference between sharper and broader tiers is not "different geometry" but "how concentrated the essential subnetwork is": GRPO's essential subnetwork is *more concentrated* (top 10% of moves keeps 80% of perf), the broader-tier arms' essential subnetwork is *spread across more weights* (top 10% only keeps 0-1% of perf). And pure OPD's "essential subnetwork" is so small or so wrongly aimed that bottom-pruning is monotonically *helpful*.

### 6.2 Cliff localization — extended prune sweep (job `72122`, 2026-05-23)

**Question.** The §6.1 sweep had a 4× gap between p=50% (flat for healthy arms) and p=90% (broader-tier death + GRPO retention). Where exactly is the cliff for the broader arms? Does GRPO's gradual decline keep holding past p=90%? Does pure OPD continue to improve?

**Setup.** Re-use `harness/run_dtheta_prune_sweep_fine.sh` (same 4 ckpts, same eval protocol, same eval seed) with `prune_pct ∈ {0.60, 0.70, 0.80, 0.85, 0.95}`. 20 evals on 4×H100s, ~90 min wall. Results merged with §6.1 to give 10 prune levels per arm: {0, 10, 30, 50, 60, 70, 80, 85, 90, 95}.

**Results (combined, T=0.6, gsm_symbolic, p@1):**

| prune% (of moved) | rl_baseline | GRPO-v2-s42 | clip1 λ=0.10 | pure OPD λ=1 |
|---:|---:|---:|---:|---:|
| 0   | 0.552 | 0.540 | 0.691 | 0.047 |
| 10  | 0.544 | 0.546 | 0.686 | 0.039 |
| 30  | 0.556 | 0.548 | 0.693 | 0.037 |
| 50  | 0.558 | 0.552 | 0.696 | 0.029 |
| 60  | 0.558 | 0.528 | 0.685 | 0.029 |
| 70  | 0.563 | 0.520 | 0.651 | 0.035 |
| **80**  | **0.483** | 0.516 | **0.508** | 0.039 |
| **85**  | **0.288** | 0.499 | **0.382** | 0.087 |
| **90**  | **0.003** | 0.439 | **0.008** | **0.161** |
| **95**  | **0.003** | **0.266** | **0.001** | **0.242** |

_Plots: extended polished version at `research/figs/dtheta/exp3_prune_curves_polished.png` (10 prune levels, in-panel callouts); diagnostic 2-panel `exp3_prune_curves.png` (now 5-level only — fine-cliff updates separately). JSONs: `research/figs/dtheta/prune_sweep_72097/` + `prune_sweep_fine_72122/`._

**Reading.**

1. **The broader-tier cliff is localized to p∈[70%, 90%].** rl_baseline and clip1 λ=0.10 both:
   - hold +0-2% above unpruned through p=70%,
   - drop ~12-27% at p=80%,
   - drop ~45-48% at p=85%,
   - collapse to near-zero (0.001-0.008) at p=90% and stay dead at p=95%.
   The cliff is **steep and concentrated in a ~20-percentage-point window of prune fraction** (or ~3 percentage points of TOTAL params reverted). Between p=80% and p=90%, both arms lose essentially everything.

2. **GRPO-v2-s42 has NO cliff — it's a gradual decline all the way.**
   - p=70% → -3.8%, p=80% → -4.5%, p=85% → -7.6%, p=90% → -18.6%, p=95% → -50.8%.
   At p=95% (= only top 5% of moves kept, ~0.7% of total params), GRPO still retains **half its pass@1**. This is qualitatively different from the broader-tier arms — GRPO's essential subnetwork is genuinely small and tightly concentrated; the rest of its update is *graded*-redundant, not all-or-nothing redundant.

3. **Pure OPD λ=1 keeps improving monotonically with pruning.** Through the cliff window the broader arms die in, pure OPD goes from p@1=0.039 (at p=80%) → 0.087 → 0.161 → **0.242 at p=95%**. That last number is **5× the unpruned ckpt's p@1=0.047** and **80× the dead rl_baseline @ p=95%**. The reading: pure OPD's update is so misaligned that the smaller you make it (revert more moves toward init), the better the model becomes. Keeping only the top ~5% of pure OPD's moves (~0.7% of total params, τ at the 95th percentile of |Δθ|) recovers most of what the base SFT init was already capable of, *minus* the harm the rest of the OPD update was doing.

4. **Geometric summary of the four arms (post-cliff data):**
   - **GRPO-v2 (sharper-tier):** small, *concentrated* essential subnetwork — gradual graceful degradation; the top 5% of moves still carries half the function.
   - **rl_baseline, clip1 λ=0.10 (broader-tier):** larger essential subnetwork dispersed across more weights — flat plateau until ~70-80% pruned, then catastrophic cliff.
   - **pure OPD λ=1:** essentially no useful subnetwork — every move is harm, so the more you revert, the better. The "essential subnetwork" if any is in the very-top quantile (probably ≤5%) and even that doesn't recover GRPO-level performance because the *direction* of those moves is also wrong.

**Connecting back to §6 static sparsity.** §6 reported all four arms have top-1% mass ≈ 0.57-0.63 and top-5% mass ≈ 0.91-0.94 — visually they look like the same sparsity regime. The cliff sweep shows that **the static numbers were missing a structural distinction**: GRPO's "essential" weights are not just concentrated in mass but *functionally* concentrated (top-5% of moves does the work); broader-tier arms' essential weights are dispersed (top-5% of moves is not enough). Static mass concentration ≠ functional concentration; the prune curves reveal the difference.

### 6.3 Effective rank of ΔW — broader-tier is LOWER-rank, not higher-rank (2026-05-23)

**Setup.** Re-run `delta_theta_snapshot.py` over the 11 canonical checkpoints with `--no-effrank` *off*, computing the **effective rank** of every 2D `weight` matrix in ΔW: `effective_rank = exp(H(σ_i / Σ σ_j))` where `H` is Shannon entropy of the normalized singular spectrum. Matrices >4096 in either dim use a random-projection proxy (preserves spectral entropy in expectation; needed for the embed/lm_head 100K×2048 case). Per-arm, report the mean ΔW eff-rank vs base eff-rank for `attn_qkv` — the submodule that always has the largest frob norm and is the cleanest cross-arm comparison.

**Why `attn_qkv` is the right comparison submodule.** The static §6 numbers showed `attn_qkv` is the most-changed submodule in every arm (changed-fraction 0.13-0.18 of weights), it's always in the top-10-by-frob across all checkpoints (so it's reliably measured here even without the full per-tensor pass), and it's where the meaningful update direction lives. Embed effective ranks are noisier (rows for rare tokens never change), MLP effective ranks track each other tightly.

**Result — attn_qkv mean ΔW eff-rank / base eff-rank:**

| arm | tier (from §6) | ΔW eff-rank | base eff-rank | **ratio** |
|---|---|---:|---:|---:|
| RL baseline | broader | 1019.6 | 1169.6 | **0.872** |
| clip λ=0.05 | broader | 1005.2 | 1260.2 | **0.798** |
| **clip λ=0.10** | **broader** | **978.8** | **1276.2** | **0.767** ← lowest |
| clip λ=0.20 | broader | 1079.4 | 1208.1 | **0.893** |
| v2.1 unclipped λ=0.05 | broader | 1065.2 | 1271.0 | **0.838** |
| GRPO-v2 s42 | sharper | 1092.8 | 1096.7 | **0.996** |
| GRPO-v2 s43 | sharper | 1068.3 | 1122.9 | **0.951** |
| clip λ=0.50 | sharper | 1119.2 | 1155.2 | **0.969** |
| clip λ=0.85 | sharper | 1113.3 | 1076.8 | **1.034** |
| **pure OPD λ=1** | **sharper** | **1151.4** | **1090.4** | **1.056** ← highest |
| v2.1 unclipped λ=1 | sharper | 1127.5 | 1267.2 | 0.890 (outlier, but dead-zone) |

_Plot: `research/figs/dtheta/exp3_attn_qkv_effrank_ratio.png` (color-coded by tier). Raw JSON: `research/figs/dtheta/dtheta_*.json`._

**Reading — the prediction was wrong, and this is the more interesting result.**

The §6 follow-up I wrote predicted "sharper-tier (pure OPD, GRPO-v2-s42) should have lower effective rank on attn_qkv ΔW than the broader-tier; this would split the geometry story between 'fewer weights moved' and 'lower-rank update structure'". **The data shows the opposite:**

- **Broader-tier ΔW is LOWER rank than base.** All 5 broader-tier arms have ratio 0.77-0.89 — meaning their attn_qkv updates compress the spectral structure relative to the SFT init. clip λ=0.10 is the lowest at 0.767.
- **Sharper-tier ΔW preserves or slightly increases base rank.** GRPO-v2-s42 = 0.996 (essentially identical to base). Pure OPD λ=1 = 1.056 (*adds* spectral dimensions).
- The single sharper-tier exception is v2.1 unclipped λ=1 (0.890), which is the unclipped-collapse dead-zone arm — its ΔW is a mess of off-trajectory teacher pushes, lower-rank because the corrections concentrate in a few badly-aimed directions.

**What this means structurally.**

The two §6 tiers turn out to be two *qualitatively different* geometric updates, not the same kind of update at different magnitudes:

- **Sharper tier (GRPO, clip λ≥0.50, pure OPD).** Few weights move, but the moves are **high-rank / spread across many directions**. Geometrically: a "preserve the existing spectral basis, edit values within it" update. Combined with §6.1's gradual prune degradation, the reading is: each individual move contributes a small piece of a full-rank update; lose ~5% of moves and you lose ~50% of perf because the lost dimensions matter; lose any individual one and you lose almost nothing because each one carries ~1/rank of the signal.

- **Broader tier (clip low-λ, RL baseline, v2.1 unclipped low-λ).** More weights move, but the moves are **low-rank / concentrated in fewer directions**. Geometrically: a "shift the existing spectral basis along a few key axes" update. Combined with §6.1's sharp cliff, the reading is: the essential subnetwork is the set of small-magnitude moves needed to *fill in* a low-rank update; remove enough of them and you cross a phase transition where the rank-N update collapses to a rank-M update with M << N, and the model dies.

This **reverses what the static §6 reading suggested** about which tier was "more concentrated". GRPO looked like the most concentrated arm by static top-1% mass (0.632 > others' 0.57-0.62), but on the dynamic measures (prune curves + effective rank), GRPO is actually the **most distributed** update across spectral dimensions — its weights are concentrated in *count* but spread in *direction*. The broader-tier arms are the opposite: distributed across *more weights* but concentrated in *fewer directions*.

**The pure-OPD anomaly fits both stories.** Pure OPD λ=1 has the *highest* attn_qkv ΔW rank (ratio 1.056 — adds spectral dimensions to the base) AND the worst pass@1 (0.047). Reading: the trainer is exploring new attention dimensions (additive-rank update), but those dimensions are *misaligned* with reward, so adding them is harmful. The prune curves confirm this: reverting most of the pure-OPD moves *improves* performance (§6.1 finding §6.2 extended to p=95%, p@1 = 0.242 = 5× unpruned). The OPD geometry signature: "broad and rank-additive, but mistargeted."

**Caveat on the effrank measurement.** The batch-mode `delta_theta_snapshot.py` only stores the top-10-by-frob tensors per checkpoint (the others are dropped to keep the JSON small). For attn_qkv (always the dominant submodule by frob norm) this is fine — every arm has all its attn_qkv layers measured. For other categories (mlp, embed) the per-arm coverage is incomplete; numbers for those would need a second pass with `--save-per-tensor`. The attn_qkv tier-separation is the primary finding and it's complete.

### 6.4 Full per-tensor effective-rank pass — §6.3 was a top-10-by-frob artifact (2026-05-24)

**Setup.** The §6.3 result was based on only the top-10-by-frob tensors per checkpoint (a limitation of the batch-mode `delta_theta_snapshot.py`). To check whether the tier separation holds when *every* 2D weight matrix is included, I wrote `harness/effrank_all_tensors.py` and ran it across the 11 canonical checkpoints — measuring effective rank of ΔW for **every** weight: 48 attn_qkv (q/k/v × 16 layers), 16 attn_o, 32 mlp_in (gate + up × 16 layers), 16 mlp_down, 2 embed (tok_embed + lm_head). Per submodule, aggregate the mean ΔW eff-rank / mean base eff-rank ratio.

**Result — full per-tensor pass (corrects §6.3):**

| arm | tier (§6.1) | attn_qkv | attn_o | mlp_in | mlp_down | embed |
|---|---|---:|---:|---:|---:|---:|
| RL baseline | broader | 1.027 | 0.993 | 1.012 | 1.002 | 0.995 |
| clip λ=0.05 | broader | 1.001 | 0.975 | 1.005 | 0.996 | 0.997 |
| clip λ=0.10 | broader | 1.004 | 0.985 | 1.009 | 0.999 | 1.006 |
| clip λ=0.20 | broader | 1.029 | 1.002 | 1.014 | 1.004 | 1.002 |
| v2.1 unclipped λ=0.05 | broader | 1.002 | 0.988 | 1.004 | 0.995 | 1.006 |
| **broader-tier mean** |  | **1.013** | **0.989** | **1.009** | **0.999** | **1.001** |
| GRPO-v2 s42 | sharper | 1.093 | 1.045 | 1.023 | 1.013 | 0.946 |
| GRPO-v2 s43 | sharper | 1.058 | 1.011 | 1.014 | 1.004 | 0.989 |
| clip λ=0.50 | sharper | 1.063 | 1.041 | 1.021 | 1.011 | 0.980 |
| clip λ=0.85 | sharper | 1.070 | 1.043 | 1.021 | 1.010 | 0.969 |
| pure OPD λ=1 | sharper | 1.069 | 1.043 | 1.021 | 1.010 | 0.957 |
| v2.1 unclipped λ=1 | sharper | 1.043 | 1.016 | 1.001 | 0.989 | 0.978 |
| **sharper-tier mean** |  | **1.066** | **1.033** | **1.017** | **1.006** | **0.970** |

_Plot: `research/figs/dtheta/exp3_effrank_full.png`. JSON: `research/figs/dtheta/effrank_full/effrank_*.json`._

**Honest correction.** The §6.3 "broader-tier ΔW is LOWER rank than base (0.77-0.89)" claim is **wrong**. It was specific to the top-10-by-frob tensors per checkpoint — the most-changed attn_qkv layers per arm, which is a biased slice (the biggest changes show the most dramatic rank effects). Averaging across all 48 attn_qkv tensors, the broader-tier ratios cluster at **1.00-1.03**, not 0.77-0.89. The original §6.3 finding measured a real subset effect, but the *generalization* "broader tier = lower-rank update everywhere" was an artifact.

**Corrected geometric picture.** The tier separation in spectral measurements is real but *small and submodule-specific*:

- **attn_qkv (q/k/v projections):** broader 1.01, sharper 1.07 — sharper tier adds ~6% spectral capacity to base; broader tier preserves base rank. This is the only submodule with substantial tier separation.
- **attn_o (output projection):** broader 0.99, sharper 1.03 — same pattern, smaller magnitude.
- **mlp_in / mlp_down:** both tiers ~1.00-1.02 across the board. **No meaningful tier separation in MLP.**
- **embed (token embed + lm_head):** broader 1.00, sharper 0.97 — **the reversal**: sharper tier *compresses* embedding rank by 3% on average (GRPO-v2-s42 is at 0.946, the lowest of any arm anywhere), while broader tier preserves it. Pure OPD λ=1 has embed ratio 0.957 — the second-lowest. This is consistent with §6 noting that sharper-tier arms touch only 1-3% of embedding rows vs broader-tier's 11-13% (heatmap `exp3_dtheta_by_category.png`): touching fewer embedding rows correlates with reducing the embed spectrum.

**The actual geometric story across submodules.**

The sharper-tier vs broader-tier distinction is not a uniform "different rank everywhere" — it's a **submodule-specific reorganization**:

- **Sharper tier (GRPO, high-λ clipped, pure OPD)**: adds spectral capacity to attention (where new behavior is learned) while *narrowing* the embedding spectrum (concentrating output on fewer effective directions). "Reshape attention to compute new things, reshape embeddings to emit fewer things."
- **Broader tier (clipped low-λ, RL baseline, unclipped v2.1 low-λ)**: preserves rank everywhere. The update is value-shifting within the existing spectral basis at every submodule. "Adjust everything without changing the dimensions."

This is consistent with the §6.1+6.2 prune-degradation pattern: sharper tier's update is geometrically more *structural* (changes spectral character of multiple submodules in opposite directions) → small concentrated changes that compose into a graceful prune curve; broader tier's update is geometrically more *additive* (rank-preserving value shifts spread across many weights) → many small changes that fail catastrophically when the bottom 80-90% are pruned because the residual rank-N → rank-M cliff hits.

**Status of §6.3's interpretation.** The original §6.3 directional reading ("sharper-tier ΔW has *near-base* rank, broader-tier ΔW has *lower* rank") is reversed by the full pass: broader-tier is at-base, sharper-tier is *above*-base on attn and *below*-base on embed. The qualitative "two tiers have different geometries" claim still holds — just at a much smaller magnitude and with a more nuanced shape than the top-10 slice suggested. The pure-OPD interpretation ("rank-additive at attn, but mistargeted") survives the correction — pure OPD remains the highest attn_qkv ratio (1.069) and one of the lowest embed ratios (0.957).

### 6.5 Same-tokenizer SFT-from-rollouts control — a third regime (job `74735` + `74736`, 2026-05-25)

**Question.** Every arm in §6 / §6.1 / §6.2 / §6.4 has `α=1` (on-policy data) and varies `λ` (teacher dose) + clip. The remaining hole in the geometry story is the `(α=0, λ=1, π_T=δ_data)` corner of the meta-algorithm — *off-policy SFT on teacher rollouts*, the canonical SFT recipe. The literature's strong prediction is "RL sparse, SFT dense / redundant" (RL's Razor). Does the §6 "broader tier" arms (RL baseline, clipped low-λ) sit *between* RL and SFT on the dense↔sparse axis, with the SFT corner being even denser? Or does SFT-from-rollouts live in its own regime?

**Setup.** Train the **same 1B student** (`allenai/OLMo-2-0425-1B-SFT`, the init used everywhere in §6) on the **same teacher rollouts** that produced the §7.2 gsm-teacher: `rft_data/gsm_symbolic_from_OLMo-2-1124-7B-Instruct_seed4242.jsonl` (2550 verifier-accepted completions from `OLMo-2-1124-7B-Instruct`). Standard cross-entropy SFT, prompt tokens masked with `-100`, cosine LR + warmup, AdamW (β=0.9/0.95), peak LR 5e-6, 2 epochs × 160 optimizer steps/epoch = 320 total. Config: `harness/run_exp3_sft_student.sh` (job `74735`). Trained in 2.2 min on one H100; loss 0.96 → 0.41. Checkpoint: `harness/checkpoints/sft-student-rft-gsm-s42/`.

**§6 sparsity / §6.4 effrank — SFT control row added:**

| arm                          | tier (§6)   | top-1% mass | top-5% mass | changed > 1e-4 | p99 \|Δθ\| | attn_qkv ER | embed ER |
|------------------------------|-------------|---:|---:|---:|---:|---:|---:|
| rl_baseline_s42              | broader     | 0.573 | 0.918 | 0.093 | 1.28e-3 | 1.027 | 0.995 |
| grpo_v2_s42                  | sharper     | 0.632 | 0.943 | 0.059 | 3.83e-4 | 1.093 | 0.946 |
| clip1, λ=0.10, s42           | broader     | 0.577 | 0.921 | 0.094 | 1.47e-3 | 1.004 | 1.006 |
| clip1, λ=1.00 (pure OPD) s42 | sharper     | 0.611 | 0.931 | 0.089 | 7.44e-4 | 1.069 | 0.957 |
| **sft_student_s42 (new)**    | **third**   | **0.729** | **0.983** | **0.016** | **1.09e-4** | **1.075** | **0.846** |

**Per-submodule effrank (full §6.4 protocol, all 114 2D weights):** attn_qkv 1.075 / attn_o 1.014 / mlp_in 1.015 / mlp_down 0.997 / embed **0.846** _(plot: `figs/dtheta/exp3_dtheta_sft_control.png`; JSON: `figs/dtheta/dtheta_sft_student_s42.json` + `figs/dtheta/effrank_full/effrank_sft_student_s42.json`)._

**Reading — three structural anomalies.**

1. **SFT-from-rollouts is the SPARSEST update of any arm**, not the densest. Top-1% mass = 0.729 (vs sharper-tier 0.61-0.63, broader-tier 0.57-0.58); top-5% mass = 0.983 (vs sharper 0.93-0.94, broader 0.91-0.93); changed > 1e-4 = **1.6% of params** (vs the α=1 family's 6-10%); p99 |Δθ| = 1.09e-4 (3-13× *smaller* than every other arm). This **directly falsifies the literature's "RL sparse / SFT dense" prediction at this scale and training intensity** — SFT on verified rollouts touches fewer weights, with smaller magnitudes, and concentrates its mass into a tighter subset than any α=1 trainer in this sweep.
2. **SFT's spectral signature is a *more extreme* version of the sharper tier**, not a different family entirely. attn_qkv effrank ratio 1.075 (sharper tier mean 1.066; sharper tier max — pure OPD — 1.069); embed ratio **0.846**, the lowest of any arm (sharper tier mean 0.970, prior min 0.946). SFT compresses embed rank by **15%** — almost 5× the sharper tier's average compression — while *adding* attention rank, exactly the sharper-tier shape but turned up. mlp_in/down ratios sit at the broader tier (1.01 / 1.00, unchanged).
3. **No prune cliff, but a gradual sharper-tier-like decline** (T=0.6 prune-degradation, job `74736`, 10 levels, 64 prompts × 16 samples):

   | prune% (of 5.4% moved) | 0 | 10 | 30 | 50 | 60 | 70 | 80 | 85 | 90 | 95 |
   |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
   | p@1 | 0.377 | 0.382 | 0.387 | 0.394 | 0.397 | 0.396 | 0.356 | 0.325 | 0.251 | 0.093 |
   | p@16 | 0.703 | 0.656 | 0.703 | 0.688 | 0.688 | **0.750** | 0.672 | 0.703 | 0.594 | 0.453 |
   | tok-ent | 0.415 | 0.417 | 0.416 | 0.412 | 0.414 | 0.419 | 0.421 | 0.443 | 0.469 | 0.492 |

   The p@1 column is **flat to slightly upward** through p=70% (0.377 → 0.397), then declines gradually. At p=95% (top-5% of moves kept ≈ 0.27% of total params), p@1 retains 25% of unpruned (0.093 / 0.377) — comparable to GRPO-v2's 49% (0.266 / 0.540) and far above broader-tier arms' ~0.5% (rl/clip1 at p=95% ≈ 0.001-0.003). **No catastrophic cliff** in the p=[80%, 90%] window where broader-tier arms collapsed; the SFT curve looks like a damped sharper-tier curve, not a broader-tier curve.
4. **p@16 *peaks* at p=70%** (0.750 vs unpruned 0.703) and **token entropy rises with pruning** (0.415 → 0.492). Both are pure-OPD signatures: a small fraction of moves is mildly misaligned, and reverting them recovers slightly better pass@k. SFT's "essential subnetwork" is even smaller than its 5.4% moved fraction — most of the moves aren't load-bearing.

**The corrected three-regime picture.**

The (α=1) family produced two tiers in §6 / §6.1. SFT-from-rollouts is a third regime, *not on the same axis* as sharper-vs-broader:

- **SFT corner (α=0, λ=1, π_T=δ_data, off-policy):** sparsest of all, sharper-tier-shaped spectral update *amplified* (high attn rank, deepest embed compression), graceful prune decline, pruning slightly improves the model. Unpruned p@1 = 0.377 — lower than RL/GRPO, consistent with off-policy SFT being less capable than on-policy outcome RL at matched compute.
- **Sharper tier (GRPO, clip1 high-λ, pure OPD):** small concentrated essential subnetwork, attn-rank-adding + embed-compressing, gradual prune decline. **Pure OPD is a noisy/mistargeted version of GRPO's geometry**, not a separate regime — they share the same shape, just with pure OPD's moves misaligned with reward.
- **Broader tier (RL baseline, clipped low-λ):** larger essential subnetwork spread across more weights, rank-preserving everywhere, catastrophic cliff at p∈[80%, 90%]. This is the regime where the *outcome* branch and the *teacher* branch are both contributing.

So the literature's "RL sparse / SFT dense" picture **misses two things** at this scale: (a) at 320-step SFT on 2550 rollouts, SFT is *sparser* than 500-step on-policy RL, because the off-policy data isn't pulling the student toward a different attractor — only filling in a narrow mode; (b) the "denser" arms in this family are the **low-λ outcome+teacher blends**, not the SFT corner. The geometry tier reflects *how many forces the trainer is balancing*, not whether the gradient is dense or sparse in the literature's sense.

**What this changes about §8.3.** The post's geometry story — "OPD picks a point in update-geometry space, and the meta-algorithm chooses where" — gets a sharper version. The dimensions that vary across (α, λ, clip, π_T) are: (i) **how concentrated the essential subnetwork is** (SFT > sharper > broader, by static top-K mass), (ii) **how much each submodule's spectrum is reshaped** (SFT > sharper > broader, by attn rank addition and embed rank compression), and (iii) **whether prune degradation is graceful or cliffed** (sharper + SFT graceful, broader cliffed). These do *not* line up neatly with α or λ alone — they line up with whether the trainer is dominated by *one* signal (SFT: only teacher rollouts; sharper tier: only outcome OR teacher-as-prior) or balancing *two* (broader tier: outcome + small teacher).

Open Exp 3 follow-ups remaining:

- **Per-layer spectral analysis.** The §6.4 + §6.5 numbers average across layers; a per-layer breakdown might reveal which transformer layers carry the tier separation (predicted: middle layers).
- **Pass@k cross-task transfer of the SFT control.** §7.8 ran clipped low-λ vs GRPO on `simple_equations`. The SFT control's cross-task profile (does its higher token entropy + lower in-dist p@1 mean better generalisation?) is the natural complement.

---

## 7. Experiment 4 — The (α, λ) interior ([[meta-algorithm-alpha-lambda]] #6, [[expert-rl-plus-opd]] #11)

**Question.** The corners are SFT/RL/OPD/OPSD. What about the interior — `0<λ<1` (a teacher-KL term *on top of* outcome RL, DeepSeek-V4-style)? D2 (per-token KL × correctness) showed the SFT-teacher's per-token signal is content-correlated (`topq_correct_lift = 2.10`), and D3 (teacher eval) showed the same teacher actually solves `gsm_symbolic` at pass@1 ≈ 0.46 — so a small teacher term on top of GRPO *might* regularize toward content-aimed tokens. Or it might poison the verifier (the teacher's hedge-on-correct-tokens, D2 `kl_mean_on_correct = -0.16`).

**Design.** 8 arms = λ ∈ {0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0} × 2 seeds {42, 43}, all with α=1, π_T=`-7B-SFT`, outcome=`grpo`; held-constant otherwise vs Exp 1 (student `-1B-SFT`, `gsm_symbolic`, `prompts_per_step=8 × num_rollouts=8 × max_new=1024`, 500 steps, eval @ steps 250 and 500). Configs `harness/configs/exp4_lambda_interior.yaml` + launcher `harness/run_exp4_lambda.sh`. Loss math after the 2026-05-13 `_run_distill_loop` refactor: `loss = λ · L_teacher_REINFORCE + (1-λ) · L_clipped_GRPO` (proper clipped policy-ratio on the outcome branch, not REINFORCE on a blended advantage; per-component `loss/teacher_term` + `loss/outcome_term` logged separately). λ=0 baseline is Exp-1's `rl-baseline-s42` (the validator forbids λ=0 + `same_family`).

**v1 cancelled (SLURM `71174`, 1h).** All 8 arms stuck at reward ~0.50 (format-only); cause: `UnifiedTokenLoss` at 0<λ<1 was REINFORCE-on-blended-advantage, skipping the policy-ratio clip. Refactor introduced the (λ·teacher-REINFORCE + (1−λ)·clipped-GRPO) split at the loss level (see [[expert-rl-plus-opd]]); smokes at λ=0.1 jumped reward 0.07→0.55 in 2 steps, and λ=1.0 numerically reproduced the original pure-OPD path.

**Results — v2 cross-seed table (SLURM `71177` seed-42 7h36m / `71188` seed-43 7h18m; 16 runs total, held-out eval at T=0.6, 64 prompts × 16 samples).**

| λ | seed-42 @500 pass@1 | seed-43 @500 pass@1 | **mean pass@1** | seed-42 pass@16 | seed-43 pass@16 |
|---|---|---|---|---|---|
| 0 (Exp-1 GRPO) | 0.577 | n/a (1-seed) | — | 0.679 | — |
| 0.05 | 0.004 | 0.020 | 0.012 | 0.062 | 0.109 |
| 0.10 | 0.004 | 0.010 | 0.007 | 0.062 | 0.141 |
| **0.20** | **0.231** | 0.021 | **0.126** | **0.453** | 0.156 |
| **0.35** | 0.006 | **0.167** | **0.087** | 0.062 | **0.531** |
| 0.50 | 0.010 | 0.017 | 0.014 | 0.078 | 0.188 |
| 0.70 | 0.006 | 0.018 | 0.012 | 0.078 | 0.172 |
| 0.85 | 0.007 | 0.013 | 0.010 | 0.078 | 0.141 |
| 1.00 (pure OPD) | 0.044 | 0.004 | 0.024 | 0.281 | 0.062 |

_(per-arm json: `results/exp4_seed42_71177/<arm>.json` + `results/exp4_seed43_71188/<arm>.json`; reward-split per-step traces `harness/logs/exp4_lam*_seed{42,43}_71{177,188}.log`.)_

**Three headline findings.**

1. **Stochastic interior breakthrough in λ ∈ [0.20, 0.35], one arm per seed.** Each seed produces *exactly one* arm with substantially elevated pass@1 (0.17–0.23, ~5–10× the dead zone), and that arm lives in the λ ∈ {0.20, 0.35} band. Crucially, **the exact λ flips between seeds**: seed-42 wins at λ=0.20 (0.231) and dies at λ=0.35 (0.006); seed-43 inverts this (λ=0.20 → 0.021, λ=0.35 → 0.167). Across both seeds: λ=0.20 mean = 0.126, λ=0.35 mean = 0.087 — both ~4–9× higher than the average dead-zone arm (0.010–0.014). The interior peak is *real*, but its location is **winner-take-one and seed-stochastic** within a narrow mid-λ band.
2. **Dead zone outside [0.20, 0.35].** λ ∈ {0.05, 0.10, 0.50, 0.70, 0.85} produce pass@1 ≤ 0.020 across both seeds — no breakthroughs, in or out, at the small-λ end (where the GRPO branch dominates) or the large-λ end (where the teacher term dominates). The dead zone is *not* uniform between seeds either: seed-43 sits 2–5× higher than seed-42 across these arms (mean 0.011 vs 0.005), suggesting a baseline seed-luck effect on top of the breakthrough lottery.
3. **Pure OPD (λ=1) reproduces Exp 1's level.** v2 λ=1 mean = 0.024 (seed-42 0.044, seed-43 0.004), statistically consistent with Exp 1's 0.014 ± 0.002 (same teacher `-7B-SFT`, same seed pair averaged). The 71177 seed-42 high outlier (0.044) was a single-seed fluke, not a refactor artifact — and seed-42 in v2 happened to be unlucky at the corner *and* lucky at the interior, while seed-43 was the inverse. Per-seed variance dominates; **single-seed numbers in this regime are unreliable to ~3-10× factors**.

**The format/accuracy split (reward-decomp from §4 [TODO done 2026-05-14]) makes the failure mode explicit.** Per-step training traces now log `acc` + `fmt` alongside the combined `reward`: every arm saturates `fmt` to ~0.93–1.0 within ~50 steps (so the "reward ≈ 0.5 plateau" we saw in Exp 1 was literally `0·acc + 0.5·1.0·fmt`); the *accuracy* component is where the action is. Breakout arms reach `acc ≈ 0.04–0.07` in rollouts; dead-zone arms stay at `acc ≈ 0.01`. The eval pass@1 numbers track rollout `acc` qualitatively but with the expected 5–10× train-vs-held-out gap (e.g. λ=0.20 seed-43 hit rollout `acc=0.056` at step 194, but held-out pass@1 was 0.021 — partial in-domain learning that didn't generalize to held-out prompts).

**Plots.** *(to add — `results/exp4/`)*: per-λ `acc` trajectories per seed (overlaid; the breakthrough arms are visually obvious as the ones that escape the 0.01 floor in the second half); per-λ `loss/teacher_term` and `loss/outcome_term` traces (which branch dominates in the breakout arms — preliminary: the breakout arms show *both* terms moving, dead-zone arms show only the teacher branch moving).

### 7.1 Interpretation

The two seeds together force a sharper reading than either alone:

- **The on-policy state-space lock-in mechanism from §4.2 is *partially*, not fully, dispositive.** At λ=1.0 (pure OPD), reverse-KL on student trajectories really does fail to transfer the teacher's math capability (mean pass@1 = 0.024 vs teacher pass@1 = 0.46 — a 19× capability gap that the per-token signal is unable to close, even though the signal is content-correlated per D2). Lock-in fits. But the interior breakthroughs show the lock-in is *escapable* when the clipped-GRPO branch supplies enough outcome-aligned exploration to occasionally land the student's trajectories in regions where the teacher's local correction is actually useful. The mid-λ band is where "GRPO finds the math" and "teacher refines it" happen to coincide — *some of the time*.
- **The lottery character is the new finding.** The seed-dependence of the breakout's exact λ location says the interior is **not a smooth Pareto frontier**: it's a discrete event ("the student's rollouts found gsm_symbolic-shaped reasoning") triggered stochastically by exploration luck, with λ controlling not *whether* but *how much it sticks* once it happens. This explains the cross-seed flip cleanly: seed-42's specific rollout trajectory under λ=0.20 found the math; the same student under λ=0.35 didn't get the lucky early rollouts and stayed locked. Seed-43 inverts the luck. Neither seed's "loser" λ in the pair was *worse* than the dead zone — they just didn't trigger.
- **No interior λ matches pure GRPO.** GRPO baseline pass@1 = 0.577 vs the best v2 interior arm 0.231. The "expert RL + OPD beats pure RL" prediction from [[expert-rl-plus-opd]] (#11) is **not supported** at this scale/task/teacher: the teacher term costs more in dead-zone arms than it pays in breakthrough arms.

**Caveats.** (i) 2 seeds is still small — a 3rd seed would tell us whether the [0.20, 0.35] band is the right window or if breakthroughs sometimes hit λ=0.10 or λ=0.50. (ii) GRPO baseline (0.577) came from `_run_rl_loop`; the interior arms came from `_run_distill_loop`. The two share the same outcome objective at λ=0, but a clean compute-matched GRPO control through the distill path would rule out subtle code-path differences (see "next experiments" below). (iii) Single-task: same conclusion on `gsm_symbolic` doesn't necessarily generalize.

### 7.2 Positive control — does teacher task-specialization rescue OPD?

The §7.1 mechanism reading made a *testable* prediction: if **on-policy state-space lock-in** is the bottleneck — the student's rollouts don't visit the teacher's high-reward regions, so the teacher's per-token correction lands on off-path trajectories — then **task-specializing the teacher** so its distribution overlaps the student's reachable trajectories should unlock pass@1. The cleanest test: rejection-sample teacher solutions on `gsm_symbolic`, SFT the same-base 7B teacher on the kept ones, then re-run the Exp-4 sweep with that specialized teacher.

**Pipeline.** Three SLURM jobs end-to-end:

1. **RFT generation** (`71200`, 29 min on one H100; `harness/rft_generate.py`, `harness/run_rft_generate.sh`). Roll `allenai/OLMo-2-1124-7B-Instruct` on 1500 `gsm_symbolic` prompts × 4 samples at T=1.0; filter by the `reasoning_gym` verifier; save accepted `(prompt, templated_prompt, completion, answer)` to `rft_data/gsm_symbolic_from_OLMo-2-1124-7B-Instruct_seed4242.jsonl`. **2550/6000 accepted (42.5%)** — close to the teacher's T=0.6 pass@1 of 0.462 (D3). Disjoint dataset seed (4242) from training (42/43) and held-out eval (1_000_042/3).
2. **SFT** (`71201`, 14 min on one H100; `harness/train_sft_rft.py`, `harness/run_train_sft_rft.sh`). Causal-LM cross-entropy on the verified completions only (prompt tokens masked with `-100`); cosine LR with 5% warmup, AdamW (β=0.9/0.95), peak LR 5e-6, 2 epochs × 318 optimizer steps/epoch = 636 total. Loss 0.62 → 0.31 → 0.25 (clean trajectory, no instabilities). Output: `harness/checkpoints/teacher_7B-SFT-gsm/` (14.6 GB bf16 model — a **task-specialized same-base 7B teacher**, identical base as Exp 1's `-7B-SFT`, only the recipe differs).
3. **Exp-4 λ-sweep with the specialized teacher** (`71202`, 6h45m on itiger01 8×H100; `harness/configs/exp4_lambda_interior_gsm_teacher.yaml`, `harness/run_exp4_lambda_gsm_teacher.sh`). 8 λ values × seed 42, everything else held identical to v2.

**Cross-run results table (held-out @500, T=0.6, 64 prompts × 16 samples).** This combines Exp 4 v2 (off-shelf `-7B-SFT` teacher, 2 seeds) with the positive-control (specialized `-7B-SFT-gsm` teacher, **4 seeds at the two breakthrough λs**). The gsm-teacher seed-43 column is from job `71208`; seeds 44+45 at λ ∈ {0.05, 0.50} from job `71242`.

| λ | v2 s42 p@1 | v2 s43 p@1 | v2 mean | gsm s42 | gsm s43 | gsm s44 | gsm s45 | **gsm mean p@1 (n=4 at λ∈{0.05,0.50}; n=2 otherwise)** | **best seed p@1** |
|---|---|---|---|---|---|---|---|---|---|
| **0.05** | 0.004 | 0.020 | 0.012 | 0.340 | 0.059 | **0.443** | 0.024 | **0.217** (sd 0.207) | 0.443 |
| 0.10 | 0.004 | 0.010 | 0.007 | 0.015 | 0.014 | — | — | 0.015 | — |
| **0.20** | **0.231** | 0.021 | 0.126 | 0.019 | 0.015 | — | — | 0.017 | — |
| **0.35** | 0.006 | **0.167** | 0.087 | 0.033 | 0.016 | — | — | 0.025 | — |
| **0.50** | 0.010 | 0.017 | 0.014 | 0.118 | 0.040 | **0.275** | **0.246** | **0.170** (sd 0.110) | 0.275 |
| 0.70 | 0.006 | 0.018 | 0.012 | 0.006 | 0.013 | — | — | 0.010 | — |
| 0.85 | 0.007 | 0.013 | 0.010 | 0.021 | 0.013 | — | — | 0.017 | — |
| **1.00 (pure OPD)** | 0.044 | 0.004 | 0.024 | 0.011 | 0.005 | — | — | 0.008 | — |

_(per-arm json: `results/exp4_gsm_teacher_71202/<arm>.json` (seed-42) and `harness/logs/{exp4gsm_lam<λ>_seed43_71208, exp4ms_<i>_gsm-lam<λ>-s<s>_71242}.log` for the rest; checkpoints `harness/checkpoints/exp4gsm_lam<λ>_seed<s>/` and `gsm-lam<λ>-s<s>/`.)_

**The four-seed picture is bimodal, not gaussian.** At λ=0.05, p@1 across seeds {42, 43, 44, 45} = {0.340, 0.059, 0.443, 0.024} — two clear modes at ~0.39 and ~0.04, gap **Δ=0.35**, with no seed landing in between. λ=0.50 is gentler ({0.118, 0.040, 0.275, 0.246}) but still shows two seeds clearly above the dead-zone floor (s44, s45) and two close to it (s42, s43). The four-seed *mean* (0.217 at λ=0.05, 0.170 at λ=0.50) understates the **best-case** lift — the best seed at λ=0.05 reaches 0.443, and pass@16 in that seed is 0.734, very close to GRPO's pass@16. So the meta-knob does enable a high-performance regime; it just does not deterministically reach it. **§7.6 (new)** dissects the in-training trajectory to find that all 4 seeds collapse to ~0.01 acc around step 100-150 and only ~half recover — i.e. the bimodality is a recovery-vs-no-recovery split, not a never-escape-the-floor split. The 71249 intervention sweep (submitted 2026-05-16) tests three ways to make the recovery deterministic.

**Three findings that lock the mechanism.** _(Finding #2 is the single-seed reading; see the seed-43 update above for the multi-seed correction.)_

1. **Pure OPD (λ=1) is NOT rescued by teacher specialization.** With the off-shelf teacher: mean 0.024 (seeds 42/43). With the task-specialized teacher: 2-seed mean **0.008** (s42 0.011, s43 0.005) — *same dead zone*, even though the specialized teacher solves `gsm_symbolic` at pass@1 ≈ 0.46 and emits CoT-shaped solutions matching the verifier's format. **On-policy state-space lock-in is fundamental to pure reverse-KL** at this 1B/7B capacity gap — the student's rollouts never reach the teacher's math-correct regions of state space, and per-token corrections on the student's wrong trajectories can't compose into globally-correct multi-step reasoning, even when the per-token signal is from a math-competent teacher. The teacher-as-hedge mechanism from §4.2 was the wrong story; the correct one is **trajectory-mismatch + multi-step reasoning is fundamentally non-marginal**. This finding is robust across both seeds.
2. **The (0 < λ < 1) interior gains *two* breakthrough arms with the specialized teacher** — λ=0.05 and λ=0.50 are the only arms that lift above the dead-zone floor in any seed. The 4-seed picture is **bimodal**: at λ=0.05, p@1 ∈ {0.024, 0.059, 0.340, 0.443} splits cleanly into a low cluster ({0.024, 0.059}, mean 0.041) and a high cluster ({0.340, 0.443}, mean 0.392), gap **Δ=0.35**, no seed in between. The best seed reaches 0.443 p@1 / 0.734 p@16 — close to GRPO's pass@16 mean (0.786) at 2-3× higher token entropy. The 4-seed mean p@1 is **0.217 ± 0.207** at λ=0.05 and **0.170 ± 0.110** at λ=0.50, so the meta-knob enables a high-performance regime — but does not deterministically reach it. The bimodality is **not** a stochastic seed-lottery in the steady state: §7.6 shows that all 4 seeds collapse to ~0.01 acc around step 100-150 and only ~half *recover* by step 500, so the regime split is a recovery-vs-no-recovery split during training. The 71249 intervention sweep tests whether scheduling λ around the collapse window makes the recovery deterministic.
3. **The breakthrough λ shifts LEFT with a better teacher.** Off-shelf `-7B-SFT`: breakouts somewhere in [0.20, 0.35] depending on seed. Specialized `-7B-SFT-gsm`: breakouts at λ ∈ {0.05, 0.50}, with the strongest peak at λ=0.05 (95% GRPO + 5% teacher). When the teacher's distribution is task-aligned, a *much smaller* λ suffices — and conversely, when the teacher is task-misaligned, the GRPO branch has to dominate (~80% weight) to compensate. The synergy is **interactive, not additive**: neither GRPO alone (3-seed mean 0.661) nor specialized OPD alone (2-seed mean 0.008) explains the λ=0.05 result (4-seed mean 0.217, best-seed 0.443). This shift is robust across all four seeds.

**GRPO baseline is multi-seed (jobs `71242` + `71209`).** Three seeds each of pure GRPO (v1-path via `_run_rl_loop`) and GRPO-via-distill (v2-path via `_run_distill_loop` at λ=0.001):

| seed | v1 (rl-loop) p@1 | v2 (distill-loop) p@1 | Δ (v2−v1) |
|---|---|---|---|
| 42 | 0.577 | 0.646 | +0.069 |
| 43 | 0.678 | 0.687 | +0.009 |
| 44 | 0.729 | 0.727 | −0.002 |
| **mean** | **0.661** (sd 0.077) | **0.687** (sd 0.041) | **+0.026** |

**The v1-vs-v2 code-path question is resolved**: the two paths are statistically indistinguishable. The seed-42 Δ=+0.069 that prompted the A3 control was a single-seed fluke; seeds 43+44 give ≤1 percentage-point gaps. Both paths produce the *same* training, so the v2.1 interior arms can be compared directly against the v1 baseline. **The seed-42 GRPO pass@1 of 0.577 turns out to be a low outlier**; the multi-seed mean is 0.661 (v1) / 0.687 (v2), 8-11 percentage points higher. This **changes the headline ratio**: best v2.1 λ=0.05 (0.443, s44) / GRPO mean (0.687) = **64%**, not the 59% single-seed claim. The four-seed mean ratio is **31%** (0.217 / 0.687) — the "expert-RL + OPD at λ=0.05 ≈ GRPO" reading from §7.2 first edit does not survive a properly-calibrated baseline.

**The mechanism, sharpened.** Putting Exp 1 + v2 + the positive control together: the failure of pure reverse-KL OPD at 1B/7B on `gsm_symbolic` is **not** about teacher recipe (D3: teachers are math-competent), **not** about per-token signal localization (D2: signal is content-correlated, top_correct_lift 2.10), and **not** about teacher task-alignment (this section: a math-specialized teacher doesn't help at λ=1). The failure mode is the *combination* of three things, none of which OPD can fix on its own: (a) the student's exploration distribution doesn't visit math-correct regions of state space without an outcome signal pushing it there; (b) per-token KL corrections on globally-wrong trajectories don't compose into globally-correct multi-step reasoning; (c) at the 1B/7B capacity gap, the student can't represent the teacher's joint multi-step distribution even when it can approximately match the per-token marginals on its own samples. **Expert-RL+OPD escapes this** because GRPO's outcome signal pushes the student into reward-aligned regions of state space *first*, and *there* the teacher's per-token signal becomes useful as a content-aimed prior. A *task-aligned* teacher amplifies this — small λ suffices.

### 7.4 Pass@k and cross-task generalisation

Two complementary follow-ups on the seed-42 gsm-teacher checkpoints test whether v2.1's higher token entropy converts to actual coverage gains (in-distribution wide-k pass@k) or to better generalisation under distribution shift (cross-task pass@k):

- **A2 (job `71210`):** wide pass@k on `gsm_symbolic`, T ∈ {0.6, 1.0}, k ∈ {1, 2, 4, 8, 16, 32, 64}, 128 prompts × 64 samples per arm. Launcher `harness/run_passk_v21_eval.sh`, outputs `results/passk_v21_gsm_symbolic_71210/<arm>.json`.
- **A4 (job `71211`):** identical recipe on `simple_equations` — a **cross-task** `reasoning_gym` probe never seen during training. Tests whether v2.1's diversity buys distribution-shift robustness. Outputs `results/passk_v21_genshift_simple_eq/<arm>.json`.

**A2 in-distribution pass@k (`gsm_symbolic`, 128 × 64, T=1.0).** RL baseline = pure GRPO seed 42 (Exp-1 lineage); v2.1 arms are seed 42, λ shown.

| arm | T | pass@1 | pass@4 | pass@16 | pass@64 | token_entropy |
|---|---|---|---|---|---|---|
| RL-baseline (GRPO s42) | 0.6 | **0.573** | **0.631** | **0.679** | **0.711** | 0.41 |
| RL-baseline (GRPO s42) | 1.0 | **0.548** | **0.643** | **0.715** | **0.766** | 1.01 |
| v2.1 λ=0.05 | 0.6 | 0.415 | 0.562 | 0.651 | 0.688 | 2.66 |
| v2.1 λ=0.05 | 1.0 | 0.332 | 0.508 | 0.624 | 0.695 | 3.55 |
| v2.1 λ=0.35 | 0.6 | 0.040 | 0.131 | 0.310 | 0.492 | 3.20 |
| v2.1 λ=0.50 | 0.6 | 0.155 | 0.328 | 0.485 | 0.586 | 2.59 |
| v2.1 λ=0.50 | 1.0 | 0.089 | 0.229 | 0.407 | 0.562 | 1.48 |
| v2.1 λ=1.00 (pure OPD) | 0.6 | 0.009 | 0.032 | 0.095 | 0.188 | 0.85 |
| v2.1 λ=1.00 (pure OPD) | 1.0 | 0.008 | 0.031 | 0.098 | 0.211 | 1.45 |

_(Full 8-row tables — all 8 λ × 2 T — in `results/passk_v21_gsm_symbolic_71210/`.)_

**Reading.** **GRPO dominates v2.1 at every k and every T on in-distribution `gsm_symbolic`.** At T=1.0 pass@64, GRPO reaches 0.766 vs v2.1 λ=0.05's 0.695, a 0.071 gap. The "v2.1 nearly catches GRPO at pass@16" headline from §7.2 (0.656 vs 0.679 at T=0.6, k=16, 64×16 eval) does *not* survive a wider eval (128×64, k=64): GRPO's slope from k=1 to k=64 is steeper than v2.1's despite v2.1's 3-7× higher token entropy. **The high token entropy is not coverage entropy.** v2.1's λ=0.05 entropy distributes mass over solution paths that are *individually* lower-quality but not *collectively* more diverse-and-correct, so larger k does not close the gap. This contradicts the entropy/coverage prediction from [[entropy-collapse-opd-vs-rl]] for this specific (1B-student, 7B-gsm-teacher, gsm_symbolic) regime — and is the cleanest piece of evidence we have that high token entropy ≠ output diversity ≠ pass@k coverage. ([[pass-at-k-vs-pass-at-1]] is materially refined.)

**A4 cross-task generalisation (`simple_equations`, 128 × 64).**

| arm | T | pass@1 | pass@4 | pass@16 | pass@64 |
|---|---|---|---|---|---|
| RL-baseline (GRPO s42) | 0.6 | **0.125** | **0.201** | **0.298** | **0.430** |
| RL-baseline (GRPO s42) | 1.0 | **0.112** | **0.211** | **0.334** | **0.492** |
| v2.1 λ=0.05 | 0.6 | 0.016 | 0.057 | 0.144 | 0.258 |
| v2.1 λ=0.05 | 1.0 | 0.011 | 0.041 | 0.128 | 0.258 |
| v2.1 λ=0.35 | 0.6 | 0.009 | 0.032 | 0.103 | 0.227 |
| v2.1 λ=0.50 | 0.6 | 0.008 | 0.028 | 0.092 | 0.219 |
| v2.1 λ=1.00 (pure OPD) | 0.6 | 0.006 | 0.022 | 0.073 | 0.164 |

**Reading.** **GRPO transfers ~2× better than v2.1 at every k and every T**, including the v2.1-favourable wide-k pass@64 corner (0.492 vs 0.258). The *ranking* between v2.1 arms transfers (λ=0.05 still the strongest interior arm, pure OPD still worst), so the on-task structure does generalise, but the *level* is everywhere worse than GRPO. The seed-42 in-distribution dominance pattern *survives* cross-task — but with a much shrunk multiplicative win for GRPO. Combined with A2, the strong version of the "expert-RL + OPD escapes the OPD failure mode and gains diversity" claim from §7.2 is refuted: v2.1's *training-eval* pass@1 lift is real and the (α=1, 0<λ<1) interior strictly dominates pure OPD, but the lift does **not** carry the diversity/coverage/generalisation advantages we hypothesised.

The combined Exp-4 finding is thus narrower than §7.2 read it: **the (α=1, 0<λ<1) interior with a task-aligned teacher is a non-trivial improvement on pure reverse-KL OPD at this capacity gap, but it does not yet compete with vanilla GRPO on any axis we measured.** The open question (§7.6 below) is whether the *mechanism* — what KL-signal localisation enables which arms to escape the dead-zone — is itself useful, even when the resulting policy is not.

### 7.5 GRPO baseline is multi-seed (job `71242`)

3 seeds each of pure GRPO via `_run_rl_loop` (v1) and GRPO-via-distill at λ=0.001 (v2-path):

| seed | v1 p@1 | v2 p@1 | Δ (v2−v1) | v1 p@16 | v2 p@16 |
|---|---|---|---|---|---|
| 42 | 0.577 | 0.646 | +0.069 | 0.679 | 0.781 |
| 43 | 0.678 | 0.687 | +0.009 | 0.703 | 0.766 |
| 44 | 0.729 | 0.727 | −0.002 | 0.781 | 0.812 |
| **mean** | **0.661** (sd 0.077) | **0.687** (sd 0.041) | **+0.026** | **0.721** | **0.786** |

The two code paths are statistically indistinguishable (mean Δ = +0.026 < 0.05; seed-42's Δ=+0.069 was a fluke). **The seed-42 GRPO baseline of 0.577 was a low outlier**, not the modal GRPO performance — the multi-seed mean is 0.661–0.687, 8-11 percentage points higher. This recalibrates every "X% of GRPO" claim in §7.2 / §7.4:

- best v2.1 λ=0.05 (s44 = 0.443) / GRPO mean (0.687) = **64%** of GRPO pass@1
- 4-seed mean v2.1 λ=0.05 (0.217) / GRPO mean (0.687) = **31%** of GRPO pass@1
- best v2.1 λ=0.05 pass@16 (s44 = 0.734) / GRPO pass@16 mean (0.786) = **93%**

So the §7.2-first-edit "λ=0.05 reaches 59% of GRPO" reading becomes "the best seed reaches 64%, the mean reaches 31%, but pass@16 in the lucky regime is 93% of GRPO." The bimodality is the headline now, not the mean — §7.6 explains why.

### 7.6 The OPD collapse-recovery mechanism

The 4-seed pattern at λ=0.05 is not gaussian seed noise. Plotting in-training rollout accuracy step-by-step across all 4 seeds, each with the gsm-teacher at λ=0.05:

```
            step  1     25    50    100   150   200   250   300   400   500   final p@1
seed 42    0.000 0.381 0.411 0.164 0.009 0.009 0.010 0.010 0.010 0.317      0.340  high
seed 43    0.000 0.381 0.397 0.025 0.010 0.010 0.009 0.009 0.010 0.087      0.059  low
seed 44    0.001 0.520 0.381 0.195 0.010 0.010 0.010 0.009 0.040 0.349      0.443  high
seed 45    0.000 0.474 0.489 0.010 0.010 0.010 0.009 0.010 0.021 0.102      0.024  low
```

**All four seeds follow the same three-phase trajectory**: (i) rapid climb to ~0.4-0.5 acc by step 25-50, (ii) catastrophic collapse to ~0.01 acc between steps 100 and 150, (iii) a flat dead-zone for ~250 steps, then (iv) a partial recovery in the final 100 steps that lands the run in one of two basins. The high-final seeds (42, 44) recover to 0.32-0.35; the low-final seeds (43, 45) only partially recover to 0.09-0.10. Held-out pass@1 at step 500 closely tracks the final rollout acc.

**GRPO does not show this pattern.** Plotting the same trajectory for v1 GRPO at seeds 43, 44 and v2 GRPO-via-distill at seeds 42, 43, 44:

```
            step  1     25    50    100   150   200   250   300   400   500
v1-s43     0.000 0.381 0.504 0.350 0.412 0.504 0.658 0.844 0.752 0.629
v1-s44     0.001 0.459 0.412 0.472 0.613 0.334 0.489 0.381 0.505 0.459
v2-s42     0.000 0.505 0.412 0.582 0.845 0.366 0.536 0.752 0.737 0.660
v2-s43     0.000 0.350 0.551 0.426 0.412 0.504 0.721 0.892 0.845 0.551
v2-s44     0.001 0.567 0.443 0.613 0.566 0.442 0.412 0.690 0.722 0.504
```

GRPO acc fluctuates but never drops below ~0.3 once it has climbed past it; there is no flat dead-zone. **So the collapse is a property of the teacher-reverse-KL signal**, not of the data, the outcome reward, or the optimizer. The collapse window happens *after* the model has reached non-trivial reward, and *during* the period when the teacher's per-token KL is pulling the student's distribution toward marginals that don't compose into correct global trajectories. The recovery happens *despite* the teacher signal, driven by the (1-λ) GRPO branch occasionally hitting a correct rollout in the dead-zone and the resulting advantage outweighing the destabilising KL push for a few steps.

**Why this matters.** The §7.2 "stochastic breakthrough" framing was incomplete: the meta-knob does **not** find a different basin per seed; it finds the *same* collapse-recovery trajectory in every seed, and the seed only determines whether the recovery is full or partial. This makes the mechanism testable: any intervention that prevents or shortens the collapse (or stabilises the recovery) should rescue the low-final seeds. Three such interventions are queued in job `71249`:

| intervention | hypothesis | implementation |
|---|---|---|
| **step_off @ 100** — λ=0.05 for steps 1–99, then λ=0 | teacher signal is *helpful as warm-up*, *harmful during collapse* | `lam_schedule=step_off`, `lam_step=100` |
| **step_on @ 200** — λ=0 for steps 1–199, then λ=0.05 | teacher signal is *useless as warm-up*, *helpful as post-RL refinement* | `lam_schedule=step_on`, `lam_step=200` |
| **linear_anneal 50→300** — λ ramps 0.05 → 0 across collapse window | teacher signal must be *smoothly removed* across the collapse — compromise between step_off and step_on | `lam_schedule=linear_anneal`, `lam_step=50`, `lam_step_end=300` |
| **per_token_kl_clip = 1.0** — clamp |teacher KL| at 1.0 per token | collapse is driven by a *few outlier tokens* — clipping disarms them | `per_token_kl_clip=1.0`, schedule unchanged |

Each intervention is run at one **known-good** seed (s42, final p@1 0.340) and one **known-bad** seed (s45, final p@1 0.024). If any intervention rescues s45 to >0.30 *and* preserves s42, that intervention identifies the mechanism. If none rescue s45, the collapse is structural and requires changing the trainer (e.g. ref-policy refresh cadence, KL-to-base penalty, or a different outcome reward weighting).

Implementation lives in `harness.unified_trainer.current_lam` (the schedule helper) and `harness/configs/exp4_lambda_interior_gsm_teacher.yaml` (the schedule fields parse from `--set lam_schedule=...`). The W&B trace also logs `meta/lam_eff` per step so the schedule is visible in the metric plots. Launcher `harness/run_exp4_collapse_recovery.sh`; eval JSONs land under `results/exp4_recovery_71249/<arm>.json`.

#### 7.6.1 Outcome (job `71249`, 2026-05-17)

| arm | s42 p@1 | s42 p@16 | s45 p@1 | s45 p@16 | s42 entropy | s45 entropy |
|---|---|---|---|---|---|---|
| baseline (const λ=0.05, no clip) | 0.340 | 0.656 | 0.024 | 0.188 | 3.05 | 1.79 |
| **per_token_kl_clip = 1.0** | **0.660** | **0.734** | **0.633** | **0.719** | 0.39 | 0.75 |
| step_off @ 100 | 0.008 | 0.109 | 0.007 | 0.016 | 2.72 | 0.04 |
| step_on @ 200 | 0.285 | 0.547 | 0.055 | 0.156 | 2.75 | 5.27 |
| linear_anneal 50→300 | 0.030 | 0.031 | 0.016 | 0.016 | 0.15 | 0.20 |

**Three things settled.** **(1)** `per_token_kl_clip = 1.0` is the mechanism — it rescues s45 by **26×** (0.024 → 0.633), boosts s42 by **2×** (0.340 → 0.660), and **eliminates the bimodality** (the two seeds finish within 0.027 of each other instead of 0.316 apart). **(2)** All three λ-scheduling interventions FAIL to rescue s45. `step_off @ 100` is the most striking failure: turning off the teacher at step 100 *causes* permanent collapse at the same step rather than preventing one — so the teacher signal is not a "harmful pull during the collapse window" that can be removed, it is the thing the (1-λ) GRPO branch needs to anchor against during early training. `linear_anneal` partially works but doesn't recover either seed above 0.03. `step_on @ 200` partly recovers s42 to 0.285 (below the no-intervention baseline of 0.340) and recovers s45 transiently around step 150-300 but collapses by step 500. **(3)** The mechanism is **outlier per-token KL signals**, not the *timing* of the teacher signal. Heavy-tail per-token reverse-KL drives token-level REINFORCE updates with effective gradients much larger than the GRPO advantage; clipping these contributions disarms the destabilising tokens while preserving the dense low-magnitude teacher signal that the recovery needed.

**In-training trajectory contrast.** All four kl_clip seeds (s42, s45 from 71249) maintain rollout acc in [0.20, 0.85] throughout training — there is no collapse at any step. By contrast every λ-schedule arm collapses to ~0.01 at step 100-150 or step 150-200, exactly tracking the unclipped baseline (§7.6).

**What this changes upstream.**
- **§7.2's "OPD failure mode is on-policy state-space lock-in + non-marginal multi-step composition"** is now *partly* wrong as written: at λ=0.05, the failure was an unclipped-REINFORCE artifact, not state-coverage. **For pure OPD (λ=1), `71250` shows clipping does not rescue the corner** (§7.7), so state-space lock-in remains the right mechanism there.
- **§7.6's "stochastic basin recovery" framing** is now refined: there is no *stochastic* basin. The unclipped trainer has a single attractor (the dead zone), and which seeds escape is determined by whether the few destabilising tokens happen to fall on positive-reward rollouts. Clipping makes the escape deterministic.
- **§7.4's "high token entropy is not coverage entropy"** *survives* and gets cleaner: the kl_clip arm achieves comparable pass@k to GRPO with token entropy 0.39-0.75 — *not* the 2-3× entropy the unclipped arm had, which we now know was largely noise from outlier KL pushes. The actual lesson is "the per-token KL signal is informative; reverse-KL is a fine objective; but it must be clipped, just like every other on-policy gradient method."

### 7.7 Clip sweep and clipped λ-interior

**Status update (post-71242, 71249, 71250, 71271 done; 71270 cancelled).** Headline: **per-token clipping turns expert-RL + OPD into a GRPO-level recipe, but only for a bounded low-λ band.** At λ=0.05, `per_token_kl_clip ∈ [0.5, 2.0]` matches vanilla GRPO (clip mean 0.66 ≈ GRPO mean 0.67). The new seed-42 λ sweep at clip=1.0 pushes the result further: λ=0.10 reaches **0.709 p@1 / 0.812 p@16**, above the 3-seed GRPO-v2 means (**0.687 / 0.786**). But clipping does **not** make the whole interior a flat plateau: λ≥0.50 remains far below GRPO, and pure OPD (λ=1) remains structurally dead even with clipping.

**71250 readout (2026-05-17):**

| arm | s42 p@1 | s45 p@1 | 2-seed mean | s42 / s45 entropy |
|---|---|---|---|---|
| clip=0.5, λ=0.05 | 0.660 | 0.663 | **0.662** | 0.29 / 1.02 |
| clip=1.0, λ=0.05 (71249) | 0.660 | 0.633 | 0.647 | 0.39 / 0.75 |
| clip=2.0, λ=0.05 | 0.651 | 0.692 | **0.672** | 1.01 / 0.67 |
| clip=5.0, λ=0.05 | 0.588 | 0.533 | 0.560 | 1.14 / 1.19 |
| **clip=1.0, λ=1.0 (pure OPD)** | **0.029** | **0.010** | **0.020** | 0.72 / 0.79 |
| _GRPO v1/v2 reference (3 seeds)_ | _0.661 / 0.687_ |  |  | _0.6 / 0.7_ |

**Reading.** **(Q1)** The optimal clip threshold is a plateau in [0.5, 2.0]: mean p@1 of 0.66 ± 0.02 across all three thresholds, ~matching GRPO. clip=5.0 (the original heavy-tail cutoff) starts to leak bimodality back (s42-s45 gap = 0.055) and drops mean p@1 by ~10 percentage points. The win is robust — kl_clip is not a magic threshold; any reasonable cutoff (≤ ~2.0) disarms the outliers enough. **(Q2)** **Per-token KL clipping does NOT rescue pure OPD.** λ=1.0 + clip=1.0 stays in the dead zone (mean 0.020 vs. unclipped 0.008 — both indistinguishable from 0). This **confirms §7.2's "on-policy state-space lock-in" mechanism at the λ=1 corner**: at pure reverse-KL there's no outcome signal to anchor the model in reward-aligned trajectories, the teacher's per-token signal lands on off-path tokens, and clipping the outliers doesn't help because the *mean* signal is also misaligned. The (1-λ) GRPO branch is essential to making the teacher signal useful, regardless of clipping.

**71271 readout (2026-05-18/19; seed 42, clip=1.0):**

| λ | pass@1 @500 | pass@16 @500 | token entropy | reading |
|---|---:|---:|---:|---|
| 0.05 | 0.638 | 0.734 | 0.34 | GRPO-level p@1; lower p@16 than GRPO-v2 mean |
| **0.10** | **0.709** | **0.812** | 0.98 | **current best single-seed point; beats GRPO-v2 mean** |
| 0.20 | 0.601 | 0.734 | 2.21 | GRPO-level p@1, but weaker p@16 |
| 0.35 | 0.441 | 0.562 | 1.59 | transition band |
| 0.50 | 0.099 | 0.203 | 0.93 | not rescued |
| 0.70 | 0.101 | 0.234 | 0.72 | not rescued |
| 0.85 | 0.029 | 0.141 | 0.78 | dead-zone / near pure-OPD behavior |

_(summary parsed from logs: `results/exp4_clip_lambda_71271/summary.json`; logs: `harness/logs/exp4cl_{0..6}_clip1.0-lam*-s42_71271.log`; checkpoints: `harness/checkpoints/clip1.0-lam*-s42-71271/`.)_

**Reading.** The two candidate stories split cleanly. The **flat-plateau hypothesis is false**: clipping does not rescue the whole `0<λ<1` interior. Instead, clipping opens a **good low-λ band** at λ∈[0.05, 0.20], with λ=0.10 the cleanest "expert-RL + OPD beats GRPO" datapoint so far. λ=0.35 is a transition point (0.441 p@1, about 2× the unclipped seed-42 λ=0.35 readout but below GRPO). λ≥0.50 remains weak (0.03-0.10 p@1), so the teacher term is helpful only as a bounded auxiliary signal; too much reverse-KL still pulls the student away from reward-aligned trajectories even when outliers are clipped.

**71395 replication (2026-05-19/20; clip=1.0, seeds 43-45, combined with 71271 seed 42):**

| λ | s42 p@1 | s43 p@1 | s44 p@1 | s45 p@1 | 4-seed mean p@1 | 4-seed mean p@16 |
|---|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.638 | 0.611 | 0.752 | 0.633 | 0.659 | 0.746 |
| **0.10** | **0.709** | **0.643** | **0.770** | **0.648** | **0.693** | **0.801** |
| 0.20 | 0.601 | 0.606 | 0.750 | 0.596 | 0.638 | 0.785 |
| 0.35 | 0.441 | 0.523 | 0.490 | 0.405 | 0.465 | 0.598 |

_(summary: `results/exp4_clip_lambda_71395/summary.json`; logs: `harness/logs/exp4clr_*_71395_*.log`; checkpoints: `harness/checkpoints/clip1.0-lam*-s*-rep-71395-*/` plus seed-42 checkpoints from `71271`.)_

**Replication read.** The low-λ clipped band is real, not just seed-42 luck. λ=0.10 is the best 4-seed point and narrowly beats the GRPO-v2 3-seed reference on both p@1 (0.693 vs 0.687) and p@16 (0.801 vs 0.786). The margin is small, so the honest headline is **GRPO-level to slightly above GRPO in-distribution**, not a large win. λ=0.35 remains a transition/falloff point, confirming that the useful teacher dose is bounded.

**The corrected mechanism story.** Putting Exp 1 + §7.2 + §7.6 + 71250 + 71271 together:

1. **Pure OPD (λ=1) is genuinely dead** at this capacity gap. State-space lock-in is real and clipping doesn't fix it.
2. **The low-λ interior** has a genuine collapse-recovery instability driven by **outlier per-token KL pushes** that pull the student off the GRPO-anchored trajectory. The bimodal low-λ picture from §7.2 was an artifact of those outlier pushes: in some seeds they accidentally landed on positive-advantage rollouts and the model recovered; in other seeds they pulled the model off the manifold permanently.
3. **Per-token clipping at |kl|≤1.0 prevents the collapse in the low-λ band**, and the resulting `(α=1, λ∈[0.05,0.20], clip=1.0)` policy matches or slightly beats GRPO in-distribution over 4 seeds. The §7.2 "low-λ breakthrough is seed-fragile" claim is **superseded** for the clipped trainer.
4. **High λ remains structurally harmful.** `71271` shows that clipping is not enough once the teacher term becomes too large: λ≥0.50 dies to 0.03-0.10 p@1, and λ=1.0 dies even harder. The outcome branch is not merely present-or-absent; it must dominate the update.

### 7.8 Cross-task pass@k for the clipped low-λ band (job `71574`, 2026-05-20/21)

**Question.** §7.7 left the in-distribution λ=0.10 ≳ GRPO claim with one open hedge: does the small edge survive when the held-out evaluation is a *different* task family? §7.4 had already found that GRPO transferred ~2× better than v2.1 on `simple_equations`, but that was on the *unclipped* trainer. The kl_clip fix in §7.7 might have closed that gap (because outlier KL pushes were also a candidate explanation for the off-task tax).

**Setup.** Re-evaluate the 4-seed `clip=1.0, λ∈{0.05, 0.10, 0.20}` checkpoints (trained on `gsm_symbolic`) plus the 3-seed GRPO-v2 reference on **`simple_equations`** — same eval protocol as §7.4 (128 prompts × 64 samples, k∈{1,2,4,8,16,32,64}, T∈{0.6, 1.0}). Job `71574`, 15 evals, 8×H100, ran 2026-05-20 15:33 → 2026-05-21 03:00.

**Results — T=0.6:**

| arm | seeds | p@1 | p@2 | p@4 | p@8 | p@16 | p@32 | p@64 | tok-ent | distinct-2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **grpo_v2** | 3 | **0.203** | **0.271** | **0.342** | **0.415** | **0.485** | 0.549 | 0.607 | 1.50 | 0.053 |
| clip1, λ=0.05 | 4 | 0.146 | 0.207 | 0.272 | 0.343 | 0.419 | 0.501 | 0.580 | 6.24 | 0.063 |
| clip1, **λ=0.10** | 4 | 0.107 | 0.171 | 0.253 | 0.351 | 0.459 | **0.565** | **0.666** | 8.11 | 0.107 |
| clip1, λ=0.20 | 4 | 0.072 | 0.123 | 0.194 | 0.282 | 0.385 | 0.500 | 0.611 | 7.88 | 0.141 |

**Results — T=1.0:**

| arm | seeds | p@1 | p@2 | p@4 | p@8 | p@16 | p@32 | p@64 | tok-ent | distinct-2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **grpo_v2** | 3 | **0.171** | **0.247** | **0.335** | **0.431** | **0.534** | **0.633** | **0.721** | 2.32 | 0.087 |
| clip1, λ=0.05 | 4 | 0.107 | 0.169 | 0.244 | 0.328 | 0.422 | 0.530 | 0.652 | 8.61 | 0.110 |
| clip1, λ=0.10 | 4 | 0.061 | 0.106 | 0.171 | 0.256 | 0.360 | 0.475 | 0.594 | 9.76 | 0.187 |
| clip1, λ=0.20 | 4 | 0.027 | 0.050 | 0.087 | 0.142 | 0.216 | 0.314 | 0.428 | 10.15 | 0.279 |

_(results: `results/passk_clip_lowlam_simple_equations_71574/{grpo_v2,clip1_lam{005,010,020}}_s4{2,3,4,5}.json`; launcher: `harness/run_clip_lowlam_cross_task_eval.sh`; .out: `harness/logs/clip-xtask-71574.out`.)_

**Reading.**

1. **The in-distribution λ=0.10 > GRPO edge does NOT survive off-task at low k.** On `simple_equations`, **GRPO wins p@1 at both temperatures by ~2×** over the best clipped-λ arm (T=0.6: 0.203 vs 0.146; T=1.0: 0.171 vs 0.107). The §7.4 generalisation gap — "GRPO transfers ~2× better than v2.1 on `simple_equations`" — is **still present under clipping**. kl_clip fixed the in-distribution collapse but did not buy back the off-task tax. So the §7.4 gap was *not* (or at least not only) a side-effect of unclipped outlier KL pushes; it survives at clip=1.0.

2. **But the OPD-blend arms catch up — and at T=0.6 cross over GRPO — at moderate-to-high k.** At T=0.6, **clip1, λ=0.10 overtakes GRPO at p@32 (0.565 vs 0.549) and beats it cleanly at p@64 (0.666 vs 0.607)**. This is the **first clean pass@k crossover** in the writeup: at k≥32 sampled answers per prompt, the higher-entropy OPD-blend students beat the sharpened GRPO baseline cross-task. It's the prediction from [[entropy-collapse-opd-vs-rl]] / [[pass-at-k-vs-pass-at-1]] made concrete — and at T=1.0 GRPO stays ahead (its T=1.0 spread is wider), so the crossover is a T=0.6 phenomenon, not a temperature-trivial one.

3. **Token entropy is preserved off-task; reward-alignment is not.** The clipped-λ arms keep their ~4–6× higher token entropy on the held-out task (~8.1–10.2 vs GRPO's 1.5–2.3) and their ~2× higher distinct-2 trigram diversity. So the diversity is durable across distributions, but it doesn't convert into low-k accuracy when the prompts shift family.

4. **The "high entropy is not coverage entropy" framing from §7.4 needs to be split.** Off-task, **high token entropy partly *is* coverage entropy** — it's what enables the p@64 crossover. The thing it isn't is *aligned* coverage: GRPO's narrow but reward-aligned distribution dominates at p@1–p@8 even cross-task; the OPD-blend's wider distribution only pays off once the verifier gets enough draws to pick out the correct ones. This is exactly the ProRL pattern (entropy bonus buys back pass@k at a small pass@1 cost).

**What this means for the headline story.** The honest summary across §7.7 + §7.8 is:

- **In-distribution `gsm_symbolic`:** clipped (α=1, λ=0.10) is *narrowly* the best point, beating GRPO-v2 by 0.6pp on p@1 and 1.5pp on p@16 (4-seed vs 3-seed).
- **Out-of-distribution `simple_equations`:** GRPO wins p@1–p@16 by a wide margin (~2×); clipped (α=1, λ=0.10) overtakes at **p@32–p@64** at T=0.6.
- **The (α, λ) interior is not strictly dominating GRPO** — it's *crossover-dominating*: better at high-k, worse at low-k, off-task.

This is a cleaner result than the in-distribution narrow edge would suggest in isolation. It maps the kl_clip low-λ band to a specific use case: *settings where you can spend test-time compute (best-of-32 or higher), care about diversity / coverage, and accept worse first-sample accuracy*. For greedy-decode or low-k settings, vanilla GRPO is still the better recipe at this scale.

Open follow-up:

- **kl_signal localisation — DONE 2026-05-21.** Extracted the per-step kl_signal/{p50, p90, p99, abs_max, heavy_tail_frac} traces from the offline W&B binaries for the 71208 (unclipped, s43 sweep) and 71395 (clipped, s43/44/45 × λ∈{0.05,0.10,0.20,0.35}) runs. Plot lives at `research/figs/exp4_kl_signal_mechanism.png`; written up as **§8.2**. The cleaner contrast turned out to be *clipped vs unclipped* (since seed-43 was universally collapsed in the unclipped sweep): unclipped p99 rises to 2.1-2.9 over training, clipped p99 sits at 0.4-1.0 just under the |kl|=1 cap; the recovery from the step-100 collapse only happens under clipping. The §7.7 mechanism story is now quantitative.
- **Generalization-gap direct test — DONE 2026-05-22, see §7.9 below.** Slurm `72040` (~5h wall on 7×H100). 3 ckpts × 3 eval seeds = 9 evals on `gsm_symbolic`. Eval-seed variance is small (sd 0.006-0.014, ~1-4% relative); the training-seed bimodality of v2.1 unclipped is the load-bearing dimension (s42 0.435 vs s43 0.085 — robust across all 3 eval seeds). Clip=1.0 dominates both unclipped seeds (0.685 ± 0.014).
- **Exp 3 (sparse-vs-dense weight delta) — DONE 2026-05-22, see §6 above.** `harness/delta_theta_snapshot.py` (sparsity proxies + by-category aggregation); 11-checkpoint batch in `figs/dtheta/`. **All α=1 arms — RL baseline, GRPO, clipped λ-interior, pure OPD λ=1 — are in the same sparse-update regime** (top-1% mass 0.57-0.63, top-5% 0.91-0.94, only 6-10% of weights moved by >1e-4). The "SFT-dense for OPD" prediction is falsified at this scale. Pruning-degradation and effective-rank remain follow-ups.
- **Exp 5 (faithful per-vocab clipped KL)** remains deferred ([[faithful-per-vocab-kl-clip]]). 71249's kl_clip arm is a partial test of #5 at the per-token level.

### 7.9 Generalization-gap robustness across eval seed pools (job `72040`, 2026-05-22)

**Question.** The v2.1-unclipped λ=0.05 sweep was bimodal in §7.2: across 4 training seeds, p@1 ranged 0.012–0.443 (mean 0.217 ± 0.207). The §7.4 reading and the headline number in §7.7 use a single eval seed pool (`--eval-seed 1000000`). Is the 0.415 p@1 "bimodal-breakthrough" at v2.1 s42 a robust property of the trained checkpoint, or an artifact of which 128 held-out prompts were drawn? And is the 5× gap between s42 (breakthrough) and s43 (collapsed) a real training-seed effect, or eval-seed noise?

**Setup.** Re-evaluate **three checkpoints** — v2.1 unclipped λ=0.05 seed 42 (breakthrough), v2.1 unclipped λ=0.05 seed 43 (collapsed), clipped λ=0.05 seed 42 (recovered) — across **three eval seed pools** (`--eval-seed ∈ {1_000_000, 2_000_000, 3_000_000}`), same protocol as §7.4/§7.8 (128 prompts × 64 samples, T∈{0.6, 1.0}, k∈{1,2,4,8,16,32,64}). 9 evals total on 7×H100 (1 GPU still held by another user), ~5h wall. Launcher: `harness/run_gen_gap_eval_seeds.sh`; outputs: `results/gengap_eval_seeds_72040/*.json`.

**Results (T=0.6):**

| ckpt | eval_seed | p@1 | p@2 | p@4 | p@8 | p@16 | p@32 | p@64 | tok-ent |
|------|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2.1, λ=0.05, s42 | 1M | 0.415 | 0.500 | 0.563 | 0.609 | 0.650 | 0.679 | 0.703 | 2.76 |
| v2.1, λ=0.05, s42 | 2M | 0.443 | 0.514 | 0.587 | 0.633 | 0.675 | 0.717 | 0.766 | 2.88 |
| v2.1, λ=0.05, s42 | 3M | 0.447 | 0.524 | 0.598 | 0.643 | 0.683 | 0.722 | 0.750 | 2.76 |
| **v2.1, λ=0.05, s42** | **mean (sd)** | **0.435 (0.014)** | 0.513 | 0.582 | 0.628 | **0.669** | 0.706 | 0.740 | 2.80 |
| v2.1, λ=0.05, s43 | 1M | 0.094 | 0.146 | 0.197 | 0.247 | 0.286 | 0.328 | 0.375 | 2.52 |
| v2.1, λ=0.05, s43 | 2M | 0.080 | 0.131 | 0.183 | 0.241 | 0.301 | 0.367 | 0.422 | 2.58 |
| v2.1, λ=0.05, s43 | 3M | 0.083 | 0.135 | 0.197 | 0.260 | 0.315 | 0.371 | 0.422 | 2.62 |
| **v2.1, λ=0.05, s43** | **mean (sd)** | **0.085 (0.006)** | 0.137 | 0.192 | 0.249 | **0.301** | 0.355 | 0.406 | 2.57 |
| clip1, λ=0.05, s42 | 1M | 0.680 | 0.726 | 0.762 | 0.785 | 0.800 | 0.815 | 0.836 | 0.68 |
| clip1, λ=0.05, s42 | 2M | 0.671 | 0.700 | 0.717 | 0.731 | 0.742 | 0.754 | 0.766 | 0.41 |
| clip1, λ=0.05, s42 | 3M | 0.705 | 0.726 | 0.741 | 0.752 | 0.762 | 0.770 | 0.781 | 0.63 |
| **clip1, λ=0.05, s42** | **mean (sd)** | **0.685 (0.014)** | 0.717 | 0.740 | 0.756 | **0.768** | 0.780 | 0.794 | 0.57 |

T=1.0 results are in `results/gengap_eval_seeds_72040/*.json`; pattern is the same with proportionally lower p@1 and higher p@k for k≥16 (the §7.4 / §7.8 entropy/coverage trade reappears).

**Reading.**

1. **Eval-seed variance is small in absolute terms — 1-4% relative.** Across 3 eval seed pools, p@1 has sd = 0.014 (v2.1 s42), 0.006 (v2.1 s43), and 0.014 (clip1 s42). On a mean of 0.085-0.685 this is ~1.4-7% relative noise. So every numerical p@1 / p@k value in §7.4–§7.8 carries ~±0.01-0.02 noise from the choice of held-out seed pool. None of those numbers is eval-seed-specific.

2. **The training-seed bimodality of v2.1 unclipped is the load-bearing variance, not eval-seed noise.** v2.1 s42 reliably gives p@1 ≈ 0.435 across all 3 eval seed pools; v2.1 s43 reliably gives p@1 ≈ 0.085. That's a **5× gap between two training seeds of the same trainer at the same hyperparameters, far larger than the within-seed eval-seed noise**. This confirms the §7.2 "bimodal at low-λ" claim from a new angle: the bimodality is *not* an eval-seed-specific quirk; it is a real instability in the unclipped trainer that resolves to one of two outcomes per training seed.

3. **Clip=1.0 dominates both unclipped seeds.** clip1 s42 (recovered) reaches 0.685 p@1 — about 1.6× the v2.1 s42 breakthrough and 8× the v2.1 s43 collapse. The clipped trainer is not just "better than the dead-zone seeds"; it also beats the lucky breakthrough seed of the unclipped trainer by ~0.25 p@1 (0.685 vs 0.435). So the §7.7 read holds: clip=1.0 is the right knob, and the unclipped bimodality is not a useful operating point even when it lands on the "good" side.

4. **The §7.4 framing of "5-10× train-vs-held-out gap" is *not* what this experiment tested.** The framing question from §7's open follow-up was whether the held-out pass@1 might just be unlucky compared to training pass@1. What we actually measure here is: held-out pass@1 is *stable* across 3 redrawn seed pools, so it's a real measurement of the model, not a quirk of which 128 prompts were drawn. The 5-10× gap from §7.4 (if it referred to training-time `reward/accuracy` vs held-out pass@1) is a separate metric-definition question — `reward/accuracy` includes format reward and is averaged over self-rollouts during training, while pass@1 is verifier-only on held-out prompts. The actual gen-gap from "training distribution" to "held-out distribution within the same task" is **small** for clip1 s42 (training reward/accuracy plateaued at ~0.69 in §7.7, held-out p@1 here is 0.685) and **also small** for v2.1 s42 (similar order of magnitude). The remaining `simple_equations` cross-task gap from §7.8 is real and large, but that's cross-task, not redrawn-seed.

**What this means for the §7 mechanism.** The follow-up confirms two things and disconfirms one:

- ✓ **§7.2 bimodality is real**, not eval-seed-luck.
- ✓ **§7.7 clip=1.0 escape from bimodality is real and robust**, beating the breakthrough unclipped seed across all redrawn eval seed pools.
- ✗ **The "5-10× generalization gap" framing in the original follow-up does not stand up to direct measurement** — within `gsm_symbolic`, held-out p@1 is essentially equal to converged training reward/accuracy. The gap was a metric-definition artifact, not a held-out distribution-shift fact.

---

## 7.10 Experiment 5 — PRMs as teachers (answer-conditioned OPSD) ([[prms-as-teachers]] #13)

**Question.** §7.2 established the "pure OPD (λ=1) is dead" mechanism as *on-policy state-space lock-in* — the student's rollouts don't visit the teacher's high-reward regions, so per-token KL corrections land on globally-wrong trajectories. §7.7 confirmed: clipping doesn't rescue λ=1 either. The [[prms-as-teachers]] proposal asks whether **conditioning the teacher on privileged info** (here: the ground-truth answer prepended to the user message; canonical OPSD per Zhao et al. 2026) changes this — does an answer-conditioned teacher's per-token signal land on *answer-aimed* trajectories that the student can actually reach? More generally: does the teacher *interface* (logit teacher vs OPSD vs PRM-as-teacher) shift the (α, λ, clip) results from §7?

**Design.** Same trainer machinery as §7 (`_run_distill_loop`, clip=1.0, seed 42, 500 steps, 8 prompts × 8 rollouts × 1024 max_new on `gsm_symbolic`) but with `teacher = PrivilegedInfoTeacher(kind="self", model_name="OLMo-2-1124-7B-SFT", condition_on="answer", frozen_at_init=True)`. The teacher is the **same 7B-SFT used as §7's logit teacher**, but its context appends `"\n\nHint: the final answer is \`{answer}\`. Use it to guide your reasoning."` to the user message before scoring the student's tokens. Three arms: **λ ∈ {0.10, 0.50, 1.0}** — match §7.7's clipped low-λ winner, the §7.7 dead-middle, and the §7 dead corner respectively.

Implementation (new this experiment): `harness/teachers.py::PrivilegedInfoTeacher.token_logprobs` (batched chat-template prompt construction with hint appended, left-padded concat with student completion, single teacher forward, scatter back to the original action-mask positions). Trainer change: rollout-time caching of teacher logprobs in `Experience.teacher_logprobs` (added a field to `policy_gradients.buffer.Experience`), used at training time to skip the per-microbatch teacher forward — a strict improvement (frozen teacher → deterministic, no recomputation cost) that also threads `entries` through the buffer for entries-aware teachers. Helper `harness/unified_trainer.py::_teacher_needs_entries` gates this behavior so existing `SameFamilyTeacher` runs are unchanged. Config + launcher: `harness/configs/exp5_opsd_lambda.yaml`, `harness/run_exp5_opsd_lambda.sh`; SLURM `75338` (3 arms × ~7h on itiger01 3×H100, completed 2026-05-26).

**Results — held-out eval @ step 500, T=0.6, 64 prompts × 16 samples** (this section's runs labelled "OPSD"; §7.7 logit-teacher reference is from job 71250 (λ=1) + 71271 (λ=0.10, 0.50), same protocol, seed 42):

| λ | logit-teacher p@1 (§7.7) | logit-teacher p@16 | **OPSD p@1** | **OPSD p@16** | Δ p@1 | tok-ent (OPSD) |
|---|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.709 | 0.812 | **0.665** | **0.797** | **−0.044** | 1.16 |
| 0.50 | 0.099 | 0.203 | **0.238** | **0.312** | **+0.139** (2.4×) | 1.54 |
| **1.00 (pure OPD)** | **0.029** | **0.141** | **0.188** | **0.312** | **+0.159** (6.5×) | 0.70 |

In-training-eval @ step 250 (mid-training): OPSD λ=0.10 = 0.595 / 0.828; λ=0.50 = 0.273 / 0.406; λ=1.0 = 0.193 / 0.391. The pure-OPD λ=1 arm is *already* above 0.19 p@1 by step 250 — well clear of the §7.7 dead zone (which stays at ~0.01-0.03 across all 500 steps).

_(per-arm logs: `harness/logs/exp5_*_opsd-clip1.0-lam*-s42_75338.log`; checkpoints: `harness/checkpoints/opsd-clip1.0-lam{0.10,0.50,1.0}-s42-75338/`.)_

**Three findings.**

1. **Pure OPD (λ=1) is meaningfully rescued by answer-conditioning** — 0.029 → **0.188 p@1**, a **6.5× lift** from the §7.2/§7.7 dead corner. The teacher's per-token reverse-KL signal, when conditioned on the answer, lands on trajectories the student can actually reach AND that move toward the verifier's correctness boundary. This **partially reverses §7.2's "on-policy state-space lock-in is fundamental at the 1B/7B capacity gap"** reading: lock-in is real, but the **teacher's effective state coverage** is what makes it lethal. An answer-conditioned teacher's distribution is shifted toward the student's reachable answer-shaped trajectories, and that's enough to give the reverse-KL signal traction without an outcome anchor. Pure OPD is still well below GRPO (mean 0.687 from §7.5), so the answer-conditioned teacher alone isn't enough to match the verifier-only signal — but the **dead-corner narrative no longer survives**.
2. **Mid-λ (0.50) flips from dead to alive** — 0.099 → 0.238 (2.4×). The §7.7 mid-λ dead band was an artifact of the logit teacher's per-token signal being content-misaligned at high teacher weight (cf. §5 taxonomy: pure logit-OPD puts 67% of |KL| mass on uncertain tokens). With answer-conditioning, the same λ=0.50 dose carries usable signal. **The optimal λ shifts left under OPSD** — where the logit teacher needs the (1-λ) GRPO branch to dominate (λ=0.10 wins at 90% GRPO weight), OPSD lets the teacher branch carry more weight (λ=0.50 = 50/50 still works).
3. **Low-λ (0.10) is slightly worse under OPSD** — 0.709 → 0.665 (−0.044). At small teacher weight, the answer-conditioned signal competes with rather than complements the GRPO outcome signal — both are pulling toward the correct answer, so 10% of the teacher's pull is redundant overhead (and possibly slightly distracts the gradient on student-reachable-but-wrong trajectories). The privileged-info conditioning *helps where the logit teacher fails* and *hurts slightly where the logit teacher already works*. This is the cleanest evidence we have that **the teacher interface and the (α, λ) blend interact** — picking the right teacher changes the right λ.

**Mechanism — first attempt + revision after the taxonomy test (§5-style diagnostic, job `77371`, 2026-05-26).**

The first cut of this section predicted: "OPSD-trained students should have their KL mass shift away from the uncertain bucket and toward content, because the answer-conditioned teacher's distribution is content-aligned (knows where the reasoning ends)." The §5 taxonomy was re-run on the 3 OPSD checkpoints (same diagnostic protocol as §5 — 64 prompts × 4 samples, T=0.6, eval_seed=2000000, **LOGIT** 7B-SFT teacher for the diagnostic so the numbers are apples-to-apples comparable with §5's logit-trained arms).

**OPSD-trained students vs §5's logit-trained at matched λ (LOGIT teacher used for diagnosis in both cases):**

| arm | acc (diag) | kl_mean | kl_p99 | format mass | uncertain mass | wrong_conf mass | content mass |
|---|---:|---:|---:|---:|---:|---:|---:|
| **§5: clip1 λ=0.10 (logit-trained)** | 0.793 | -0.406 | 1.17 | 0.173 | **0.445** | 0.067 | **0.315** |
| **§7.10: OPSD λ=0.10 (this expt)** | 0.641 | -0.510 | 1.24 | 0.051 | **0.814** | 0.013 | **0.121** |
| **§5: clip1 λ=1.0 (logit-trained, dead)** | 0.047 | -0.370 | 0.77 | 0.153 | **0.669** | 0.012 | **0.166** |
| **§7.10: OPSD λ=1.0 (this expt, alive)** | 0.301 | -0.086 | 0.73 | 0.073 | **0.818** | 0.001 | **0.108** |

Matched-λ deltas (OPSD − logit, both vs the same logit diagnostic teacher):
| λ | Δformat | **Δuncertain** | Δwrong_conf | **Δcontent** |
|---|---:|---:|---:|---:|
| 0.10 | −0.122 | **+0.370** | −0.054 | **−0.194** |
| 1.00 | −0.080 | **+0.150** | −0.012 | **−0.059** |

**The prediction was wrong, and the way it was wrong is informative.** OPSD-trained students have *more* uncertain-bucket KL mass and *less* content-bucket mass than the matched-λ logit-trained arms — not the predicted shift. Yet the headline rescue (λ=1 from 0.029 → 0.188 p@1) is real. So the mechanism that's actually doing the work is not "OPSD pulls the teacher's signal into the content bucket"; it's something else. Three observations re-anchor the story:

1. **OPSD-trained students are HIGHER-entropy than logit-trained students at matched λ.** The "uncertain" bucket in §5 is defined by per-token *student* entropy > 1.0 nats. OPSD λ=0.10 puts 81% of mass there (vs logit λ=0.10's 45%); OPSD λ=1.0 puts 82% there (vs logit λ=1.0's 67%). The OPSD trainer produces a more diffuse student, not a more peaked one. This is opposite of what reverse-KL OPD is "supposed" to do — but consistent with the answer-conditioned teacher itself having more spread (the teacher knows the answer but stays neutral on *how* to reason there, so its distribution stays diffuse).
2. **OPSD-trained students' disagreement with the LOGIT teacher is larger at uncertain positions.** `kl_mean` is more negative for OPSD-trained students (-0.51 vs -0.41 at λ=0.10) — they're further from the logit teacher's distribution. This is expected: OPSD pulled them toward the answer-conditioned teacher, which differs from the logit teacher most on the positions where the answer-knowledge changes the distribution.
3. **The rescue at λ=1 is NOT via the §5 "content concentration" story.** OPSD λ=1's mass is 11% content (vs logit λ=1's 17% — slightly *less* content concentration). The rescue must come from somewhere other than per-token-KL mass landing on content tokens. Two candidates: (a) the answer-conditioned teacher's per-token signal at *uncertain* positions is qualitatively different from the logit teacher's (its hedge is over answer-aimed continuations, not arbitrary continuations) — the bucketing is the wrong unit of analysis; (b) the OPSD trainer is doing something more like *exploration shaping* than *direct correction* — the student's higher entropy is an artifact of being trained to match a teacher that emits diverse answer-aimed paths.

**Revised mechanism for the §7.10 rescue.** The §7.2 / §7.7 "pure OPD is dead because of on-policy state-space lock-in" reading still survives at λ=1, but the rescue mechanism is **not** "OPSD shifts KL mass to content" (falsified by the taxonomy). The actual story has to involve: the answer-conditioned teacher's distribution is *substantively different from the logit teacher's* (mean Δ −0.02 ± 0.35 in the smoke-test means individual positions differ by up to ~3 nats), and that difference is information-rich about answer-aimed continuations *at the uncertain positions* (where the logit teacher hedges over irrelevant alternatives and the answer-conditioned teacher hedges over answer-relevant ones). So §5's bucket labels are decorrelated from the per-token signal quality once the teacher's conditioning changes — the same uncertain-bucket token can carry very different signal under different teacher contexts. The §5 / §7.7 mechanism (clip removes uncertain-bucket noise) was correct in the logit-teacher regime; under OPSD, uncertain-bucket signal is no longer noise.

This is a **mechanism revision, not retraction**: the §7.10 rescue is real (0.029 → 0.188 at λ=1, 0.099 → 0.238 at λ=0.50) — and **now multi-seed confirmed** (λ=1 mean 0.252 ± 0.032, λ=0.50 mean 0.269 ± 0.055 across seeds 42–45; see the §7.10 follow-up below), substantially above the §7 dead-corner baseline. But the proposal-stage prediction "answer-conditioning shifts mass to content" is not what the taxonomy shows. The right framing is "answer-conditioning changes what the uncertain-bucket signal *means*, not where the mass lives."

**Caveats.** (i) ~~Single seed (42 only)~~ **RESOLVED by the multi-seed follow-up above** (job `77369`, seeds 43–45): the s42 numbers are stable (λ=1.0: 0.252 ± 0.032; λ=0.50: 0.269 ± 0.055; λ=0.10: 0.696 ± 0.085) and unimodal — the §7.6/§7.7 unclipped seed-bimodality does **not** appear under clip + answer-conditioning. (ii) The "answer in the prompt" formulation is heavy-handed; the [[prms-as-teachers]] proposal's variants (b) (a small PRM scoring prefixes) and (c) (PRM-reweighted OPSD per-token KL) would test more realistic settings where the answer isn't free. (iii) The teacher's chat-template was reused with the hint appended to the user message; tokenization edge cases (BPE boundary at the appended hint) might bias the per-token log-probs slightly. The smoke-test (`harness/smoke_opsd.py`, mean Δ(OPSD - plain teacher) = −0.02 ± 0.35 over 295 action tokens on a 4-prompt sample) confirms the signal is non-trivial and well-behaved. (iv) Cross-task transfer (§7.8) not yet measured for OPSD; if the answer-conditioning bakes in answer-shape priors that don't transfer, OPSD's in-dist edge may disappear off-task. (v) The taxonomy ran with the LOGIT teacher for diagnosis (so §5's numbers are matched-comparable). A parallel taxonomy with the ANSWER-CONDITIONED teacher as the diagnostic π_T would test the "uncertain signal is different in kind" hypothesis above directly.

**§7.10 follow-up — multi-seed (job `77369`, seeds 43/44/45, completed 2026-05-26).** The three-seed array ran the same 3 λ arms (clip=1.0, 500 steps, answer-conditioned `PrivilegedInfoTeacher`) under the §7.10 protocol; all 9 runs exited `rc=0` on itiger01 (3×H100). Held-out eval @ step 500, T=0.6, 64 prompts × 16 samples (per-arm logs: `harness/logs/exp5ms_{0,1,2}_opsd-clip1.0-lam{0.10,0.50,1.0}-s{43,44,45}-ms_77369_*.log`; checkpoints: `harness/checkpoints/opsd-clip1.0-lam*-s{43,44,45}-ms-77369-*/`):

| λ | s42 p@1 (§7.10) | s43 | s44 | s45 | **s43–45 p@1 mean ± sd** | **4-seed p@1** | 4-seed p@16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.665 | 0.628 | 0.815 | 0.645 | **0.696 ± 0.085** | **0.688** | 0.797 |
| 0.50 | 0.238 | 0.230 | 0.347 | 0.229 | **0.269 ± 0.055** | **0.261** | 0.398 |
| **1.00 (pure OPD)** | 0.188 | 0.224 | 0.236 | 0.297 | **0.252 ± 0.032** | **0.236** | 0.446 |

**The single-seed headline holds, and the dead corner stays dead.** Every arm's multi-seed mean lands at or slightly *above* its s42 value — there is no regression-to-the-mean back toward the §7 dead corner. Pure OPD (λ=1) is robustly **0.252 ± 0.032 p@1** across seeds 43–45 (0.236 over all four seeds), an ~8× lift over the §7.2/§7.7 logit-teacher dead corner (0.029) that survives multi-seed. Mid-λ (0.50) confirms alive at **0.269 ± 0.055**, and low-λ (0.10) remains the winner at **0.696 ± 0.085**. The arm ordering λ=0.10 ≫ λ=0.50 ≳ λ=1.0 is identical on all four seeds.

**No bimodality under clip + answer-conditioning — caveat (i) resolved.** The concern was that §7.6/§7.7's unclipped low-λ seed-bimodality might mean the s42 numbers were a lucky draw. It isn't: the per-seed spread is *tight* and unimodal — λ=1.0 has sd 0.032 (range 0.073 across 3 seeds), λ=0.50 sd 0.055, with no seed falling into a "dead" mode. The one bit of spread is λ=0.10's s44 high draw (0.815), but all three seeds sit firmly in the winning band (0.628–0.815), not split. Whatever drove the unclipped bimodality, the (clip=1.0 + answer-conditioned teacher) combination does not exhibit it. The 0.188/0.238 s42 numbers were representative, not lucky.

---

## 7.11 Experiment 6 — PRM-reweighted OPSD (self-referential answer-info-gain) ([[prms-as-teachers]] #13, variant c)

**Question.** §8.4 names the frontier object the post asks for — *dense AND (more-)unbiased AND on-policy, natively* — and points at the untested [[prms-as-teachers]] variant (c) as the natural next attempt: instead of **bluntly clipping** the outlier per-token KL pushes (§8.2's `per_token_kl_clip`, which simply *bounds* the heavy tail), **reweight** the per-token OPSD reverse-KL by a per-token *process importance* `w_t`, redistributing the KL mass off the uncertain/style tokens and onto the causally-important content/pivot tokens (§5 taxonomy: pure OPD dumps ~67% of |KL| mass on uncertain tokens, only ~17% on content). The headline test ([[prms-as-teachers]] line 45) is sharp: **does an importance reweight make OPSD safe *without* the clip?** If learned importance can replace blunt clipping, the no-clip-but-reweighted arm should recover the clipped baseline — a strictly better, less ad-hoc stabilizer, and a step toward the unbiasedness leg.

**Design.** The PRM here is **self-referential — no separately-trained PRM** (the proposal's line-52 open question, the user-chosen variant): the importance is the *answer-information-gain* of each token under the *same* 7B-SFT teacher,
`g_t = log π_T^answer(y_t) − log π_T^no-answer(y_t)`,
the answer-conditioned forward minus a matched no-answer forward — large on tokens the ground-truth answer disambiguates (the pivots/content), ≈0 on stylistic tokens. It is mapped to a **mass-preserving softmax** weight `w_t` (mean 1 over action tokens, temperature 1, uncapped), then `a_teacher_t = w_t · clip(log π_T^answer − log π_θ)`. Everything else is identical to Exp 5 (§7.10): student `OLMo-2-0425-1B-SFT` ← teacher `OLMo-2-1124-7B-SFT`+answer, `gsm_symbolic`, α=1, **λ=1** (pure OPSD; reweighting is wired at λ=1), 500 steps, 8 prompts × 8 rollouts × 1024 max_new, seed 42. **Four arms**, the launcher overriding only `per_token_kl_clip` and `prm_reweight`:

- **A** `clip=1.0, rw=off` — the OPSD baseline (≡ §7.10 λ=1, s42).
- **B** `clip=null, rw=off` — does unclipped OPSD collapse? (§7.7/§8.2 say yes.)
- **C** `clip=null, rw=on` — **the conjecture: reweighting *replaces* the blunt clip.**
- **D** `clip=1.0, rw=on` — do reweight + clip stack?

Implementation (all new this experiment): `harness/config.py` (`prm_reweight`/`prm_source`/`prm_weight_fn`/`prm_temperature`/`prm_weight_ceiling` + validators requiring teacher kind=self+answer, λ=1; `opsd_prm` recipe); `harness/teachers.py` (`PrivilegedInfoTeacher._conditioned_logprobs(hint_fn)` refactor + `answer_info_gain()` — the no-answer forward reuses the answer forward's machinery with an empty hint); `harness/distill_losses.py` (`prm_importance_weights()`, mass-preserving softmax/linear; `reverse_kl_distill_advantage(..., prm_weights=)`); `harness/unified_trainer.py` (rollout-time compute+cache into `Experience.prm_weights`, loss-time reweight, `prm/weight_{mean,max,p99}` W&B diagnostics). Config `harness/configs/exp6_prm_reweighted_opsd.yaml`; launcher `harness/run_exp6_prm_reweighted_opsd.sh`; smokes `harness/smoke_prm_reweight.py` + `harness/run_smoke_prm.sh` (GPU smoke job `111266` passed first). SLURM `111363` (4 arms on 4×H100 itiger01, completed 2026-06-07, 9h28m); results `results/exp6_prm_reweighted_opsd_111363/eval_<arm>_s42.json`, per-arm logs `harness/logs/exp6_{0,1,2,3}_*_111363.log`.

**Results — held-out eval @ step 500, T=0.6, 64 prompts × 16 samples** (same protocol as §7.10; grad-norm column = steady-state range over the final 50 steps from the per-arm logs):

| arm | clip | reweight | **p@1** | p@16 | tok-ent | grad_norm (final 50) | verdict |
|---|---|---|---:|---:|---:|---|---|
| **A** baseline | 1.0 | — | **0.204** | 0.391 | 0.76 | ~1.3–2.7 | survives (≡ §7.10 λ=1 s42, 0.188) |
| **B** noclip | null | — | **0.007** | 0.078 | 0.67 | ~6–10 | collapsed (§7.7/§8.2) |
| **C** noclip+rw | null | ✓ | **0.006** | 0.031 | 0.88 | ~9–**45** | **collapsed harder** |
| **D** clip+rw | 1.0 | ✓ | **0.157** | 0.328 | 0.69 | ~1.7–6.8 | survives, but **below A** |

(In-training eval @ step 500 agrees on ordering and magnitude: A 0.188 / B 0.007 / C 0.006 / D 0.135.)

**The conjecture is falsified — and instructively.**

1. **Reweighting does NOT replace the clip; the headline test answers *no*.** Arm C (no-clip + reweight) lands at **0.006 p@1**, statistically identical to plain no-clip B (0.007) and squarely in the §7.6 dead zone. Learned (here, self-referential) per-token importance is *not* a substitute for the blunt outlier bound. The clipped baseline A (0.204) and the clip+reweight arm D (0.157) survive; **both surviving arms have the clip, both collapsing arms lack it** — clip is the single load-bearing bit, exactly as §8.2 found, and the reweight does not move it.

2. **Reweighting *amplifies* the §8.2 instability — it sharpens the tail the clip exists to bound.** C does not merely fail to help; it collapses *harder* than B: pass@16 0.031 vs 0.078, and grad norms inflated **3–6× at matched steps** (step-2 grad 122 vs B's 20; final-50 range to 45 vs B's ~10; loss runs to −6.4 vs B's −1.5). The mechanism is now obvious and is the crux of the negative result: **the clip and the reweight are opposing operations on the per-token KL distribution.** §8.2 showed the clip works by *bounding* the heavy tail of per-token pushes (p99 pinned below the cap). A mass-preserving softmax reweight does the opposite — it *concentrates* mass onto a few high-`g_t` tokens, making the per-token push on exactly those tokens *larger*. Without a bound, that sharpened tail drives the §7.6 collapse faster and deeper (both no-clip arms peak at acc ~0.26–0.29 by step ~6, then C is in the dead zone by step 20). So reweighting cannot stand in for clipping *by construction*: clipping removes outliers, reweighting manufactures them.

3. **Reweighting doesn't even help stacked on the clip — it slightly hurts.** D (clip + reweight) = 0.157 < A (clip only) = 0.204. Even with the tail bounded so the run is stable, redistributing KL mass toward the answer-info-gain tokens *degrades* a working recipe. This mirrors §7.10's finding-3 precisely — the answer-conditioned/privileged signal **helps where the plain teacher fails and hurts where it already works**. Here the clip already makes OPSD work (A=0.204), so the importance reweight is redundant overhead that distorts the gradient rather than focusing it. The `g_t` signal is informative about *where the teacher's distribution depends on the answer*, but that is not the same as *where the per-token correction is most useful on a student-reachable trajectory* — and on the stable (clipped) run, tilting toward the former costs ~0.05 p@1.

**Verdict.** The frontier object of §8.4 is **not** reached by self-referential PRM reweighting. Variant (c)'s promise — replace the ad-hoc clip with a principled importance reweight — fails: the reweight is the wrong *kind* of operation (concentration, not bounding) to stabilize the on-policy reverse-KL dynamic, and adds nothing when the clip is present. This sharpens, rather than softens, the §8.1/§8.2 conclusion: the per-token clip is a **structural stabilizer of the on-policy reverse-KL dynamic** (§8.1 follow-up 2's "init-independent dead-zone attractor"), not a blunt instrument waiting to be replaced by something smarter. A better teacher *interface* (answer-conditioning, §7.10) shifts *which states the signal lands on*; a per-token *reweight within* that signal does not address the dynamical instability at all.

**Caveats.** (i) **Self-referential PRM only.** This tests the cheapest variant — the answer-info-gain `g_t`, not a separately-trained step-level PRM ([[prms-as-teachers]] variant (b)). A trained PRM scores *process correctness* rather than *answer-dependence*; the two coincide only loosely, and it remains possible (though now less likely, given finding-2's structural argument) that a true PRM's importance is bounded enough to avoid the tail-sharpening. (ii) **softmax, temperature 1, uncapped.** A *capped* reweight (`prm_weight_ceiling`) is itself a soft clip — deliberately left off on arm C to test *pure* reweight-vs-clip; the smooth interpolation "reweight + ceiling" between C and a clip was not swept and could in principle find a stable middle (but that is just re-introducing the clip the conjecture sought to remove). Higher `prm_temperature` → flatter `w_t` → C limits to plain unclipped OPSD (B), so the temperature axis only interpolates between two already-dead points. (iii) **Single seed (42).** No seed-43 follow-up was run: C showed *no signal* (dead-zone), so a seed sweep cannot change the qualitative verdict, unlike §7.10 where a positive s42 warranted multi-seed confirmation. (iv) The per-token `prm/weight_{mean,max,p99}` diagnostics went to W&B only (not stdout); the grad-norm inflation in finding-2 is read from the main-log `grad_norm` column, which is the relevant quantity for the collapse mechanism anyway.

---

## 7.12 Experiment 7 — Does the on-policy OPD collapse persist at scale?

**Question.** Every result above is **1B-student / 7B-teacher**. §8.1 established that at that capacity
gap, on-policy pure reverse-KL OPD is not merely weak but *dead* (p@1 ≈ 0.008), and §8.1 follow-up 2
showed *student competence* does not rescue it (a warm-started 1B at p@1 0.383 is collapsed to 0.003).
The §9 conclusion explicitly defers **one** remaining out: *true scale* — a larger student whose
on-policy distribution might be stable under reverse-KL. Exp 7 runs it.

> **Teacher-match note.** Exp 7 uses teacher **7B-Instruct** (to reuse the §8.1/L3 off-policy buffer).
> The clean, same-teacher 1B references are **A** (on-policy pure-OPD ← 7B-Instruct, §8.1: 0.008) and
> **B** (off-policy reverse-KD ← 7B-Instruct, §8.1/L3: 0.298). The 1B references for **C** (clipped
> pure-OPD ~0.05) and **D** (low-λ clip 0.709) come from the §7 interior + §7.7 clip work, which used
> π_T = 7B-**SFT** — so any 1B↔7B comparison for C/D is cross-teacher and is hedged below. The
> load-bearing claims are built on **within-Exp-7, same-teacher (7B-Instruct) contrasts** (A↔C↔B↔D all
> at 7B), which need no 1B reference at all.

**Design.** Native **7B-SFT student ← 7B-Instruct teacher** (same OLMo-2 base, shared tokenizer; the
teacher is the *same* 7B-Instruct as §8.1/L3, so the off-policy buffer is reused). Full-FT 7B fits one
80 GB H100 via bnb 8-bit Adam (`fit.optimizer_8bit`) with the frozen teacher **co-resident** on the
student's card (`teacher.device_id=0`). `gsm_symbolic`, 250 steps, seed 42, eval at the §8.1/Exp-5
protocol (T=0.6, 64 prompts × 16, eval-seed 1e6). Four arms (jobs `115090/115092/115093/115094`),
each one GPU, the launcher (`harness/run_exp7_arm_1gpu.sh`, config `harness/configs/exp7_scale_7b.yaml`)
overriding only alpha / lam / per_token_kl_clip / offpolicy buffer:

- **A** `alpha=1 lam=1 clip=null` — on-policy pure-OPD, **the** dead-at-1B arm. (KEY: revive at 7B?)
- **B** `alpha=0 lam=1 clip=null` +buffer — off-policy reverse-KD, the §8.1/L3 "alive" control.
- **C** `alpha=1 lam=1 clip=1.0` — on-policy pure-OPD **with the clip**. Does the clip rescue it at 7B?
- **D** `alpha=1 lam=0.10 clip=1.0` grpo — the best-1B recipe (clipped low-λ interior).

**Results — held-out eval @ step 250, T=0.6, 64×16.**

| arm | recipe | **p@1** | p@2 | p@4 | p@8 | p@16 | tok-ent | 1B ref |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **A** | on-policy pure-OPD, no clip | **0.0088** | 0.017 | 0.034 | 0.063 | 0.109 | 1.20 | 0.008 (dead) |
| **C** | on-policy pure-OPD, clip=1.0 | **0.3232** | 0.421 | 0.506 | 0.581 | 0.641 | 0.39 | ~0.05\* (dead) |
| **B** | off-policy reverse-KD | **0.4082** | 0.531 | 0.633 | 0.722 | 0.797 | 0.36 | 0.298 |
| **D** | on-policy low-λ clip (best) | **0.6611** | 0.742 | 0.802 | 0.848 | 0.875 | 2.76 | 0.709\* |

`*` C/D 1B refs used the 7B-**SFT** / task-specialized teacher; Exp 7 uses 7B-**Instruct**
(cross-teacher — see note above). **7B ranking: D 0.661 > B 0.408 > C 0.323 > A 0.009.**

**Training dynamics (the heart of the result).**
- **Arm A — delayed collapse, NOT revival.** Held a noisy mean acc ~0.42 to step ~120, then oscillated
  through increasingly violent collapse-recovery spikes (grad_norm 11→25→38), tipped into the dead zone
  at **step ~152** (acc 0.010, the same value 1B hit by step ~20) and **stuck** (steps 152→250 mean
  acc 0.017; grad_norm settling 38→13 as it sank in; rev_kl ballooning to 0.55). Final p@1 0.0088 ≈
  the 1B dead value (0.008). **The §7.6/§8.1 dead-zone attractor is scale-robust** — scale buys only a
  *delayed, more oscillatory onset*, not escape. (Texture: 7B's p@16 0.109 > 1B's 0.062 and tok-ent
  1.20 < 1B's 2.51 — the collapsed 7B is slightly less diffuse, but p@1 is equally dead.)
- **Arm C — the clip averts it, end-to-end.** grad_norm pinned ≤ 1.9 for all 250 steps; at step 152
  (where A had acc 0.010 / gn 19.8) C has acc 0.304 / gn 1.3. Zero dead-zone steps. The per-token clip
  bounds the heavy tail of per-token KL pushes (§8.2) and the on-policy reverse-KL dynamic stays stable
  at 7B exactly as the mechanism predicts.
- **Arm B — off-policy is stable + fast** (no rollout generation: ~11s/step vs ~35–50s for on-policy),
  trains cleanly to 0.408.
- **Arm D — the best recipe, stable and high.** Low-λ outcome anchor + clip: grad_norm ≤ ~4.6, acc
  tracking ~0.74–0.94 in training, zero dead-zone steps. Final p@1 **0.661**, the top arm at 7B, and
  uniquely **high-entropy (tok-ent 2.76** vs 0.36–0.39 for B/C). At 1B the high-entropy low-λ blend
  bought pass@k coverage at the cost of pass@1 precision (§7.8's diversity-for-coverage trade); at 7B
  it delivers **both** — top p@1 (0.661) *and* top p@16 (0.875). Scale appears to dissolve that trade.

**Four findings.**

1. **The on-policy reverse-KL collapse is scale-robust (same teacher, same objective).** Unclipped pure
   on-policy OPD ← 7B-Instruct dies at 1B (§8.1: 0.008) and at 7B (arm A: 0.009) — a *clean same-teacher
   1B↔7B contrast*. Scale alone does not stabilize the dynamic; §9's "needs genuine scale" out is
   **closed in the negative** for the *unclipped* recipe.

2. **The clip rescues pure on-policy OPD at 7B — a 37× within-experiment, same-teacher effect.** The
   airtight statement needs no 1B ref: holding teacher (7B-Instruct), scale (7B), and everything but the
   clip fixed, **A (no clip) 0.009 → C (clip=1.0) 0.323** — the per-token clip alone converts the dead
   pure-OPD recipe into a live one (and trains stably end-to-end, grad_norm ≤ 1.9, through the exact
   step-152 window where A collapsed). *Softer, cross-teacher add-on:* at 1B the clip did **not** make
   *pure* λ=1 OPD usefully alive (clipped pure-OPD ~0.05 with the 7B-**SFT** teacher; the live 1B recipe
   needed the low-λ anchor → 0.71), so the clip's *stabilizing* role looks scale-invariant while its
   *accuracy payoff on pure OPD* appears to grow with scale — the confirming 1B clipped-pure-OPD ←
   7B-**Instruct** run (**Exp 7b**, `harness/run_exp7b_1b_clip_instruct.sh`, job `115135`) **has now
   run: p@1 0.09–0.15** (s42 0.148 / s43 0.088), i.e. clip still yields a *live, non-collapsed* 1B
   student under an Instruct teacher (a modest lift over the ~0.05 with the SFT teacher — so the
   stabilizing role is scale- *and* teacher-invariant, while the absolute payoff stays well below the
   7B 0.323, consistent with the payoff growing with scale).

3. **The on-policy↔off-policy gap is small at 7B, and §8.1's "on-policy is a liability" does not
   generalize upward.** *Clean at 7B (same teacher):* clipped on-policy (C) 0.323 vs off-policy (B) 0.408
   → off-policy wins just **1.26×** across the whole pass@k curve. Contrast §8.1, where at the 1B/7B gap
   off-policy beat *unclipped/SFT* on-policy by ~38–55×. So the catastrophic on-policy penalty is a
   **small-student/large-capacity-gap artifact**, not a scale-general law — partially rehabilitating the
   post's "on-policy is load-bearing" intuition, **conditional on the clip**. *Hedge:* the precise
   narrowing multiplier uses a cross-teacher 1B clipped-on-policy figure (~0.05, 7B-SFT teacher); the
   *direction* (gap shrinks dramatically with scale/gap-closure) is robust, the exact multiplier is not.

4. **The best recipe (low-λ clip) stays best at scale, and the diversity↔precision trade dissolves.**
   D is the top arm at 7B on *both* p@1 (0.661) and p@16 (0.875) — the same recipe that topped 1B
   (0.709). The notable shift: at 1B the low-λ blend's high token entropy traded *against* pass@1
   precision off-task (§7.8); at 7B D is simultaneously the highest-entropy arm (tok-ent 2.76) **and**
   the highest-p@1 arm. The bigger student can be diverse *and* accurate at once — the ProRL-style
   coverage-for-precision tension looks like another small-student artifact. (Cross-teacher hedge applies
   to the 0.709→0.661 1B↔7B comparison; the *within-7B* "D is best on every k" is clean.)

**Caveat (load-bearing): scale and capacity-gap-closure are confounded here.** Exp 7 is 7B-student ←
**7B**-teacher, i.e. it simultaneously (a) makes the student bigger and (b) closes the 1B/7B capacity
gap to 7B/7B (student = teacher size). These cannot be separated in this run. The clean disambiguator
is a **7B-student ← 13B-Instruct** run — **Exp 7c (config
`harness/configs/exp7c_scale_13b_teacher.yaml`, launcher `harness/run_exp7c_arm.sh`, arms A/C/D with
the 13B teacher on a second card), whose A/C arms have now run (jobs `115136`/`115137`, 2026-06-12) and
resolve the confound toward scale/gap-robust instability, not gap-closure**: re-opening the capacity gap
does *not* prevent the collapse — unclipped on-policy OPD (A13) still dives to the §7.6 dead zone (train
acc 0.52 → 0.005–0.009 over steps ~75–200, grad_norm → 51), it merely escapes *unstably* (it oscillates
violently out in the last ~50 steps, so its final-checkpoint eval p@1 0.237 is a snapshot of a still-broken
run, not a stable recovery), while the clip (C13) is stable end-to-end (acc 0.21–0.68, grad_norm ≤ 5,
never dead) and lands p@1 0.403. So the collapse now reproduces across 1B←7B, 7B←7B, *and* 7B←13B, and
the clip stays the structural stabilizer regardless of the gap; the one new (n=1) wrinkle is that the
larger, more-informative teacher makes the dead zone *escapable-but-unstable* rather than the permanent
attractor seen at 7B←7B. The low-λ arm **D13 is the best arm at this gap too — p@1 0.746** (job `121947`,
stable end-to-end; vs the 7B←7B 0.661) **and the highest-entropy arm** (tok_ent 1.29), so the §7.8
precision↔coverage trade dissolves under the re-opened gap as well; more broadly the more-capable 13B
teacher *lifts the stable clipped recipes* (C 0.323 → 0.403, D 0.661 → 0.746) while leaving the unstable
unclipped arm A dead — i.e. teacher capability pays off only once the on-policy signal is stabilized by
the clip. Other caveats: single task
(`gsm_symbolic`), single seed (42) — arm A's collapse is unambiguous so no seed sweep is warranted for
it, but C/B/D single-seed numbers carry the usual ±0.01–0.05 caution; arm A's eval is a single
checkpoint drawn from a (pre-collapse) noisy trajectory, though it landed firmly in the dead zone so
the reading is robust.

---

## 7.13 Experiment 8 — Does a searched *task-level* hint rescue pure OPD? ([[per-task-hint-search-gepa]] #9)

**Question.** §7.10 showed that conditioning the *teacher* on the ground-truth answer (OPSD) rescues pure OPD from the dead corner (0.029 → 0.252 p@1) by moving the teacher's distribution onto reachable, answer-shaped states. But answer-conditioning is *per-problem privileged information* — the teacher sees the label. The natural question for a practitioner is whether a cheaper, **non-privileged** intervention buys the same rescue: a single fixed, *task-level* hint — one string, the same for every problem, with no access to the answer — appended to the teacher's context. If it does, the §7.10 rescue is a general "shift the teacher onto answer-shaped paths" effect and survives without privileged info; if it does not, the rescue was per-problem privilege all along, and task-level hint engineering is a dead end at λ=1.

**Design.** Two stages. **Stage 1 (search, no training):** a GEPA-style reflective search over fixed teacher-conditioning strings (`harness/hint_search.py`), scored on the Lagrangian `teacher_acc − β·kl_p99` against *one fixed* 64×4 student-rollout set — i.e. reward the hint for making the teacher correct on the student's own states while *not* inflating the heavy KL tail (`kl_p99` is the §8.2 collapse statistic). The 7B-Instruct model is the reflective mutator; 8 seed hints + 6×3 mutation rounds. **Stage 2 (train):** pure OPD (λ=1, clip=1.0) from the 7B-SFT teacher with the stage-1 winner wired in as a fixed `condition_on` string (`TeacherSpec.fixed_hint`; config `harness/configs/exp8_fixed_hint_opd.yaml`), matched to the §7.10 protocol. Two arms, **3 seeds each {42,43,44}**: **best** (the searched winner) vs **placebo** (a task-irrelevant tone hint, "Respond in a calm, friendly, and encouraging tone."). The placebo is the load-bearing control — it isolates *task-relevant distribution shift* from *"any appended conditioning string."*

**Stage-1 winner** (job `115142`): *"Ensure each step logically follows from the previous, showing all calculations clearly; aim for a simple, straightforward solution."* (teacher acc 0.547, kl_p99 4.881, Lagrangian score −0.429 — the best-scoring of all candidates).

**Results.** Stage-2, T=0.6, 64 prompts × 16 samples, held-out eval-seed 1e6 (jobs `125277`, `126571`–`126573`; the matched 3-seed control completed 2026-06-24).

| arm | seed 42 | seed 43 | seed 44 | **mean p@1** | p@16 (s44) |
|---|---|---|---|---|---|
| **best** (searched hint) | 0.245 | 0.126 | 0.247 | **0.206** | 0.453 |
| **placebo** (tone hint) | 0.210 | 0.219 | 0.146 | **0.192** | 0.313 |

**Read — task-level hint search is a dead end at λ=1.** The searched, task-relevant winner (0.206) does **not** beat the task-irrelevant placebo (0.192); the difference is within the per-seed spread (both arms are bimodal across seeds, exactly the failure mode the design warned about). Two conclusions follow. (1) Whatever small lift a fixed conditioning string buys over the unconditioned dead corner (≈0.01–0.05, §4) comes from *appending any string at all*, not from the string being task-relevant — so it is not the §7.10 mechanism. (2) The §7.10 rescue therefore **does not survive removing the per-problem answer conditioning**: it is *per-problem privilege*, not a general distribution-shift-onto-answer-shaped-paths effect. A non-privileged, task-level hint cannot stand in for the answer. This sharpens §8.4's frontier statement: the thing that makes OPSD work is *privileged, per-problem* information injected into the teacher, and the open problem is getting density + on-policy + outcome-alignment *without* that crutch — which is exactly what the clipped low-λ blend (§7.7) approximates from the outcome side and what PRM-reweight (§7.11) failed to reach.

_(per-arm eval json: `results/exp8_fixed_hint_opd_{best,placebo}_*/eval_*.json`; stage-1 search log + winner: `results/exp8_hint_search_115142/`; the §7.10 per-problem answer-conditioned rescue is the contrast.)_

---

## 8. Discussion — what the corners and the interior say

The experiments above measured points; this section reads them together. Four threads: what's actually load-bearing (§8.1), the per-token KL mechanism that makes the dense teacher term usable (§8.2), what the update geometry tracks (§8.3), and how close the interior gets to the post's "dense + unbiased + on-policy" frontier (§8.4).

### 8.1 What's load-bearing — on-policy state coverage, or teacher novelty?

**Question.** The post's strong claim is that on-policy data is load-bearing *because it trains on student-visited states* rather than fixed teacher traces. The newer OPD work (roadmap §13) reframes: teacher quality alone isn't the variable — what matters is whether the teacher creates *informative gradients on the student's actual error states*. Exp 1 already pinned one half: **teacher novelty is not the variable** — same-base SFT / DPO / Instruct teachers all drive pure OPD to the same near-zero pass@1 (§4: 0.011–0.014). The other half — *is on-policy state coverage itself load-bearing?* — needs the deferred off-policy-SFT arm of Exp 1 (line 109), which we now close.

**Design (L2, confound-controlled; jobs `80388` train + `80389` eval, 2026-06-04).** The naive comparison — off-policy SFT (§6.5, p@1 0.377) vs on-policy pure-OPD ← Instruct (§4, p@1 ~0.01) — is confounded: the §6.5 SFT data is *verifier-correct-only* (RFT/STaR), so "off-policy beats on-policy" could just be the correctness filter. To isolate it, we regenerated the **same** Instruct-teacher rollouts **without** the filter (`harness/rft_generate.py --keep_all`, new flag) and carved two datasets from one generation draw (6000 completions, 2550 correct = 42.5%, reproducing the original draw): **CORRECT** (the 2550 acc≥1.0 subset) and **UNFILT** (a random *same-size* 2550-subset, 44% correct). Both SFT the same 1B-SFT init with identical hp / #steps (epochs 2, lr 5e-6, 320 steps) — **only the correctness filter differs** — at seeds 42 & 43. We then re-eval the on-policy `opd-instruct7b-s42/s43` checkpoints at the matched protocol (T=0.6, 64 prompts × 16, eval-seed 1 000 000). Script `harness/run_exp1_offpolicy_sft.sh` (+ eval-only `run_exp1_offpolicy_eval.sh`); results in `results/exp1_offpolicy_sft_80389/`; checkpoints `harness/checkpoints/sft-rft-{correct,unfilt}-s4{2,3}/`.

| arm | recipe | p@1 (s42 / s43) | **mean p@1** | mean p@16 | tok-ent |
|---|---|---:|---:|---:|---:|
| **CORRECT** | off-policy SFT, verifier-filtered (RFT) | 0.383 / 0.393 | **0.388** | 0.672 | 0.41 |
| **UNFILT** | off-policy SFT, **un**filtered (44% correct) | 0.289 / 0.306 | **0.297** | 0.633 | 0.46 |
| **OPD ← Instruct** | **on-policy** pure-OPD (λ=1, reverse-KL) | 0.006 / 0.005 | **0.005** | 0.062 | 2.51 |

**Three findings.**

1. **On-policy state coverage is *not* load-bearing here — the post's central claim is reversed at the 1B/7B gap.** Off-policy SFT beats on-policy pure-OPD by **~55–72×** at pass@1 (0.297–0.388 vs 0.005) and ~10× at pass@16 (0.63–0.67 vs 0.06). Training on *fixed teacher traces* — the recipe the post argues should suffer train/test mismatch — dramatically outperforms correcting the student on *its own visited states*. The mechanism is §7.2's on-policy state-space lock-in seen from the other side: when the student starts at pass@1 ≈ 0.001, its own trajectories never enter answer-reachable regions, so the on-policy per-token signal is the teacher's local opinion on a globally-lost trajectory; the teacher's *own* traces, by contrast, are coherent and answer-shaped, and forward-CE on them transfers structure the student can execute.
2. **The correctness filter is a second-order effect, not the driver.** UNFILT — SFT on a trace mix that is *56% wrong* — still reaches **0.297 p@1**, only 0.09 below the filtered RFT arm (0.388) and still ~55× above on-policy OPD. So the filter buys a real but modest +0.09; it is emphatically *not* what makes off-policy SFT win. This is the confound the L2 design was built to kill, and it's dead: even unfiltered off-policy imitation crushes on-policy reverse-KL distillation.
3. **Putting it with Exp 1: neither axis the literature proposed is the binding constraint.** Teacher *novelty/recipe* doesn't move pure OPD (§4, all teachers ≈ 0.01); on-policy *state coverage* doesn't rescue it either (this section). What separates the winners (0.3–0.4) from the losers (0.005) is whether the learning signal lands on **coherent, answer-shaped trajectories at all** — off-policy SFT inherits them from the teacher; on-policy reverse-KL OPD, at this capacity gap, computes them on the student's near-random rollouts where the per-token teacher correction has no purchase. The entropy split corroborates the regimes: SFT students are peaked (tok-ent ~0.4), the OPD-Instruct student is diffuse (~2.5, the §4 high-entropy DPO/Instruct-teacher signature) — high entropy *and* near-zero accuracy, i.e. lock-in, not productive exploration.

**Caveats — and why this motivates L3.** This L2 contrast cleanly removes the *correctness-filter* confound, but it still varies three things at once between the winning and losing arms: on-policy-ness, **KL direction** (forward-CE on teacher traces vs reverse-KL on student traces), and **trace coherence**. So the precise statement licensed is *"off-policy SFT ≫ on-policy pure-OPD, and not because of the correctness filter"* — strong enough to reverse the post's headline at this scale, but it does **not** isolate on-policy-ness alone. The clean isolation (L3, **now run — resolved below**) holds teacher and the per-token reverse-KL objective fixed and varies *only* the states the loss is computed on — off-policy reverse-KD on **teacher**-sampled sequences vs on-policy OPD on student-sampled sequences, matched steps/LR. If off-policy reverse-KD *also* beats on-policy OPD, on-policy-ness is decisively not load-bearing; if it collapses to OPD's level, then the reverse-KL-on-student-states is the culprit and on-policy-ness was carrying the weight after all. Other caveats: single task (`gsm_symbolic`), 2 seeds per arm (within-arm spread ≤ 0.017, so the 0.09 CORRECT−UNFILT gap and the 55× off-vs-on gap are both well outside noise), 1B/7B capacity gap specifically — the conclusion may invert when the student is close enough to the teacher that its on-policy trajectories already reach answer-shaped regions.

**§8.1 follow-up — L3: the clean isolation (job `99626`, 2026-06-06).** We close the last confound. L3 trains an **off-policy reverse-KD** arm — the *same* per-token reverse-KL objective and the *same* teacher (7B-Instruct) as on-policy OPD, but with the experience buffer filled from the teacher's **own** rollouts (the `_ALL.jsonl` from §8.1) instead of `student.generate()`. Only the states the loss lands on differ; init / LR (1e-5) / #steps (500) / #seqs-per-step (8×8) / clip (null) all match the on-policy `opd-instruct7b` arm. Implemented as `offpolicy_teacher_states` in `harness/unified_trainer.py::_run_distill_loop` (config `harness/configs/exp81_offpolicy_revkd.yaml`, launcher `harness/run_exp81_offpolicy_revkd.sh`); checkpoints `harness/checkpoints/offpolicy-revkd-instruct7b-s4{2,3}/`; results `results/exp81_offpolicy_revkd_99626/`.

| arm | states | objective | p@1 (s42 / s43) | **mean p@1** | mean p@16 | tok-ent |
|---|---|---|---:|---:|---:|---:|
| **off-policy reverse-KD** | **teacher**-sampled | reverse-KL (clip=null) | 0.310 / 0.286 | **0.298** | 0.664 | 0.51 |
| **on-policy OPD** ← Instruct | **student**-sampled | reverse-KL (clip=null) | 0.010 / 0.006 | **0.008** | 0.070 | 2.51 |

**The first branch of the prediction fires: on-policy-ness is *not* load-bearing here — at the 1B/7B gap, pure on-policy OPD is actively *harmful*.** Holding the teacher and the reverse-KL objective fixed and moving the loss from student-sampled to teacher-sampled states lifts p@1 from **0.008 → 0.298 (~38×)** and p@16 from 0.070 → 0.664 (~9×). Reverse-KL on the student's own (here, pass@1≈0) trajectories is *exactly* what kills on-policy OPD; the same objective on the teacher's fixed answer-shaped trajectories is alive and well. (The obvious "but a *competent* student would be fine on-policy" rescue — the sign-flip — is tested in **follow-up 2** below and **falsified**: the failure is a training *instability*, not an initial-state-coherence effect.)

Three things this nails down that L2 could not:
1. **KL direction is not what drives the OPD failure here.** Off-policy reverse-KD (0.298) ≈ off-policy SFT-unfilt (0.297, L2) — within seed noise. Swapping forward-CE for reverse-KL, holding the states off-policy, moves essentially nothing. This does **not** show KL direction *never* matters — only that it is ruled out as the explanation for *the OPD failure in this experiment* (it does not separate forward/reverse-KL in general).
2. **The gap is the state distribution, in this setting.** Combining L2 (correctness filter = +0.09, second-order) and L3 (KL direction ≈ 0), the on-vs-off gap (~38–55×) is attributable to **which states the loss is computed on** — not the correctness filter, not the KL direction, not teacher novelty (§4). Every *measured* confound the L2 contrast bundled is ruled out **at this scale**; scale and task remain external-validity limits these controls do not touch.
3. **Entropy confirms the lock-in mechanism.** Off-policy reverse-KD trains a *peaked* student (tok-ent 0.51, matching the off-policy SFT students at 0.41–0.46); on-policy OPD a *diffuse* one (2.51) — same teacher, same objective, opposite entropy regimes, decided purely by the states. The on-policy diffuseness is the §8.1 "high entropy *and* near-zero accuracy = lock-in, not exploration" signature, now produced as a controlled contrast.

**Net, across L2 + L3:** none of the axes the literature proposed — teacher novelty (§4), on-policy state coverage (L2/L3), correctness filtering (L2), or KL direction (L3) — is the binding constraint at the 1B/7B gap. The one variable that moves p@1 from ~0.01 to ~0.30 is whether the per-token signal is computed on *coherent, answer-shaped trajectories at all* (the teacher's), rather than the student's own lost ones. On-policy-ness — the post's headline mechanism — is, here, a liability.

One scoping is load-bearing in every statement above: **"state distribution" here bundles trajectory *coherence*.** L3 alone showed teacher-sampled *answer-shaped* states useful and student-sampled *lost* states not — but at the 1B/7B gap the cold student's "on-policy" and "incoherent" coincide, so L3 could not say *which* was doing the work. The obvious test is the **sign-flip**: warm-start the student so its *own* rollouts are answer-shaped, and see whether on-policy OPD revives. We ran it — and it does **not**.

**§8.1 follow-up 2 — the scale/coherence test (job `100411`, 2026-06-07): the sign-flip does NOT fire.** Warm-start the 1B student from `sft-rft-correct` (off-policy-SFT'd, eval p@1 0.383 — and at *training* step 1 its on-policy rollouts score **acc 0.505**, i.e. *more* answer-shaped than the teacher's 0.46), then run both arms from that competent init, teacher/objective(reverse-KL, clip=null)/LR/steps all matched (launcher `harness/run_exp81_scale_warmstart.sh`; results `results/exp81_scale_warmstart_100411/`).

| arm (from warm init, p@1 0.383) | mean p@1 (s42/s43) | mean p@16 | tok-ent | vs cold-start L3 |
|---|---:|---:|---:|---|
| **on-policy OPD** (student states) | **0.003** (0.004/0.002) | 0.039 | **2.68** | cold OPD 0.008 — *same dead zone* |
| **off-policy reverse-KD** (teacher states) | **0.290** (0.293/0.287) | 0.664 | 0.51 | cold revKD 0.298 — *unchanged* |

**On-policy pure-OPD does not revive — it *collapses the competent student*.** From an init whose on-policy rollouts are 50% correct, on-policy reverse-KL drives p@1 from 0.383 → **0.003** and token entropy 0.41 → 2.68, reaching the *same* dead-zone attractor as the cold start (0.008). The training trace pins it: rollout accuracy is **0.505 at step 1**, then collapses to ~0.010 by **step 20** and decays to ~0.003 — the §7.6/§8.2 unclipped-OPD collapse, now shown to be **init-independent** (the student is dragged to the dead zone from *above* just as the cold student arrives from below). Off-policy reverse-KD, by contrast, is **stable from any init** (0.290 warm ≈ 0.298 cold), because it trains on a *fixed* coherent teacher distribution — there is no policy chasing its own collapsing distribution.

This **refines (and partly overturns) the L3 reading**: the binding variable is *not* the coherence of the student's **initial** states — a competent, answer-shaped init does not help. It is the difference between training on a **fixed** coherent distribution (off-policy teacher states → stable) and on the policy's **own moving** distribution, which unclipped on-policy reverse-KL **collapses** (the dead-zone attractor of §7.6), regardless of where it starts. So the post's "on-policy is load-bearing" headline is **not** rescued by student competence; pure on-policy OPD is destructive even to a good student.

Strongest defensible phrasing: *at the 1B/7B gap, pure unclipped on-policy reverse-KL OPD converges to a dead-zone attractor from any initialization — warm (acc 0.505 → 0.003) or cold (≈0 → 0.008) — while the same teacher and objective on fixed off-policy teacher trajectories is stable and effective (≈0.29). The failure is a training instability of the on-policy reverse-KL dynamic, not an initial-state-coherence effect, and not a property of off-policy data being "bad."* Caveats: this is **pure λ=1, clip=null** OPD (matched to L3) — the §7.7 clipped low-λ *blend* (with an outcome anchor) reaches 0.71 and §7.10 answer-conditioning reaches 0.25, so what rescues on-policy training is the anchor or privileged-info teacher, **not** the warm start; clipped *pure* OPD was also ≈0.05 dead cold (§6.1), so warm+clip is unlikely to flip it but is the one clean remaining check. Single task, 2 seeds; the warm init is itself off-policy-SFT'd on this teacher's traces.

**Scale update (2026-06-11):** the "external-validity limit" this section repeatedly flags — *does any of this survive a genuinely larger student?* — is now **answered with nuance in §7.12 (Exp 7)**: the unclipped on-policy collapse is **scale-robust** (a native 7B student ← the same 7B-Instruct teacher dies at p@1 0.009, identical to 1B's 0.008, just with a delayed step-~152 onset), but the on↔off-policy *gap* is what scale closes (clipped on-policy 0.323 vs off-policy 0.408 at 7B — 1.26×, vs 38–55× here). So §8.1's "on-policy is a liability" is a *small-student/large-gap* statement, not a scale-general law; the collapse-dynamic statement, by contrast, generalizes.

### 8.2 The per-token KL mechanism — unclipped collapse vs clipped recovery

The §7.7 result hinges on a single mechanistic claim: the unclipped low-λ trainer fails because **outlier per-token KL pushes drive a collapse-recovery instability**, and `per_token_kl_clip` works because it bounds those outliers. We can now check this directly. The harness logs `kl_signal/{p50, p90, p99, abs_max, heavy_tail_frac}` per step into W&B (Phase B diagnostic; landed before the 71208 seed-43 sweep). Pulling the offline binaries for the **unclipped 71208 (seed-43, full λ sweep)** and the **clipped 71395 (seeds 43/44/45, λ∈{0.05, 0.10, 0.20, 0.35}, clip=1.0)** runs gives a direct apples-to-apples contrast (blog-ready figure: `research/figs/exp4_kl_signal_mechanism_polished.png`; raw 4-panel diagnostic: `research/figs/exp4_kl_signal_mechanism.png`; extraction: `research/harness/extract_kl_signal_traces.py`; plot scripts: `research/harness/plot_kl_signal_mechanism.py` + `plot_kl_signal_polished.py`).

**Final-50-step means (the clean number):**

| λ | unclipped (s43): p99 / heavy_tail / acc | clipped (3-seed): p99 / heavy_tail / acc |
|---|---|---|
| 0.05 | **2.16 / 0.017 / 0.04** (collapsed) | **0.41 / 0.010 / 0.71** (recovered) |
| 0.10 | **2.46 / 0.023 / 0.02** (collapsed) | **0.46 / 0.007 / 0.73** (recovered) |
| 0.20 | **2.94 / 0.043 / 0.02** (collapsed) | **0.67 / 0.011 / 0.71** (recovered) |
| 0.35 | 2.08 / 0.013 / 0.02 (collapsed)     | 0.96 / 0.013 / 0.46 (partial recovery)   |

**The mechanism the figure shows.** In the unclipped sweep, rollout accuracy peaks around step 50 (acc ~0.5 for the lowest λ), then collapses to ~0.02 by step 100-150 for *all* low-λ arms on seed 43; meanwhile p99 climbs from ~1.0 to ~2-3 and heavy_tail_frac climbs from ~0.025 to ~0.04. In the clipped 3-seed sweep, p99 sits just below the |kl|=1.0 cap (0.41-0.96), heavy_tail_frac flattens at ~0.007-0.013 (it can't rise above the cap), the collapse around step 100-150 happens **but then recovers** by step 200-300, and final accuracy stabilizes at 0.71±0.03 for λ∈{0.05, 0.10, 0.20}. The peak unclipped p99 is largest at λ=0.20 (2.94) — exactly the predicted "interior tug-of-war" point between the teacher and outcome branches — and that's also where the unclipped heavy_tail_frac peaks (0.043, almost double the other arms).

**This makes the §7.6 / §7.7 mechanism narrative quantitative.** The §7.6 claim was "outlier KL pushes drive collapse"; the §7.7 claim was "clip=1.0 prevents collapse in the low-λ band." The kl_signal traces show both effects directly: (a) unclipped p99 *rises* over training (the outliers grow), (b) clipped p99 sits at the cap and *doesn't rise*, (c) the accuracy collapse-and-recovery is locked to whether p99 is bounded. The §7.7 prediction "heavy_tail_frac plateauing as rollout-acc lifts" was right *in the clipped runs* (they plateau low), and the unclipped arms show the opposite (heavy_tail rises). The contrast isn't a within-λ pattern (because seed-43 universally collapsed unclipped); it's the *clipped-vs-unclipped* contrast that makes the figure.

**Reverse-KL's sharpening cost, in this framework.** The pass@k crossover story from §7.4 and §7.8 fits cleanly on top: clipped low-λ trains a high-entropy (~8× token entropy vs GRPO) student that under-performs at p@1 cross-task but **beats GRPO at p@32-p@64** at T=0.6 on `simple_equations`. So the (α=1, λ=0.10, clip=1.0) interior is **not** the strictly-dominant recipe the in-distribution numbers might suggest — it's a *crossover-dominant* recipe that buys pass@k coverage at the cost of pass@1 precision off-task. That's the ProRL-style trade made measurable, which is exactly what [[entropy-collapse-opd-vs-rl]] and [[pass-at-k-vs-pass-at-1]] predicted at proposal stage.

The ordering forward-KL SFT < RL < reverse-KL OPD on diversity proxies (the proposal-stage prediction) is *not* what we see. We see: **RL (GRPO) collapses entropy → narrow but reward-aligned; clipped low-λ OPD-blend keeps entropy → wider but more diffuse**. The "OPD/OPSD is even more mode-seeking than RL" prediction is wrong for this regime — *because* per-token clipping prevents the trainer from going fully reverse-KL. The interior point at (α=1, λ=0.10, clip=1.0) sits closer to the OPSD wing than the GRPO wing on diversity, but not by collapsing further than GRPO; rather, by *avoiding* GRPO's collapse.

### 8.3 Update geometry ≈ loss structure, *not* teacher recipe

The post frames the meta-algorithm as a way of *picking a point in update-geometry space*: SFT supposedly writes a dense, redundant update; RL a sparse, essential one (RL's Razor); and OPD, being distillation, should look SFT-shaped. The natural strong hypothesis — **"OPD's update geometry tracks its teacher's recipe"** (RL-teacher → RL-sparse, SFT-teacher → SFT-dense) — is the cleanest version of that story, and Exp 3 (§6–§6.5) was built to test it. It is **falsified**, and the way it fails is more interesting than the hypothesis.

**Finding 1 — every on-policy (α=1) trainer is RL-sparse, regardless of teacher dose.** Across the whole sweep — RL baseline, GRPO, clipped low-λ interior, clipped high-λ, pure reverse-KL OPD (λ=1), unclipped v2.1 — the static sparsity sits in one band: top-1% |Δθ| mass 0.57–0.63, top-5% mass 0.91–0.94, only 6–10% of weights moved by >1e-4 (§6). **Pure OPD on a same-base teacher produces an RL-shaped sparse update, not the SFT-shaped dense update the literature predicts for distillation.** Mixing in more teacher signal (λ: 0.05→1.0) does not slide the update toward "dense" — it stays sparse. So the teacher *dose* is not the geometry knob the strong hypothesis assumed.

**Finding 2 — two *functional* tiers, separated by how concentrated the essential subnetwork is.** The static mass numbers hide a real split that the prune-degradation curves (§6.1/§6.2) expose. Reverting the bottom p% of moved weights to init and re-evaluating:

| arm (tier) | p@1 @0% | p@1 @50% | p@1 @90% | p@1 @95% | shape |
|---|---:|---:|---:|---:|---|
| GRPO-v2-s42 (**sharper**) | 0.540 | 0.552 | 0.439 | 0.266 | small, *concentrated* essential subnet → graceful decline |
| clip1 λ=0.10 (**broader**) | 0.691 | 0.696 | 0.008 | 0.001 | larger, *dispersed* essential subnet → cliff at p∈[80,90] |
| rl_baseline (**broader**) | 0.552 | 0.558 | 0.003 | 0.003 | same cliff |
| pure OPD λ=1 (**dead**) | 0.047 | 0.029 | 0.161 | **0.242** | *mistargeted* — pruning *helps* (5× at p95) |

GRPO keeps half its accuracy with only the top 5% of moves; the broader-tier arms collapse to ~0 once you cross the p∈[80%,90%] cliff. Same static sparsity, qualitatively different *functional* concentration.

**Finding 3 — the tier difference is a submodule-specific spectral reorganization, not a uniform "denser vs sparser".** The full per-tensor effective-rank pass (§6.4, correcting a top-10-by-frob artifact in §6.3) shows the tiers diverge only in specific submodules: the **sharper tier adds ~6% spectral capacity to attention** (attn_qkv ΔW eff-rank ratio 1.07 vs broader 1.01) while **compressing the embedding spectrum ~3%** (embed 0.97 vs broader 1.00); the broader tier is rank-preserving everywhere. Read structurally: sharper = "reshape attention to compute new things, narrow the embeddings to emit fewer things" (concentrated, structural → graceful prune); broader = "value-shift within the existing basis at every submodule" (dispersed, additive → catastrophic rank-collapse cliff). **Pure OPD is GRPO's geometry, mistargeted** — it has the *highest* attn-rank addition (1.069) and one of the lowest embed ratios (0.957), i.e. the sharper-tier shape, but its moves point the wrong way, which is why reverting most of them *improves* pass@1 (§6.2: 0.047 → 0.242 at p=95%). The geometry signature is "sharper-shaped and rank-additive, but aimed off the reward."

**Finding 4 — off-policy SFT-from-rollouts is a *third* regime, and it is the SPARSEST of all.** The (α=0, λ=1, π_T=δ_data) corner (§6.5) — the canonical SFT recipe — has top-1% mass 0.729 (vs sharper 0.61–0.63), only 1.6% of weights moved >1e-4, and the deepest embedding-rank compression of any arm (0.846). It is a *more extreme* version of the sharper-tier spectral shape (attn-rank-adding + embed-compressing), with a graceful (no-cliff) prune curve. This **falsifies "RL sparse / SFT dense" from the other direction**: at this scale and training intensity, SFT is *sparser* than on-policy RL, because off-policy imitation only fills in a narrow mode rather than pulling the policy toward a new attractor.

**Verdict.** The post's geometry story survives *in spirit but not in mechanism*. In spirit: the (α, λ, clip, π_T) knobs do move you to measurably different points in update-geometry space — three regimes (SFT-corner / sharper / broader) that differ in (i) how concentrated the essential subnetwork is, (ii) how each submodule's spectrum is reshaped, and (iii) whether prune degradation is graceful or cliffed. But the *specific* claim "update geometry tracks the teacher's recipe" is wrong: geometry tracks the **structure of the loss — how many forces the trainer is balancing** — not the teacher's identity. One dominant signal (SFT's teacher traces, or a high-λ/pure-teacher term, or pure outcome RL) → a concentrated, sharper/SFT-shaped update; two competing signals (the broader low-λ outcome+teacher blend) → a dispersed, rank-preserving update with a prune cliff. Crucially, **on-policy reverse-KL OPD never produces an SFT-shaped update regardless of which same-base teacher drives it** — it is always RL-sparse, because the on-policy outcome geometry, not the teacher's recipe, sets the shape. This dovetails with §4 (teacher *novelty* doesn't move pure-OPD accuracy) and §8.1 (on-policy *coverage* doesn't rescue it): the teacher's identity is not the load-bearing variable on any axis we measured — capability, or geometry.

### 8.4 The frontier: dense, on-policy, *and* outcome-aligned

The post closes on an open problem: find an update with **the density of distillation** (a learning signal on *every* token, not just a sparse sequence-level reward), **the unbiasedness of RL** (aligned with the *true* reward, not a teacher's possibly-skewed preferences), and **the on-policy property of both** (trained on the policy's own visited states). The three corners each give up one leg: RL is unbiased + on-policy but signal-sparse; on-policy OPD is dense + on-policy but biased toward the teacher (and, as §8.1 shows, dead at this gap when the teacher signal lands on lost trajectories); off-policy SFT is dense but off-policy and biased. Do any of our interior / teacher-interface points hit all three at once?

**The clipped low-λ interior is an approximate "yes" — with two asterisks.** The (α=1, λ=0.10, clip=1.0) point (§7.7) is, by construction, dense (the per-token teacher reverse-KL term touches every token), on-policy (α=1), and outcome-aligned (the (1−λ)=0.9 clipped-GRPO branch anchors it to the verifier). It is the best in-distribution recipe in the whole study — p@1 **0.693** (4-seed mean; §7.7/§7.8; 0.709 single-seed best), edging the multi-seed GRPO baseline (~0.687, §7.5) — and at high pass@k off-task it *dominates* GRPO (beats it at p@32–p@64 on `simple_equations`, §7.8). So density + on-policy + outcome-alignment is not just achievable, it's mildly *better* than pure RL where it counts. The asterisks: **(1) density is only safe when clipped.** §8.2 makes the mechanism quantitative — the unclipped dense teacher term has a heavy tail of outlier per-token KL pushes (p99 climbing to 2–3, heavy-tail fraction rising over training) that drive the step-100–150 collapse; `per_token_kl_clip=1.0` bounds them (p99 pinned below the cap, collapse-then-recovery instead of death). Without the clip, "density" is *destabilizing*, not helpful. **(2) it is crossover-dominant, not strictly dominant** — the high-pass@k coverage is bought with a reverse-KL sharpening cost that *lowers* off-task pass@1 precision (§7.8). There is no free lunch; there is a coverage-for-precision trade, gated by the clip.

**PRM-as-teacher attacks the same bottleneck from a different side.** §8.1/§8.3 located the real binding constraint: the per-token signal has to land on *coherent, answer-shaped, student-reachable* trajectories — teacher *novelty* (§4) and on-policy *coverage* (§8.1) are both off the critical path. The clipped low-λ recipe satisfies that constraint by borrowing reachability from the outcome branch (the GRPO advantage keeps the policy near reward-reachable states). Exp 5 (§7.10) satisfies it a different way: **answer-conditioning the teacher** shifts *the teacher's own distribution* onto student-reachable answer-shaped paths, which rescues pure OPD (λ=1) from the dead corner — 0.029 → **0.252 p@1** (4-seed) — and mid-λ from 0.099 → **0.269**, with *no outcome anchor at all*. That is a dense, on-policy signal carried entirely by the teacher term. But it trades away the *unbiasedness* leg: the answer-conditioned teacher sees privileged info, and pure OPSD (0.252) still sits well below GRPO (0.687). Tellingly, the optimal λ shifts *left* under OPSD (the teacher can safely carry more weight: λ=0.50 stays alive at 0.269, where the logit teacher's λ=0.50 was dead at 0.099) — picking the right teacher interface changes the right blend. So the (α, λ) knob and the π_T knob **interact**: they are not independent axes.

**What stays open.** None of our points closes all three legs *simultaneously and natively*. The clipped low-λ blend gets unbiasedness only by *importing* it from the outcome branch (set λ→1 and it dies; the teacher alone is biased); OPSD gets density-without-an-anchor only by *spending* unbiasedness (privileged info). The missing object is a teacher that is **itself dense and itself (more) unbiased** — i.e. a learned process reward model whose per-token signal is both everywhere and reward-correlated. The first cut at it — [[prms-as-teachers]] variant (c), **PRM-reweighted OPSD**, where a per-token process importance reweights the teacher-KL term — **has now run (§7.11, job `111363`) with a self-referential answer-info-gain importance, and it does not deliver.** Its sharp headline test was whether the reweight could *replace* the blunt clip; it cannot (no-clip+reweight = 0.006 p@1, the §7.6 dead zone, *worse* than no-clip alone). The reason is structural and worth carrying forward: **a mass-preserving reweight and the clip are opposing operations** — the clip *bounds* the heavy tail of per-token KL pushes (§8.2), a reweight *concentrates* mass onto a few tokens and thereby *sharpens* that tail, firing the §7.6 collapse harder (grad norms 3–6× higher than the un-reweighted no-clip arm). And stacked on the clip it slightly *hurts* (0.157 vs 0.204), the §7.10 "helps where the teacher fails, hurts where it already works" pattern again. So reweighting the per-token signal does not touch the binding constraint — the *dynamical instability* of the on-policy reverse-KL signal (§8.1 follow-up 2), which only the clip/anchor stabilizes. What remains genuinely untested is variant (b) — a *separately-trained* step-level PRM scoring process correctness rather than answer-dependence — though §7.11's structural argument (reweight concentrates, doesn't bound) predicts it too will need a bound to be safe. Three other open edges: (a) the precision↔coverage trade (§7.8) — is there a recipe that is *strictly* dominant, or is the trade fundamental to reverse-KL? **§7.12 partially answers this too: at 7B the trade dissolves** (arm D is simultaneously the highest-entropy and highest-p@1 arm), so the trade looks like another small-student artifact rather than a reverse-KL fundamental; (b) **scale** — **now run (§7.12, Exp 7) and answered with nuance**: the unclipped on-policy reverse-KL collapse is *scale-robust* (native 7B student ← 7B-Instruct dies at 0.009, same dead zone, delayed onset), so "genuine scale" does *not* revive the unclipped recipe; what scale buys is (i) the clip's accuracy payoff on *pure* OPD (0.009 → 0.323 at 7B, a same-teacher 37× contrast) and (ii) a collapse of the on↔off-policy gap (38–55× at 1B/7B → 1.26× at 7B/7B). The successor open question is the §7.12 confound: **scale vs capacity-gap-closure** (Exp 7 changed both at once) — now **largely resolved toward *scale/gap-robust instability*** by Exp 7c's A/C arms (7B ← **13B**-Instruct; jobs `115136`/`115137`, 2026-06-12): re-opening the capacity gap does *not* prevent the collapse — unclipped A13 still dives to the §7.6 dead zone (escaping only *unstably*, so its final-checkpoint p@1 0.237 is a snapshot, not a recovery), while clipped C13 stays alive (p@1 0.403). So gap-closure was *not* what produced the §7.12 rescue; the unclipped instability persists across the re-opened gap and the clip remains the stabilizer either way; the low-λ arm D13 is again the best arm (p@1 0.746, job `121947`, vs the 7B←7B 0.661) *and* the highest-entropy one, so the precision↔coverage trade dissolves at 7B←13B too, and the more-capable teacher *lifts the stable clipped recipes* (C 0.323 → 0.403, D 0.661 → 0.746) while the unstable unclipped arm A stays dead — teacher capability pays off only once the clip stabilizes the on-policy signal (the 1B clipped-pure-OPD ← Instruct hedge-closer Exp 7b has separately run at p@1 0.09–0.15 — clip still yields a *live* 1B student under an Instruct teacher); (c) **what the outcome anchor is really repairing — now answered by L3 + the warm-start test (§8.1).** L3 shows the culprit in pure on-policy OPD is not the KL *direction* but the on-policy *dynamic*: off-policy reverse-KD (teacher-sampled states) reaches 0.298 with *no* anchor, while on-policy OPD collapses to 0.008 — *and the warm-start test shows it collapses a competent student (0.383 → 0.003) just the same*. So the clipped low-λ anchor's job is **not** merely to bridge a capability gap (competence didn't help) — it is to **stabilize the on-policy reverse-KL dynamic against the §7.6 dead-zone collapse**. That makes the anchor (or per-token clip) a *structural* component of any on-policy dense-distillation recipe at this gap, not a temporary crutch — and reframes the frontier object: density+unbiased+on-policy is gated by *dynamical stability of the on-policy signal*, which the off-policy setting gets for free (fixed teacher states) and the on-policy setting must engineer.

---

## 9. Conclusion

**One knob set, not three recipes.** We took the post's framing literally and built a single trainer parameterized by `(α, λ, π_T)` — how on-policy the states are (α), how much of the per-token advantage is the teacher's reverse-KL vs. the outcome reward (λ), and which teacher (π_T). SFT, RL, OPD, and OPSD are then not four algorithms but four *corners* of one space, and everything interesting lives in how the loss puts probability mass where — which the unified harness (`research/harness/`) lets us sweep continuously. The payoff of the framing is that the same machinery produced every number in this document, and the cross-corner comparisons are apples-to-apples by construction (shared student init, data, eval protocol, and code path).

**The headline findings.**
- **Pure on-policy OPD is dead at the 1B/7B gap, and no teacher rescues it.** Every same-base teacher — SFT, DPO, Instruct — drives λ=1 OPD to p@1 ≈ 0.01 (§4). Teacher *novelty/recipe is not the variable.*
- **On-policy state coverage is not load-bearing here either — the post's central claim reverses at this scale.** Off-policy SFT beats on-policy pure-OPD ~55–72× (0.30–0.39 vs 0.005), and the correctness filter is only a +0.09 second-order effect (§8.1). What separates winners from losers is whether the per-token signal lands on *coherent, answer-shaped, reachable* trajectories at all — not whose states they are.
- **A single scalar separates collapse from the best recipe.** `per_token_kl_clip=1.0` is the OPD collapse-recovery mechanism (§7.6/§7.7): it bounds the heavy tail of outlier per-token KL pushes that otherwise drive the step-100–150 death (§8.2). With it, the dense teacher term becomes usable, and the **clipped low-λ interior (α=1, λ=0.10, clip=1.0) is the study's best recipe** — dense + on-policy + outcome-aligned, edging GRPO in-distribution (0.693 vs 0.687, 4-seed means) and dominating it at high pass@k off-task (§7.8).
- **Update geometry tracks the loss structure, not the teacher.** All on-policy reverse-KL updates are RL-sparse regardless of teacher; the real variation is how concentrated the essential subnetwork is and which submodules' spectra get reshaped, and that tracks *how many forces the trainer balances* (§6/§8.3). Off-policy SFT is the *sparsest* update of all, not the densest.
- **The teacher interface and the blend interact.** Answer-conditioning the teacher (OPSD) rescues pure OPD from the dead corner (0.029 → 0.252 p@1, 4-seed; §7.10) by moving the *teacher's* distribution onto reachable answer-shaped states — and it shifts the optimal λ leftward. Picking π_T changes the right λ.
- **The OPSD rescue is *per-problem privilege*, not generic distribution shift.** A searched, task-relevant, *non-privileged* hint (one fixed string for every problem) does not beat a task-irrelevant placebo tone hint (§7.13: 0.206 vs 0.192, 3-seed, both bimodal). Task-level hint engineering is a dead end at λ=1; what carries the §7.10 rescue is the *answer*, injected per problem — which sharpens, not softens, the frontier statement below.
- **The collapse is scale-robust; the gap is not (§7.12).** A native 7B student ← the same 7B-Instruct teacher: unclipped on-policy pure-OPD still dies (0.009, delayed onset), but the clip alone now rescues it (0.323, 37× same-teacher), the on↔off-policy gap shrinks 38–55× → 1.26×, and the clipped low-λ recipe tops every k while staying the most diverse arm. Scale (confounded with gap-closure) changes *who wins*, not *what collapses* — and Exp 7c's A/C arms now confirm this: re-opening the gap to a 13B teacher does not prevent the unclipped collapse (A13 still dives to the dead zone), while the clip stays stable (C13 0.403), so the collapse is robust to the gap, not just to scale.

**What surprised us.** Three results ran against the proposal-stage intuitions. (1) OPD, a distillation method, writes an *RL-shaped* (sparse) update, not an SFT-shaped (dense) one — and SFT-from-rollouts is sparser still. (2) High token entropy is *not* exploration: the dead OPD student is simultaneously diffuse (entropy ~2.5) and near-zero accuracy — lock-in, not coverage (§7.4). (3) The post's headline — on-policy data is load-bearing because it trains on student-visited states — *inverts* at the 1B/7B capacity gap: when the student's own trajectories never reach answer-shaped regions, training on the teacher's coherent traces is dramatically better than correcting the student on its own lost ones.

**What stays open.** The frontier object the post asks for — density *and* unbiasedness *and* on-policy, natively — is only approximated here: the clipped low-λ blend imports unbiasedness from the outcome branch, and OPSD spends unbiasedness on privileged info (§8.4). The missing piece is a teacher that is itself dense and itself reward-correlated. The first attempt at it — **PRM-reweighted OPSD** ([[prms-as-teachers]] variant c) — **has now run (§7.11, job `111363`) and failed** in its self-referential (answer-info-gain) form: reweighting the per-token teacher-KL by importance **cannot replace the clip** (no-clip+reweight collapses to 0.006 p@1, the §7.6 dead zone) and **slightly hurts when stacked on it** (0.157 vs the 0.204 clipped baseline). The reason is structural — a mass-preserving reweight *concentrates* per-token KL mass and so *sharpens* the very heavy tail the clip exists to *bound* (§8.2), firing the §7.6 collapse harder (grad norms 3–6×). So the per-token clip is confirmed a **structural stabilizer of the on-policy reverse-KL dynamic, not an ad-hoc instrument awaiting replacement**; importance reweighting does not address the dynamical instability that is the real binding constraint. What remains untested is a *separately-trained* step-level PRM (variant b) scoring process correctness rather than answer-dependence — though §7.11's argument predicts it too would need a bound to be safe. Two structural caveats bound every claim: **scale** — all results are 1B-student / 7B-teacher, and §8.1's reasoning predicts the picture may invert once the student is close enough to the teacher that its on-policy rollouts already reach answer-shaped regions; and **KL direction** — the L3 control (§8.1 follow-up, off-policy reverse-KD on teacher states; `offpolicy_teacher_states` in the harness, `harness/run_exp81_offpolicy_revkd.sh`, job `99626`) **has now run**: holding teacher + reverse-KL objective fixed and moving only the states from student-sampled to teacher-sampled lifts p@1 from 0.008 to **0.298** (~38×) — matching off-policy SFT (0.297). So on-policy-ness is *not* load-bearing at this gap — pure on-policy OPD is harmful. The on-vs-off gap is, in this setting, the state distribution (which **bundles trajectory coherence**), not the correctness filter (L2) and not the KL direction (L3 — ruled out as the explanation *here*, not in general). The **warm-start/sign-flip test** (job `100411`, §8.1 follow-up 2) has now **run and falsified the obvious rescue**: a *competent* init (p@1 0.383; on-policy rollouts 50% correct at step 1) is *also* collapsed by on-policy pure-OPD — to 0.003 — reaching the same §7.6 dead-zone attractor as the cold start. So the failure is a **dynamical instability of the on-policy reverse-KL signal**, init-independent, not an initial-state-coherence effect; off-policy reverse-KD avoids it by training on a *fixed* coherent teacher distribution. The **scale** out has now also been run (**Exp 7, §7.12**, native 7B-SFT student ← the same 7B-Instruct teacher) and is **closed in the negative for the unclipped recipe**: on-policy pure reverse-KL OPD collapses at 7B exactly as at 1B (p@1 0.009 vs 0.008; the same dead-zone attractor, with a delayed, more oscillatory onset at step ~152) — the instability is scale-robust. What scale *does* change: the per-token clip alone now converts dead pure on-policy OPD into a live recipe (0.009 → 0.323, same-teacher within-7B contrast), the on↔off-policy gap collapses from 38–55× to 1.26× (so "on-policy is a liability" is a small-student/large-gap artifact, partially rehabilitating the post's intuition *conditional on the clip*), and the best recipe (clipped low-λ, p@1 0.661) is simultaneously the most diverse and most accurate arm — the §7.8 coverage↔precision trade dissolves. What genuinely stayed open after Exp 7 was the **scale vs capacity-gap-closure confound** (Exp 7 made the student bigger *and* equal-sized to its teacher at once) — now **answered by Exp 7c's A/C arms** (7B ← **13B**-Instruct; jobs `115136`/`115137`): the confound resolves toward *scale/gap-robust instability, not gap-closure*. Re-opening the capacity gap does not prevent the unclipped collapse (A13 still dives to the dead zone, escaping only unstably), and the clip again stabilizes (C13 p@1 0.403) — so the unclipped on-policy reverse-KL collapse now reproduces across 1B←7B, 7B←7B *and* 7B←13B, with the clip as the structural fix at every gap. The low-λ arm D13 is again the best arm at this gap (p@1 **0.746**, job `121947`, vs the 7B←7B 0.661) and simultaneously the most diverse (tok_ent 1.29), so the §7.8 coverage↔precision trade dissolves under the re-opened gap as well; and the more-capable 13B teacher *lifts the stable clipped recipes* (C 0.323→0.403, D 0.661→0.746) while leaving the unstable unclipped arm A dead — teacher capability pays off only once the clip stabilizes the on-policy signal. (The 1B clipped-pure-OPD ← Instruct hedge-closer, Exp 7b, has separately run at p@1 0.09–0.15, a live 1B student — clip's payoff is not contingent on the teacher being SFT vs Instruct.)

**Reproducibility.** The harness is self-contained under `research/` (the RL/PG reference is vendored as `research/policy_gradients/`; no reach-back into other trees). Every corner and interior point reproduces from a config: `python -m harness.unified_trainer --config harness/configs/<corner>.yaml [--set key=value]`; every pass@k number from `python -m harness.eval_passk --ckpt <ckpt> --task gsm_symbolic --temps 0.6 --eval-seed 1000000`. Configs and SLURM launchers are under `harness/configs/` and `harness/run_*.sh`; the geometry analyses (Δθ sparsity, prune sweeps, effective rank, KL-signal traces) under `harness/{delta_theta_snapshot,prune_dtheta_eval,effrank_all_tensors,extract_kl_signal_traces}.py` with figures in `figs/`; experiment tracking is W&B (`WANDB_PROJECT=distill-harness`). Models are the OLMo-2 family (shared tokenizer across 1B/7B/13B/32B), so all same-family teacher setups are tokenizer-clean.

---

## Citation

```
@misc{<...>2026opd-distributional,
  title  = {On-Policy Distillation Through a Distributional Lens: which objective puts mass where},
  author = {<...>},
  year   = {2026},
  note   = {research/RESULTS.md; replicates and extends Brown & Claude, "SFT, RL, and On-Policy
            Distillation Through a Distributional Lens" (2026). Code: research/harness/.}
}
```

_Acknowledgments: the vendored `policy_gradients/` reference is by Zafir Stojanovski (Apache 2.0), adapted for the RLHF Book by Nathan Lambert. Writeup format after [Thinking Machines, "On-Policy Distillation"](https://thinkingmachines.ai/blog/on-policy-distillation/)._
