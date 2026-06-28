# Exp 7 — the SCALE test (FOLDED into RESULTS.md §7.12 on 2026-06-11 — this file is now historical)

> **Status: FOLDED.** This draft was integrated into RESULTS.md as §7.12 (plus the §8.1/§8.4/§9
> revisions and status-tracker rows listed at the bottom) on 2026-06-11. The canonical text now lives
> in RESULTS.md; follow-ups Exp 7b (1B clipped-pure-OPD ← Instruct, job 115135) and Exp 7c
> (7B ← 13B-Instruct) were launched the same day. — 2026-06-10/11.
>
> **TEACHER-MATCH WARNING (verified 2026-06-10).** Exp 7 uses teacher **7B-Instruct** (to reuse the
> §8.1/L3 off-policy buffer). The clean, same-teacher 1B references are **A** (on-policy pure-OPD ←
> 7B-Instruct, §8.1: 0.008) and **B** (off-policy reverse-KD ← 7B-Instruct, §8.1/L3: 0.298). BUT the
> 1B references for **C** (clipped pure-OPD ~0.05) and **D** (low-λ clip 0.709) come from the **§7 (Exp 4)
> interior + §7.7 clip work, which used π_T = 7B-SFT** (and §7.2's positive control used a
> *task-specialized* teacher) — **NOT** 7B-Instruct. So any 1B↔7B comparison for C/D is cross-teacher and
> must be hedged. The load-bearing claims below are therefore built on **within-Exp-7, same-teacher (7B-Instruct)
> contrasts** (A↔C↔B all at 7B), which need no 1B ref at all; the 1B comparisons are secondary/softer.

## Proposed section: §7.12 Experiment 7 — Does the on-policy OPD collapse persist at scale?

**Question.** Every result in this document is **1B-student / 7B-teacher**. §8.1 established that at
that capacity gap, on-policy pure reverse-KL OPD is not merely weak but *dead* (p@1 ≈ 0.008), and
§8.1-fu2 showed *student competence* does not rescue it (a warm-started 1B at p@1 0.383 is collapsed
to 0.003). The §9 conclusion explicitly defers **one** remaining out: *true scale* — a larger student
whose on-policy distribution might be stable under reverse-KL. Exp 7 runs it.

**Design.** Native **7B-SFT student ← 7B-Instruct teacher** (same OLMo-2 base, shared tokenizer; the
teacher is the *same* 7B-Instruct as §8.1/L3, so the off-policy buffer is reused). Full-FT 7B fits one
80 GB H100 via bnb 8-bit Adam (`fit.optimizer_8bit`) with the frozen teacher **co-resident** on the
student's card (`teacher.device_id=0`, the harness default). `gsm_symbolic`, 250 steps, seed 42, eval
at the §8.1/Exp-5 protocol (T=0.6, 64 prompts × 16, eval-seed 1e6). Four arms (jobs 115090/92/93/94),
each one GPU, the launcher overriding only alpha / lam / per_token_kl_clip / offpolicy buffer:

- **A** `alpha=1 lam=1 clip=null` — on-policy pure-OPD, **the** dead-at-1B arm. (KEY: revive at 7B?)
- **B** `alpha=0 lam=1 clip=null` +buffer — off-policy reverse-KD, the §8.1/L3 "alive" control.
- **C** `alpha=1 lam=1 clip=1.0` — on-policy pure-OPD **with the clip**. Does the clip rescue it at 7B?
- **D** `alpha=1 lam=0.10 clip=1.0` grpo — the best-1B recipe (clipped low-λ interior).

**Results — held-out eval @ step 500... (250), T=0.6, 64×16.**

| arm | recipe | **p@1** | p@2 | p@4 | p@8 | p@16 | tok-ent | 1B ref `[verify]` |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **A** | on-policy pure-OPD, no clip | **0.0088** | 0.017 | 0.034 | 0.063 | 0.109 | 1.20 | 0.008 (dead) |
| **C** | on-policy pure-OPD, clip=1.0 | **0.3232** | 0.421 | 0.506 | 0.581 | 0.641 | 0.39 | ~0.05* (dead) |
| **B** | off-policy reverse-KD | **0.4082** | 0.531 | 0.633 | 0.722 | 0.797 | 0.36 | 0.298 |
| **D** | on-policy low-λ clip (best) | **0.6611** | 0.742 | 0.802 | 0.848 | 0.875 | 2.76 | 0.709* |

`*` C/D 1B refs used the 7B-**SFT** / task-specialized teacher; Exp 7 uses 7B-**Instruct** (cross-teacher — see warning above). **7B ranking: D 0.661 > B 0.408 > C 0.323 > A 0.009.**

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

**Three findings.**

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
   *accuracy payoff on pure OPD* appears to grow with scale — but confirming that needs a 1B
   clipped-pure-OPD ← 7B-**Instruct** run, which we don't currently have.

3. **The on-policy↔off-policy gap is small at 7B, and §8.1's "on-policy is a liability" does not
   generalize upward.** *Clean at 7B (same teacher):* clipped on-policy (C) 0.323 vs off-policy (B) 0.408
   → off-policy wins just **1.26×** across the whole pass@k curve. Contrast §8.1, where at the 1B/7B gap
   off-policy beat *unclipped/SFT* on-policy by ~38–55×. So the catastrophic on-policy penalty is a
   **small-student/large-capacity-gap artifact**, not a scale-general law — partially rehabilitating the
   post's "on-policy is load-bearing" intuition, **conditional on the clip**. *Hedge:* the precise
   "6× → 1.26×" narrowing uses a cross-teacher 1B clipped-on-policy figure (~0.05, 7B-SFT teacher); the
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
is a **7B-student ← 13B- or 32B-teacher** run (OLMo-2 has both): if on-policy still revives-with-clip
there, absolute scale is the driver; if it reverts toward the 1B picture, capacity-gap-closure was.
Other caveats: single task (`gsm_symbolic`), single seed (42) — arm A's collapse is unambiguous so no
seed sweep is warranted for it, but C/B/D single-seed numbers carry the usual ±0.01–0.05 caution;
arm A's eval is a single checkpoint drawn from a (pre-collapse) noisy trajectory, though it landed
firmly in the dead zone so the reading is robust.

## Downstream edits this implies (for when D lands + refs verified)
- **§8.1**: the "scale remains an external-validity limit" / "needs genuine scale" hedge is now
  *answered* — add a forward-pointer to §7.12. The unclipped on-policy collapse generalizes to 7B;
  the *gap* (not the collapse) is what scale closes.
- **§8.4 / §9 "what stays open"**: move "scale" from open → answered-with-nuance (collapse scale-robust;
  gap scale-dependent; clip scale-invariant). New open item: disentangle scale vs capacity-gap
  (7B←13B/32B).
- **Status tracker**: add Exp 7 row.
- **Memory**: new note `research_exp7_scale` + MEMORY.md pointer.
