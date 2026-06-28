# research/ — OPD through a distributional lens

Replication + extension of Brown & Claude, *"SFT, RL, and On-Policy Distillation Through a Distributional Lens"* (2026). The post writes SFT / RL / OPD as three points in one `(α, λ, π_T)` token-level policy gradient and asks which corner — or which interior point — puts probability mass where. This repo executes that program: a unified trainer that reaches the corners and the interior, plus five experiments and cross-cutting controls mapping the space.

**Status (2026-06-11):**
- **Exp 1 — teacher recipe + off-policy SFT control**: done + diagnosed. Same-family SFT / DPO / Instruct 7B teachers all give similarly weak pure-OPD students; teacher recipe is *not* the load-bearing variable at this gap. The §8.1 L2 control is also done: off-policy SFT on Instruct-teacher rollouts reaches p@1 **0.388** filtered / **0.297** unfiltered vs on-policy pure-OPD **0.005**.
- **Exp 2 — per-token KL taxonomy**: done offline. Pure OPD puts most |KL| mass on uncertain high-entropy tokens; GRPO shifts mass toward content; per-token clipping suppresses the uncertain bucket.
- **Exp 3 — sparse-vs-dense updates**: done end-to-end (static + dynamic + spectral). See "What Exp 3 found" below.
- **Exp 4 — the (α, λ) interior**: done end-to-end (in-distribution + cross-task + gen-gap). See "What Exp 4 found" below.
- **Exp 5 — PRMs as teachers / answer-conditioned OPSD**: pilot + taxonomy + multi-seed done. Answer-conditioning rescues pure OPD from the dead logit-teacher corner but does not beat the low-λ clipped interior.
- **Exp 6 — PRM-reweighted OPSD**: done, **negative** (§7.11). Self-referential answer-info-gain reweighting cannot replace the per-token clip (no-clip+reweight collapses *harder*) and slightly hurts stacked on it — the clip is a structural stabilizer, not an ad-hoc instrument.
- **Exp 7 — the scale test (§7.12)**: done 2026-06-11. Native 7B-SFT student ← 7B-Instruct teacher: the unclipped on-policy collapse is **scale-robust** (p@1 0.009, delayed onset), the clip alone rescues pure on-policy OPD at 7B (0.323, 37× same-teacher), the on↔off-policy gap shrinks 38–55× → 1.26×, and the clipped low-λ recipe (0.661) is best on every k *while staying the most diverse arm*. Confound: scale vs capacity-gap-closure → **Exp 7b** (1B clipped-pure-OPD ← Instruct, job `115135`) and **Exp 7c** (7B ← 13B-Instruct, jobs `115136/115137/115138`) launched 2026-06-11. **Exp 7c update (A/C done 2026-06-12): the confound resolves toward *scale/gap-robust instability, not gap-closure*** — re-opening the gap to a 13B teacher does *not* prevent the unclipped collapse (A13 still dives to the dead zone, escaping only unstably; final-checkpoint p@1 0.237 is a snapshot of a broken run), while the clip stays stable (C13 p@1 0.403). The collapse now reproduces across 1B←7B, 7B←7B *and* 7B←13B. **Exp 7c complete (D done 2026-06-20):** low-λ arm **D13 is the best arm — p@1 0.746** (job `121947`; vs 7B←7B 0.661) and also the highest-entropy arm (the §7.8 trade dissolves at this gap too); the more-capable 13B teacher *lifts the stable clipped recipes* (C 0.323→0.403, D 0.661→0.746) but *cannot rescue the unstable unclipped arm* A — teacher capability pays off only once the clip stabilizes the on-policy signal. **Exp 7b** ran at p@1 0.09–0.15 (clip gives a live 1B student under an Instruct teacher).

Full writeup with measured tables, figures, and discussion: **[`RESULTS.md`](RESULTS.md)**. Proposals: the per-`.md` files in this directory (`opd-different-teachers.md`, `meta-algorithm-alpha-lambda.md`, `sparse-vs-dense-updates.md`, …). High-level priorities: [`roadmap.md`](roadmap.md).

## What Exp 4 found (§7, the (α, λ) interior)

The headline single-number recipe is `(α=1, λ=0.10, per_token_kl_clip=1.0)`:

- **In-distribution `gsm_symbolic`** (4-seed mean): p@1 = **0.693**, p@16 = **0.801** — narrowly beats the 3-seed GRPO-v2 reference (0.687 / 0.786). _Source: SLURM `71271` + `71395`._
- **Out-of-distribution `simple_equations`** (T=0.6, 4-seed mean): **pass@k crossover** — GRPO wins p@1–p@16 by ~2× (0.203 vs 0.107) but **clipped λ=0.10 overtakes at p@32 (0.565 vs 0.549) and p@64 (0.666 vs 0.607)**. The interior is *crossover-dominant*, not strictly dominant: trades p@1 precision for p@k coverage off-task. _Source: SLURM `71574`._
- **Mechanism (§8.2)**: unclipped low-λ trainers drive `kl_signal/p99` to 2.1–2.9 over training → rollout accuracy collapses around step 100; clipped trainers keep p99 below the |kl|=1 cap → accuracy *recovers* to 0.71±0.03. The collapse-recovery instability of unclipped expert-RL + OPD is quantitatively explained by the per-token KL signal. _Source: offline W&B binaries of SLURM `71208` + `71395`._
- **Eval-seed variance is tiny** (sd 0.006–0.014 on 0.43–0.69 means): every number in §7.4–7.8 carries ~±0.02 noise; the training-seed bimodality of unclipped v2.1 is real, not eval-seed luck. _Source: SLURM `72040`._

Open headline questions after the June controls: L3 off-policy reverse-KD vs on-policy OPD isolation, OPSD cross-task eval, answer-conditioned teacher as diagnostic teacher, and hint-writer co-evolution.

## What Exp 3 found (§6, sparse-vs-dense updates)

The 11-checkpoint Δθ snapshot + 40-eval prune sweep + per-layer effective-rank pass tell a single geometric story across **static** (sparsity proxies), **dynamic** (prune-degradation curves), and **spectral** (effective rank of ΔW) measurements.

**Static (§6, 11 ckpts).** All α=1 arms — RL baseline, GRPO, clipped λ-interior, **even pure OPD λ=1.0** — sit in the same sparse-update regime (top-1% mass 0.57–0.63, top-5% mass 0.91–0.94, only 6–10% of weights moved by >1e-4). The predicted "SFT-dense for OPD-from-SFT-teacher" pattern **does not appear**. Two visible tiers within the sparse regime, mostly differing in embed-touched fraction.

**Dynamic (§6.1 + §6.2, 4 ckpts × 10 prune levels = 40 evals).**
- **Bottom 50% of moves is functionally dead weight** for every healthy arm — p@1 within ±0.02 of unpruned at p=50%, which corresponds to 7–8% of total params reverted to base.
- **Broader-tier arms (RL baseline, clip λ=0.10)** have a sharp cliff in p∈[80%, 90%]: ~50% retention at p=85% → ~0 retention at p=90%.
- **Sharper-tier arm (GRPO-v2-s42)** degrades **gradually**: still 0.439 p@1 at p=90% (~81% retention), 0.266 at p=95% (~50% retention). No cliff.
- **Pure OPD λ=1 *improves monotonically* with pruning**: 0.047 unpruned → 0.161 at p=90% → **0.242 at p=95%** (5× the unpruned p@1). Its moves are mistargeted; reverting them undoes harm.

**Spectral (§6.3 + §6.4, 11 ckpts, effective rank of ΔW per submodule).** §6.3 (top-10-by-frob slice) initially read as "broader = lower rank than base, sharper ≈ base"; the §6.4 full per-tensor pass corrects this — the tier separation is small and submodule-specific:
- **attn_qkv:** broader 1.01, sharper **1.07** — sharper tier adds ~6% spectral capacity; broader preserves base rank.
- **embed (token embed + lm_head):** broader 1.00, sharper **0.97** — reversal; sharper tier *compresses* embeddings (consistent with §6 noting sharper tier touches only 1-3% of embed rows vs broader's 11-13%).
- **mlp_in / mlp_down:** both tiers ~1.00-1.02, no tier separation.

**Unified reading.** Sharper tier = "narrow in count" (§6 static) + "graceful degradation" (§6.1/6.2 dynamic) + "reorganizes attention + embed spectra in opposite directions" (§6.4 spectral). Broader tier = "broad in count" + "sharp cliff at p≈85%" + "value-shifts within the existing spectral basis everywhere". Pure OPD λ=1 has the highest attn_qkv ratio (1.069) AND one of the lowest embed ratios (0.957) AND improves under pruning — broad and rank-additive at attn, but mistargeted.

This **falsifies the strong post-thesis claim** at this scale ("OPD inherits update geometry from the teacher recipe"): every trainer in the α=1 family produces an RL-shaped sparse subnetwork regardless of teacher dose. What the meta-algorithm changes is *which subspace* the update lives in, not whether the update is sparse vs dense.

## What's in this directory

- **[`RESULTS.md`](RESULTS.md)** — TM-style writeup. §1–3 framing + setup; §4 Exp 1 + diagnostics; §5 Exp 2; §6 Exp 3; §7 Exp 4; §7.10 Exp 5; §7.11 Exp 6 (PRM-reweight, negative); §7.12 Exp 7 (the scale test); §8 discussion including the §8.1 off-policy SFT control + scale update; §9 conclusion. Remaining `[TO FILL]` markers are mostly intro/framing.
- **[`roadmap.md`](roadmap.md)** — high-level priorities, deep research assessment from the literature.
- Per-proposal `*.md` files (idea status, not measured): `opd-different-teachers.md`, `meta-algorithm-alpha-lambda.md`, `sparse-vs-dense-updates.md`, `entropy-collapse-opd-vs-rl.md`, `per-token-kl-pivot-vs-style.md`, `expert-rl-plus-opd.md`, `pass-at-k-vs-pass-at-1.md`, `prms-as-teachers.md`, `student-beats-teacher-opd.md`, `hint-writer-rl.md`, `hint-rewriter-distillation.md`, `co-evolving-hint-writer.md`, `per-task-hint-search-gepa.md`, `cross-family-teacher-tax.md`, `sft-rl-tipping-point.md`.
- **[`harness/`](harness/README.md)** — the unified `(α, λ, π_T)` trainer + eval scripts. See `harness/README.md` for the runnable corners, configs, hardware notes.
- **[`policy_gradients/`](policy_gradients/)** — vendored reference (GRPO/RLOO/PPO/REINFORCE/CISPO/GSPO + rollout + advantages), original by Zafir Stojanovski (Apache 2.0), adapted from `vibe/code/policy_gradients/` (mlrunx removed).
- `figs/` — measured plots; everything here is `numpy/matplotlib` output of the saved JSONs/npz, not AI-generated. Headline figures: `exp4_kl_signal_mechanism_polished.png` (§8.2), `dtheta/exp3_prune_curves_polished.png` (§6.1+6.2), `dtheta/exp3_attn_qkv_effrank_ratio.png` (§6.3), `dtheta/exp3_dtheta_by_category.png` (§6 submodule split), `dtheta/exp3_dtheta_sparsity.png` (§6 mass concentration).
- `results/` — per-eval JSON outputs from `eval_passk.py` runs (Exp 4 cross-task, gen-gap, prune sweeps).
- `rft_data/` — RFT-generation outputs from the §7.2 positive-control teacher specialization step and the §8.1 filtered/unfiltered off-policy SFT control.

## Reproducing the headline numbers

```bash
# Re-run §6 static sparsity batch on the 11 canonical ckpts (~30 min CPU):
python -m harness.delta_theta_snapshot --batch --no-effrank
python harness/plot_dtheta_summary.py

# Re-run the §6.1/6.2 prune sweep (4 ckpts × 5 prune levels each = 20 evals, ~90 min on 4×H100):
sbatch harness/run_dtheta_prune_sweep.sh
sbatch harness/run_dtheta_prune_sweep_fine.sh
python harness/plot_prune_curves_polished.py

# Re-render the §8.2 mechanism plot (post-hoc extraction from the offline W&B binaries):
python harness/extract_kl_signal_traces.py
python harness/plot_kl_signal_polished.py

# Re-run §7.8 cross-task pass@k eval (15 ckpts, ~12h on 8×H100):
sbatch harness/run_clip_lowlam_cross_task_eval.sh

# Re-run §7.9 gen-gap eval-seed robustness (9 evals, ~5h on 7×H100):
sbatch harness/run_gen_gap_eval_seeds.sh

# Re-run §8.1 off-policy SFT control (train + eval; eval-only fallback exists if needed):
sbatch harness/run_exp1_offpolicy_sft.sh
sbatch harness/run_exp1_offpolicy_eval.sh
```

Every script writes its JSON outputs alongside the SLURM logs in `harness/logs/` and `results/` or `figs/dtheta/`. Plot scripts (`plot_*.py`) load only those JSONs/npz — they fabricate nothing.

## Citation

```
@misc{<...>2026opd-distributional,
  title  = {On-Policy Distillation Through a Distributional Lens: which objective puts mass where},
  author = {<...>},
  year   = {2026},
  note   = {research/RESULTS.md; replicates and extends Brown & Claude,
            "SFT, RL, and On-Policy Distillation Through a Distributional Lens" (2026).
            Code: research/harness/.}
}
```

_Acknowledgments: the vendored `policy_gradients/` reference is by Zafir Stojanovski (Apache 2.0), adapted for the RLHF Book by Nathan Lambert. Writeup format after [Thinking Machines, "On-Policy Distillation"](https://thinkingmachines.ai/blog/on-policy-distillation/)._
