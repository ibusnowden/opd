# On-Policy Distillation From Different Teachers (SFT vs RL)

_Status: idea · Started: 2026-05-11 · Last updated: 2026-05-11_
_Source: `research/ideas.md` — §"On-Policy Distillation With Different Teachers" / §"The Explanation I Prefer: On-Policy Data". This is a replication-and-extension of the post's core experiment._

## Introduction
**Question.** When you distill a student via on-policy distillation (OPD), how much does the *teacher* matter versus the fact that the data is *on-policy* (sampled from the student)?

**Hypothesis (from the post).** On-policy data is the load-bearing ingredient. An OPD student distilled from an SFT teacher and one distilled from an RL teacher should end up in *similar* places, both can *beat* the SFT teacher, and both should *forget less* than the SFT teacher — even when that teacher is itself a degraded SFT model — because the state distribution being trained on is the student's, not the teacher's.

**Why it matters.** If true, you can "overtrain" a specialized capability cheaply (even brute-force SFT, accepting degradation) and then *recover* it into a generalist via OPD while preserving the rest of the model. That changes how you'd build expert→merge pipelines (GLM-5 / DeepSeek-V4 style).

**Prior work.** Agarwal et al. 2023 (OPD, students surpassing teachers on GSM8K); Lu & Thinking Machines Lab 2025 (OPD post); Qwen3 technical report 2025 (same-family OPD); Shenfeld et al. 2026 (on-policy data ⇒ less forgetting); Ross et al. 2010 (DAGGER / exposure bias). The "Minimal Code Editing" environment is the author's own prior post.

## Data
- **Task: Minimal Code Editing.** Model is given a buggy, corrupted function and must fix *only* the bug. Scored on (a) does the fixed function pass tests, (b) was any unrelated part rewritten vs. the original uncorrupted function.
  - Two disjoint corruption-type sets: `train_corruptions` and `eval_corruptions` → measures whether the model learned *general minimum-editing behavior* vs. *reversing specific corruptions*.
  - Stats to record on construction: # functions, # corruption types per split, avg function length (tokens), corruption "size" distribution, base-model pass rate.
- **Forgetting probe: LiveCodeBench** — general code generation; minimum-editing is niche, so degradation here is a clean catastrophic-forgetting signal.
- **Example (schematic):**
  ```
  # corrupted (off-by-one introduced)
  def first_n(xs, n): return xs[:n+1]
  # target fix: change n+1 → n, touch nothing else
  ```
- Adapter to `reasoning_gym`-style scoring (the `policy_gradients` harness already extracts/scores answers); the code-editing checker is new and would need to be written.

## Method and model
**Pipeline.** `base → {SFT teacher, RL teacher}` (independent) → `{OPD-from-SFT student, OPD-from-RL student}` (both distilled onto the *same* base, on-policy).

**Modules.**
1. **SFT teacher.** Cross-entropy on demonstration fixes for `train_corruptions`. *(SFT loop not in repo — to be built.)*
2. **RL teacher.** Policy gradient with reward = `passes_tests ∧ ¬unrelated_edit`. Reuse `vibe/code/policy_gradients/` — `loss.py::GRPOLoss`/`ReinforceLoss`, `train.py::rollout()`, `compute_advantages()`, `apply_reward_kl()`. New: the code-editing reward fn.
3. **OPD student.** On-policy sampling from the student; per-token reverse-KL toward the (frozen) teacher's logprobs over the student's tokens. *(Teacher-logprob pass + reverse-KL loss not in repo — to be built; can prototype the loss on `gpt_from_scratch/run.py` first.)*
4. **Eval harness.** Pass-rate & unrelated-edit-rate on `eval_corruptions`; LiveCodeBench pass@1; entropy over training.

**Training setup.** Same-family teacher/student from the **OLMo-2** family (shared tokenizer across OLMo-2 sizes — required for token-level KL, and the reason we use OLMo here): student `allenai/OLMo-2-0425-1B`, teacher `allenai/OLMo-2-1124-7B-Instruct` (scale to 13B/32B later). Start small on a single **RTX GPU**; scale GPU count / model size only once the loop is solid. W&B logging + SLURM driver à la `scripts/run_all_policy_gradients.sh` (`bigTiger`). Harness: `research/harness/` (the unified trainer scaffold).

**Ablations.** (a) teacher = SFT vs RL vs SFT-RS (rejection-sampled SFT); (b) on-policy vs off-policy distillation (teacher-generated trajectories) holding teacher fixed; (c) with/without explicit KL-to-base penalty; (d) student init = base vs lightly-SFT'd.

## Evaluation *(proposed — no results yet)*
| Model | Min-edit (train corr.) | Min-edit (eval corr.) | LiveCodeBench pass@1 (Δ vs base) | Train entropy trend |
|---|---|---|---|---|
| base | — | — | (ref) | — |
| SFT teacher | high | ↓ (overfits corruptions?) | ↓ (degradation expected) | n/a |
| RL teacher | high | ↑ (generalizes) | ≈ 0 | n/a |
| OPD ← SFT teacher | ? | **≥ SFT teacher** (expected) | small ↓ (expected) | sharper ↓ |
| OPD ← RL teacher | ? | **≈ OPD ← SFT** (expected) | small ↓ | sharper ↓ |

- **Headline metrics:** generalization gap (eval vs train corruptions), forgetting (LiveCodeBench Δ), student-vs-teacher delta.
- **Expected:** the two OPD students converge; both beat the SFT teacher; both forget less than the SFT teacher; off-policy ablation (b) breaks the result → confirms on-policy data is load-bearing.
- **Where it breaks:** if the SFT teacher is *so* degraded its logprobs are uninformative on the niche behavior; if same-family constraint can't be met; if the unrelated-edit checker is noisy.

## Takeaways *(predictions)*
- Likely conclusion: "the source of the data matters a lot; the teacher matters less than expected" — supporting the on-policy-data thesis and the overtrain-then-OPD recipe.
- Risks: result may be specific to this niche behavior; entropy collapse under OPD could hurt downstream diversity (see `entropy-collapse-opd-vs-rl.md`).
- Open: how degraded can the teacher be before OPD stops recovering the capability? Does SFT-RS-teacher OPD differ from SFT-teacher OPD at all?
