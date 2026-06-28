# `harness/` — the unified (α, λ, π_T) post-training trainer (SCAFFOLD)

Skeleton implementation of the meta-algorithm in `../meta-algorithm-alpha-lambda.md`. Backbone for
proposals #6–#14 in `../SKILLS.md` (everything that needs an SFT loop, a teacher-logprob /
reverse-KL term, or an outcome-reward term — all of OPD / OPSD / SDFT / expert-RL+OPD /
PRMs-as-teachers / the hint-writer family).

**Status: RL corner + OPD runnable (single- and — for the RL corner — multi-GPU); the rest scaffolded.**
The **RL corner** — `(α=1, λ=0)`, i.e. on-policy policy gradient (GRPO / RLOO / GSPO / CISPO / Dr.GRPO /
PPO via `outcome_loss`), optional KL-to-base penalty, W&B logging — is **implemented** in
`unified_trainer.py::_run_rl_loop` (a port of the reference `policy_gradients.train.main`, built on the
`_pg.*` helpers); it runs single-GPU or **multi-GPU DDP** under `torchrun` (WORLD_SIZE>1 — DDP-wrapped
student, per-rank `DistributedSampler`, grad all-reduce, metric reduction, rank-0 logging; PPO+DDP
raises). The **`λ>0` path** — plain OPD (`teacher=same_family`), OPSD-style per-token KL clip, expert-RL+OPD
(`0<λ<1`) — is **implemented** in `unified_trainer.py::_run_distill_loop` (single-GPU only; a frozen
teacher on `cfg.teacher.device_id`, recomputed per training batch, fed to `UnifiedTokenLoss`). Still
stubbed (`NotImplementedError`): the SFT corner; off-policy mixing (`0<α<1`); `teacher=self`/`hint_writer`
privileged-info conditioning (`teachers._build_inputs`); LoRA/quant wiring (`cfg.fit.*`); DDP for the
`λ>0` path; PPO+DDP; FSDP (7B+ *student*). Has been run on hardware (8×H100 sweeps + smokes).

> **Eval hooks — implemented (`eval_passk.py` + `_maybe_eval`).** Every `cfg.eval_every` steps (and on
> the final step) the trainer rolls a held-out `reasoning_gym` prompt set, scoring
> **pass@1..k, accuracy, token-entropy, distinct-n** at the temperatures in `cfg.eval_temps`. Configs
> set `eval_every`, `eval_n_prompts`, `eval_n_samples` (k_values = [1, 2, …, n_samples]), `eval_temps`.
> Held-out prompt seed = `cfg.seed + cfg.eval_seed_offset`. The standalone CLI
> `python -m harness.eval_passk --ckpt <path> --task <reasoning_gym task> ...` re-evaluates a saved
> checkpoint post-hoc (used by `run_passk_eval_sweep.sh` / `run_teacher_eval.sh`).

**Models.** Use the **OLMo-2 family** (`allenai/OLMo-2-*`) for every experiment — it shares one
tokenizer across 1B / 7B / 13B / 32B, so same-family teacher setups (OPD / OPSD / SDFT) are clean,
no tokenizer-mismatch tax (cf. `../cross-family-teacher-tax.md`). Configs default to
`allenai/OLMo-2-0425-1B`; scale the `(teacher, student)` pair (1B←7B → 1B←13B → …) once the loop
runs end-to-end.

**Hardware: RTX 6000 Ada, 48 GB.** OLMo-2-1B full fine-tune fits with gradient checkpointing;
OLMo-2-1B student ← OLMo-2-7B-Instruct teacher (bf16, frozen) also fits (~32 GB) — no quantization
needed at that scale. See "Fitting bigger OLMo-2 on the 6000 Ada" below for 13B/32B teachers.

## Self-contained under `research/` — no external deps, no lazy imports
Everything the harness needs is here. The reference policy-gradient code (GRPO / GSPO / PPO / RLOO /
REINFORCE / CISPO, rollout, advantages, KL utils, model loading, distributed setup) is **vendored**
at `../policy_gradients/` — copied from `/project/inniang/vibe/code/policy_gradients/` (original by
Zafir Stojanovski, Apache 2.0; adapted for the RLHF Book by Nathan Lambert) with **one change: the
`mlrunx` dependency removed** — the standalone `train.py::main()` is gone (the training entry point
is `harness/unified_trainer.py`) and the `mlrunx_*` config fields are gone (logging is **Weights &
Biases**, `harness/wandb_logging.py`). `loss.py` / `buffer.py` / `utils.py` are verbatim; `config.py`
and `train.py` carry those trims. `harness/_pg.py` re-exports the pieces we use in one place — plain
**eager** `from policy_gradients import ...` (no `sys.path` shim, no lazy `train_helpers()`); `harness.*`
modules import from `harness._pg`.

| from the vendored `policy_gradients` (via `harness._pg`) | used for |
|---|---|
| `loss` — `approx_kl` (k3), `masked_mean`, `GRPOLoss`/`GSPOLoss`/`ReinforceLoss`/`CISPOLoss`/`PPOLoss` | the outcome-reward (λ→0) corner; KL utilities reused verbatim |
| `buffer` — `Experience`, `ReplayBuffer`, `join_experiences_batch` | rollout storage / batching |
| `config` — `Config`, `DataConfig`, `DatasetSpec` | base config; `harness.config.ResearchConfig` subclasses `Config` |
| `train` (helpers) — `rollout`, `compute_log_probs`/`compute_values`/`compute_rewards`/`compute_advantages`/`compute_gae`/`apply_reward_kl`, `get_loss_objective`, `load_model`/`get_ref_model`/`get_val_model`, `create_dataset`, `iter_training_batches`, `setup_distributed`/`cleanup_distributed`, `unwrap_model`/`get_model_device`/`seed_everything` | rollout + log-prob + advantage + model/dist plumbing |

> Deps the vendored code pulls in (already in the venv): `torch`, `transformers`, `reasoning_gym`,
> `rich`, `numpy`, `pydantic`, `pyyaml`, `pynvml`. For logging: `pip install wandb` + `wandb login`
> (the harness no-ops cleanly if `wandb` is absent). For LoRA/quant: `pip install peft bitsandbytes`
> (peft is present; bitsandbytes is not yet — only needed when you turn on `cfg.fit.*` quantization).

## Layout
```
research/
  policy_gradients/   # VENDORED reference (mlrunx removed): __init__, loss, buffer, config, utils, train(helpers)
  harness/
    _pg.py            # eager re-exports from policy_gradients.* (single hub; no sys.path tricks)
    wandb_logging.py  # thin W&B wrapper (init/log/log_params/finish); no-ops if wandb absent/offline/non-main
    config.py         # ResearchConfig(Config): + alpha, lam, teacher spec, per_token_kl_clip, fit (mem knobs), wandb_*
    teachers.py       # Teacher ABC + NoTeacher / DatasetTeacher / SameFamilyTeacher / PrivilegedInfoTeacher / HintWriterTeacher
    distill_losses.py # sft_ce_loss, reverse_kl_distill_advantage, per_token_kl_clip, UnifiedTokenLoss
    unified_trainer.py# main(cfg): the (α, λ, π_T) loop — corners reproduce SFT/RL/OPD/OPSD
    run_research.sh   # SLURM (bigTiger) / local launcher: (torch.distributed.run) -m harness.unified_trainer
    configs/
      sft.yaml        # corner: (alpha=0, lam=1, teacher=dataset)            → cross-entropy SFT
      rl_grpo.yaml    # corner: (alpha=1, lam=0)                             → GRPO
      opd.yaml        # corner: (alpha=1, lam=1, teacher=same_family)        → on-policy distillation
      opsd.yaml       # corner: (alpha=1, lam=1, teacher=self+answer, clip)  → OPSD
```

## How to run the proof-of-life (RL corner)
```bash
cd /project/inniang/research
pip install wandb && wandb login                # one-time (harness no-ops if wandb is absent)
# the reasoning_gym rollout calls tokenizer.apply_chat_template — use an instruct checkpoint:
python -m harness.unified_trainer --config harness/configs/rl_grpo.yaml \
       --set model_name=allenai/OLMo-2-0425-1B-Instruct
# quick sanity run (a handful of steps), then watch `reward` move:
python -m harness.unified_trainer --config harness/configs/rl_grpo.yaml \
       --set model_name=allenai/OLMo-2-0425-1B-Instruct --set num_steps=5
# OPD (λ=1, same-family teacher) — single GPU, ~30 GB (1B student + 7B bf16 teacher):
python -m harness.unified_trainer --config harness/configs/opd.yaml --set num_steps=5
# multi-GPU DDP (RL corner only), one run across N GPUs:
python -m torch.distributed.run --standalone --nproc_per_node=8 \
       -m harness.unified_trainer --config harness/configs/rl_grpo.yaml \
       --set model_name=allenai/OLMo-2-0425-1B-Instruct
# or via SLURM (8×H100 itiger01):
sbatch harness/run_h100_ddp.sh                                   # one DDP run, all 8 GPUs
sbatch harness/run_h100_sweep.sh                                 # 8 independent RL runs (outcome-loss sweep)
sbatch harness/run_opd_lambda_sweep.sh                           # 8 independent OPD runs (λ sweep)
WANDB_PROJECT=distill-harness sbatch harness/run_research.sh harness/configs/rl_grpo.yaml   # single config, NPROC>1 -> torchrun
```
> `--set KEY=VALUE` overrides a config key (repeatable; **dotted keys nest** —
> `--set teacher.model_name=allenai/OLMo-2-1124-7B-SFT`, `--set teacher.kind=none`). `rl_grpo.yaml`
> defaults to `allenai/OLMo-2-0425-1B`; if that base checkpoint has
> no chat template the loop fails fast with a clear message — switch to `…-1B-Instruct` (the launchers
> do this for you). Multi-GPU DDP works for the RL corner (`torchrun` / `NPROC>1` / `run_h100_ddp.sh`);
> PPO+DDP and DDP for the `λ>0` path raise — run those with NPROC=1.

## Corners
| corner | (α, λ, π_T) | status | models | matches |
|---|---|---|---|---|
| RL | (1, 0, —) | **implemented** (`_run_rl_loop`; single-GPU + DDP, PPO single-GPU only) | OLMo-2-1B | the vendored `policy_gradients` GRPO/RLOO loop |
| SFT | (0, 1, δ_dataset) | TODO (`train_sft`) | OLMo-2-1B | a vanilla cross-entropy SFT run |
| OPD | (1, 1, bigger same-family teacher) | **implemented** (`_run_distill_loop`; single-GPU) | student OLMo-2-1B-Instruct ← teacher OLMo-2-7B-Instruct (→ 13B/32B) | Thinking-Machines / Qwen3-report OPD (OLMo-2 for tokenizer match) |
| expert RL + OPD | (1, 0<λ<1, same-family) | **implemented** (`_run_distill_loop`; `loss = λ·L_teacher_REINFORCE + (1-λ)·L_outcome_clipped` — blended at the loss level with the proper clipped GRPO/RLOO/etc. objective; PPO+λ>0 raises) | OLMo-2-1B-Instruct ← OLMo-2-7B-Instruct | DeepSeek-V4-style (`../expert-rl-plus-opd.md`) |
| OPSD | (1, 1, self conditioned on the answer, + per-token KL clip) | partial — loss/clip wired, `teacher=self` privileged-info conditioning stubbed (`teachers._build_inputs`) | OLMo-2-1B (loaded twice: trainable + frozen) | Zhao et al. 2026 |

Interior `(α, λ) ∈ (0,1)` and learned-teacher (`HintWriterTeacher`) settings are the research
surface — see `../meta-algorithm-alpha-lambda.md`, `../hint-writer-rl.md`, `../expert-rl-plus-opd.md`,
`../prms-as-teachers.md`.

## Fitting bigger OLMo-2 on the 6000 Ada (48 GB)

OLMo-2 sizes: **1B / 7B / 13B / 32B**. Rough VRAM, per role (bf16; activations on top — gradient
checkpointing, on by default, keeps those small):

| role | precision | 1B | 7B | 13B | 32B |
|---|---|---|---|---|---|
| **frozen teacher** (forward only — prefill over the student's tokens, tiny KV cache) | bf16 ≈ 2 B/param | ~2 GB | ~14 GB | ~26 GB | ~64 GB |
| | 8-bit | ~1 GB | ~7 GB | ~13 GB | ~32 GB |
| | 4-bit (NF4) | ~0.7 GB | ~4 GB | ~7 GB | ~18 GB |
| **trainable student**, full FT + AdamW (params 2 + grads 2 + Adam fp32 8 + master fp32 4 ≈ 16 B/param) | bf16 mixed | ~16–20 GB | ~110 GB+ | — | — |
| **trainable student**, **QLoRA** (NF4 base + small LoRA adapters + their Adam states) | 4-bit base | ~3–5 GB | ~6–10 GB | ~10–14 GB | ~22–26 GB |

On the **48 GB 6000 Ada**:
- **OLMo-2-1B student, full FT** — fits easily with gradient checkpointing. ← the default; start here (RL corner first).
- **OPD, OLMo-2-1B ← OLMo-2-7B teacher** — 1B full-FT (~18 GB) + 7B bf16 frozen teacher (~14 GB) ≈ 32 GB → **fits**, no quantization. (`opd.yaml` is set up this way.)
- **OPD with a 13B teacher** — 13B 8-bit (~13 GB) + 1B full-FT (~18 GB) ≈ 31 GB → fits; set `fit.teacher_load_in_8bit: true` (`pip install bitsandbytes`).
- **OPD with a 32B teacher** — 32B 4-bit (~18 GB) + 1B full-FT (~18 GB) ≈ 36 GB → fits at NF4; set `fit.teacher_load_in_4bit: true`. Or `fit.teacher_vllm_url` to serve the teacher in a separate process / on another card.
- **7B+ as the *student*** — full FT won't fit (~110 GB); LoRA/QLoRA it (`fit.student_lora`, `fit.student_load_in_4bit`) → ~6–10 GB, easy.
- **OPSD** — one OLMo-2-1B loaded twice (trainable + frozen) ≈ ~20 GB → trivially fits.

**Existing mechanisms (standard; sketched as `cfg.fit` knobs in `config.py` — wiring is TODO):**
- **Gradient checkpointing** — already used by `policy_gradients.train.load_model(gradient_checkpointing=True)`; `cfg.fit.gradient_checkpointing` (default `True`).
- **LoRA / QLoRA on the student** — `peft` (`LoraConfig` + `get_peft_model`) + `bitsandbytes` 4-bit base; `cfg.fit.student_lora`, `lora_r/alpha/dropout/target_modules`, `student_load_in_4bit`. (The repo has a *task-specific* `AttentionLoRA` reference at `vibe/autoresearch/chal/ablations/train_ablation.py`; for OLMo-2 just use `peft`.)
- **Quantize the frozen teacher** — `transformers` `load_in_8bit` / `load_in_4bit` via `bitsandbytes`; `cfg.fit.teacher_load_in_8bit / teacher_load_in_4bit`. A quantized teacher is fine — it's inference-only and the proposals already treat teachers as imperfect.
- **vLLM-served teacher** — run the teacher as a vLLM endpoint (another card / quantized) and request prompt logprobs; `cfg.fit.teacher_vllm_url`. Decouples teacher VRAM from the trainer; reuses the repo's vLLM serve infra (`/project/inniang/inference/`, the hermes/pi setup).
- **8-bit optimizer** — `bitsandbytes.optim.AdamW8bit` (Adam state 8→2 B/param); `cfg.fit.optimizer_8bit`.
- **CPU offload of optimizer/params** — DeepSpeed ZeRO-Offload / FSDP CPU offload / bnb paged optimizers; `cfg.fit.cpu_offload_optimizer`. Slow; last resort.
- **FSDP / ZeRO across multiple GPUs** — `run_research.sh` already has `NPROC` for `torch.distributed.run`; FSDP wrapping is TODO.
- **flash-attention / `torch.compile`** — `policy_gradients.train` already auto-selects flash-attn-2/3 when available.
