"""ResearchConfig — extends `policy_gradients.config.Config` with the (alpha, lambda, teacher) knobs.

Reuses everything in `policy_gradients.config.Config` (model_name, sampling params, optimizer,
batch shapes, the GRPO/PPO loss hyperparams, ...) and adds:

  * `alpha`            in [0,1]  — how on-policy the sampling distribution is
                                   (1 = sample from current student; 0 = fixed dataset / off-policy buffer).
  * `lam`              in [0,1]  — fraction of the per-token advantage that comes from the teacher
                                   reverse-KL term vs. the sequence-level outcome reward.
  * `teacher`          TeacherSpec — which teacher policy, conditioned on what (see harness.teachers).
  * `per_token_kl_clip` float|None — OPSD-style per-vocab-entry KL clip on the teacher term (None = off).
  * `outcome_loss`     str       — which outcome-RL objective to use when lam < 1 (grpo/rloo/...).
  * `is_kl_clip_warmup` etc.     — left as TODO knobs; add as the OPSD/expert-RL work needs them.
  * W&B logging fields (replaces the upstream `mlrunx_*` fields).

The four "corners" (see ./configs/): SFT = (alpha=0, lam=1, teacher=dataset);
RL = (alpha=1, lam=0); OPD = (alpha=1, lam=1, teacher=same_family);
OPSD = (alpha=1, lam=1, teacher=self+answer, per_token_kl_clip set).
"""

from __future__ import annotations

import yaml
from pydantic import BaseModel, Field, model_validator

from ._pg import Config as _BaseConfig


class TeacherSpec(BaseModel):
    """How to build the teacher policy pi_T for the reverse-KL term.

    kind:
      "dataset"     — pi_T = delta_{y_data}; recovers cross-entropy SFT (use with alpha=0, lam=1).
      "none"        — no teacher (use with lam=0; pure outcome RL).
      "same_family" — a separate, tokenizer-matched checkpoint (`model_name`); plain OPD.
      "self"        — the student itself, optionally conditioned on privileged info (`condition_on`).
      "hint_writer" — pi_T = same model conditioned on a hint produced by a learned hint-writer
                      (`hint_writer_model`); see ../hint-writer-rl.md / ../hint-rewriter-distillation.md.
    condition_on:  for kind in {"self","same_family","hint_writer"}:
                   None | "answer" | "demo" | "whitebox_rf" | "fixed_hint".
                   "fixed_hint" (Exp 8, per-task-hint-search-gepa.md): the teacher sees a FIXED
                   task-level instruction string (`fixed_hint`) appended to the user turn — unlike
                   "answer" this carries NO per-problem privileged info, so it does not spend the
                   §8.4 unbiasedness leg. The string comes from the GEPA-style search
                   (harness.hint_search).
    """

    kind: str = "none"
    model_name: str | None = None          # for "same_family"
    condition_on: str | None = None        # "answer" | "demo" | "whitebox_rf" | "fixed_hint" | None
    fixed_hint: str | None = None          # for condition_on == "fixed_hint": the searched task-level hint
    hint_writer_model: str | None = None    # for "hint_writer"
    device_id: int = 0
    # frozen vs. tracking student (OPSD fixes the teacher to the initial policy):
    frozen_at_init: bool = False


class FitConfig(BaseModel):
    """Knobs for fitting bigger OLMo-2 models on a single RTX GPU.

    Defaults = "small model, full fine-tune, gradient checkpointing on" — what OLMo-2-1B wants
    on a 24 GB card. Turn these on as you scale (see ../harness/README.md "Fitting bigger OLMo-2
    on the RTX GPU"):
      * `gradient_checkpointing`        — trade compute for activation memory (passed to load_model).
      * `student_lora` / `lora_*`        — LoRA / QLoRA on the *student* (needs `peft`); the cheapest
                                           way to fine-tune a >1B student on one GPU.
      * `student_load_in_4bit`           — base in NF4 (QLoRA; needs `bitsandbytes`).
      * `teacher_load_in_8bit/4bit`      — quantize the *frozen* teacher (inference-only → quality
                                           cost is fine, and the proposals already treat teachers as
                                           imperfect). Lets a 13B/32B teacher sit next to a small student.
      * `teacher_vllm_url`               — instead of loading the teacher in-process, hit a vLLM
                                           endpoint that returns prompt logprobs (decouples teacher
                                           memory entirely; reuses the repo's vLLM serve infra).
      * `optimizer_8bit`                 — bitsandbytes AdamW8bit (Adam state 8→2 bytes/param).
      * `cpu_offload_optimizer`          — offload optimizer state to CPU (DeepSpeed ZeRO-Offload /
                                           FSDP CPU offload) — slow, last resort for full-FT of 7B.
    All wiring is TODO in load_model / build_teacher / unified_trainer; the fields define the contract.
    """

    gradient_checkpointing: bool = True
    student_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    student_load_in_4bit: bool = False        # QLoRA-style NF4 base for the student
    teacher_load_in_8bit: bool = False
    teacher_load_in_4bit: bool = False
    teacher_vllm_url: str | None = None       # e.g. "http://localhost:8000/v1" — return prompt logprobs
    optimizer_8bit: bool = False
    cpu_offload_optimizer: bool = False


class ResearchConfig(_BaseConfig):
    """`policy_gradients.config.Config` + the meta-algorithm knobs + memory/fit knobs + W&B logging.

    Model family: OLMo-2 (allenai/OLMo-2-*). The whole OLMo-2 family shares one tokenizer across
    1B / 7B / 13B / 32B, so same-family teacher setups (OPD / OPSD / SDFT) are clean — no
    tokenizer-mismatch tax (see ../cross-family-teacher-tax.md). Start at OLMo-2-1B on a single
    RTX GPU; scale the (teacher, student) pair up only once the loop runs end-to-end (use `fit`).
    """

    # OLMo-2 by default (overrides the upstream Qwen default); smallest OLMo-2 = 1B.
    model_name: str = "allenai/OLMo-2-0425-1B"

    # --- meta-algorithm knobs ---
    # NOTE: `lam` here is the meta-algorithm's λ (teacher reverse-KL weight vs. outcome reward),
    # in [0,1] — NOT PPO's GAE lambda. It *shadows* the inherited Config.lam (GAE λ); the GAE λ now
    # lives in `gae_lambda` below, and unified_trainer passes that one to compute_advantages.
    alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    lam: float = Field(default=0.0, ge=0.0, le=1.0)
    # Optional schedule for λ over training steps. See `current_lam` in unified_trainer.
    # Shapes: "const" (default, returns cfg.lam) | "step_off" | "step_on" | "linear_anneal".
    # step_off / step_on / linear_anneal: cfg.lam is the "high" value; the low is 0.
    lam_schedule: str = "const"
    # For step_off / step_on: the step at which λ flips. (step_off: λ=lam if step<step, else 0;
    # step_on: λ=0 if step<step, else lam.) For linear_anneal: anneal start.
    lam_step: int = 100
    # For linear_anneal: anneal end. λ ramps linearly from `lam` to 0 over [lam_step, lam_step_end].
    lam_step_end: int = 300
    teacher: TeacherSpec = TeacherSpec()
    per_token_kl_clip: float | None = None
    # which outcome objective to use for the (1 - lam) branch when lam < 1:
    outcome_loss: str = "grpo"
    # PPO's GAE lambda (was Config.lam upstream; renamed here to free `lam` for the meta-knob):
    gae_lambda: float = 0.95

    # importance-sampling clip for the off-policy fraction when 0 < alpha < 1
    # (PPO-style; mirrors clip_eps_lo/hi but for the alpha-mix correction):
    is_clip_lo: float = 0.2
    is_clip_hi: float = 0.2
    # size of the off-policy buffer drawn from when alpha < 1 (0 disables off-policy mixing):
    offpolicy_buffer_size: int = 0

    # --- L3 off-policy reverse-KD (§8.1 follow-up: "is on-policy-ness alone load-bearing?") ---
    # When set, the lam==1 distill loop (`_run_distill_loop`) builds its experience buffer from
    # TEACHER-sampled sequences read from this JSONL (a `harness.rft_generate` output: fields
    # templated_prompt / completion / answer / accuracy / format) instead of on-policy student
    # rollouts. The teacher and the per-token reverse-KL objective are held fixed vs on-policy OPD;
    # only the states the loss lands on change (teacher-sampled vs student-sampled). This isolates
    # on-policy-ness alone — the confound L2 (§8.1) could not remove. None = on-policy (default).
    offpolicy_teacher_states: str | None = None

    # --- PRM-reweighted OPSD (prms-as-teachers.md variant (c); RESULTS §8.4 / §9 "natural next experiment") ---
    # When set, the per-token OPSD teacher term (log pi_T^answer - log pi_theta) is *reweighted* by a
    # per-token process-importance weight w_t before entering the loss — redistributing the reverse-KL
    # mass toward causally-important (content/pivot) tokens instead of bluntly clipping the uncertain
    # outliers (the §5 taxonomy showed pure OPD dumps most |KL| mass on uncertain tokens). The headline
    # test (prms-as-teachers.md): does PRM-reweighting make OPSD safe WITHOUT the blunt per_token_kl_clip?
    prm_reweight: bool = False
    # `prm_source`: how the per-token importance g_t is derived.
    #   "answer_info_gain" — SELF-REFERENTIAL, no separately-trained PRM (the proposal's line-52 open
    #     question): g_t = log pi_T^answer(y_t) - log pi_T^no-answer(y_t), the "value of knowing the
    #     answer" at each token, from the SAME 7B teacher. Large on tokens the ground-truth answer makes
    #     more predictable (content), ~0 on stylistic tokens — a dense, reward-correlated process signal.
    #   "trained" — variant (b): a SEPARATELY-TRAINED step-level PRM (harness/train_prm.py) scores
    #     P(step correct | trajectory prefix) per token. The PRM checkpoint lives at `prm_model_path`
    #     (a directory with prm_head.pt + base_model_name.txt). The teacher's reverse-KL logits still
    #     come from the 7B-SFT answer-conditioned teacher (PrivilegedInfoTeacher); only the importance
    #     signal changes. The roadmap prediction (§8.4 item 1): no-clip+trained-PRM collapses like
    #     variant (c); trained-PRM+clip matches or beats the clipped logit-teacher baseline (A=0.204).
    prm_source: str = "answer_info_gain"
    # `prm_model_path`: directory containing the trained PRM checkpoint (prm_head.pt + base_model_name.txt
    # + tokenizer). Required when prm_source="trained"; built by `python -m harness.train_prm stage=train`.
    prm_model_path: str | None = None
    # `prm_weight_fn`: map per-token importance g_t -> a mass-preserving multiplicative weight w_t
    # (mean(w_t) over a sequence's action tokens == 1, so reweighting REDISTRIBUTES the KL step without
    # changing its overall magnitude — keeps arm-vs-arm comparisons about *where* mass lands, not *how
    # much*). "softmax": w = n_act * softmax(g / prm_temperature). "linear": standardize g then shift to
    # mean 1, clamp >= 0, renormalize. See harness.distill_losses.prm_importance_weights.
    prm_weight_fn: str = "softmax"
    prm_temperature: float = 1.0           # softmax sharpness (higher = flatter / closer to uniform)
    prm_weight_ceiling: float | None = None  # optional per-token cap on w_t after normalization (None = uncapped;
                                             # a cap is itself a soft clip, so leave None for the "no blunt clip" arm)

    # --- memory / "fit it on the RTX" knobs ---
    fit: FitConfig = FitConfig()

    # --- eval hook (harness.eval_passk: pass@k / accuracy / diversity on a held-out reasoning_gym set) ---
    eval_every: int = 0                    # run the eval every N training steps (and on the last step); 0 = off
    eval_n_prompts: int = 64               # held-out prompts per eval
    eval_n_samples: int = 16               # completions sampled per prompt (k_values = powers of 2 up to this)
    eval_temps: list[float] = [0.6]        # decoding temperature(s) for the in-loop eval (the standalone CLI sweeps more)
    eval_seed_offset: int = 1_000_000      # eval prompt set uses seed = cfg.seed + this (≠ the training set)

    # --- checkpoint saving (needed for post-hoc pass@k eval and for #5 sparse-vs-dense) ---
    save_ckpt: bool = False                # save_pretrained the final student (rank 0)
    ckpt_dir: str | None = None            # default: harness/checkpoints/<wandb_run_name or recipe_seedN>

    # --- W&B logging (replaces mlrunx_project_id / mlrunx_run_name) ---
    wandb_project: str | None = None       # falls back to $WANDB_PROJECT, then "distill-harness"
    wandb_run_name: str | None = None      # auto: f"{recipe}_seed{seed}"
    wandb_group: str | None = None
    wandb_tags: list[str] = []

    @model_validator(mode="after")
    def _validate_meta(self) -> "ResearchConfig":
        if self.lam == 0.0 and self.teacher.kind not in {"none"}:
            # not fatal, but almost certainly a mistake — a teacher is configured but unused.
            raise ValueError("lam == 0 but teacher.kind != 'none': the teacher term is off; "
                             "set teacher.kind='none' or lam>0.")
        if self.lam > 0.0 and self.teacher.kind == "none":
            raise ValueError("lam > 0 requires a teacher: set teacher.kind to "
                             "'dataset' | 'same_family' | 'self' | 'hint_writer'.")
        if self.teacher.kind == "dataset" and self.alpha != 0.0:
            raise ValueError("teacher.kind=='dataset' (cross-entropy SFT) requires alpha==0.")
        if self.teacher.kind == "same_family" and not self.teacher.model_name:
            raise ValueError("teacher.kind=='same_family' requires teacher.model_name.")
        if self.teacher.kind == "hint_writer" and not self.teacher.hint_writer_model:
            raise ValueError("teacher.kind=='hint_writer' requires teacher.hint_writer_model.")
        if self.teacher.condition_on == "fixed_hint" and not self.teacher.fixed_hint:
            raise ValueError("teacher.condition_on=='fixed_hint' requires teacher.fixed_hint "
                             "(the searched task-level hint string; see harness.hint_search).")
        if not (0.0 <= self.alpha <= 1.0 and 0.0 <= self.lam <= 1.0):
            raise ValueError("alpha and lam must each be in [0, 1].")
        if self.offpolicy_buffer_size < 0:
            raise ValueError("offpolicy_buffer_size must be >= 0.")
        if self.offpolicy_teacher_states is not None:
            if self.lam != 1.0:
                raise ValueError(
                    "offpolicy_teacher_states is only wired for lam==1 (pure off-policy reverse-KD); "
                    "a 0<lam<1 off-policy blend would need a PPO-style IS correction (not implemented)."
                )
            if self.teacher.kind != "same_family":
                # Off-policy revKD currently supports only the stateless same_family teacher: the
                # off-policy buffer stores teacher_logprobs=None and the trainer recomputes them with
                # entries=None per micro-batch. An entries-aware teacher (kind='self'/OPSD,
                # 'hint_writer') would raise there — off-policy OPSD needs entries threaded into
                # _fill_offpolicy_buffer (the JSONL rows carry question/answer) and cached, like the
                # on-policy entries-aware path. Left as a future extension.
                raise ValueError(
                    "offpolicy_teacher_states is only wired for teacher.kind=='same_family' "
                    "(stateless logit teacher); entries-aware off-policy (OPSD/hint_writer) not implemented."
                )
        if self.prm_reweight:
            # The importance signal needs an entries-aware teacher. variant (c) uses the
            # self-referential answer-info-gain (needs the OPSD answer-conditioned teacher);
            # variant (b) uses a separately-trained PRM (needs a prm_model_path checkpoint).
            if self.prm_source not in {"answer_info_gain", "trained"}:
                raise ValueError(
                    f"prm_reweight: prm_source={self.prm_source!r} not implemented; only 'answer_info_gain' "
                    "(variant c, self-referential) and 'trained' (variant b, separately-trained PRM) are wired."
                )
            if self.prm_source == "answer_info_gain":
                if not (self.teacher.kind == "self" and self.teacher.condition_on == "answer"):
                    raise ValueError(
                        "prm_reweight (answer_info_gain) requires the OPSD answer-conditioned teacher: "
                        "teacher.kind='self' + condition_on='answer'."
                    )
            elif self.prm_source == "trained":
                if not self.prm_model_path:
                    raise ValueError(
                        "prm_reweight (prm_source='trained') requires prm_model_path "
                        "(a directory with prm_head.pt + base_model_name.txt; built by "
                        "`python -m harness.train_prm stage=train`)."
                    )
                # The teacher still provides the reverse-KL logits — variant (b) keeps the 7B-SFT
                # answer-conditioned teacher (PrivilegedInfoTeacher) and only swaps the importance
                # signal. So teacher.kind='self' + condition_on='answer' is still required.
                if not (self.teacher.kind == "self" and self.teacher.condition_on == "answer"):
                    raise ValueError(
                        "prm_reweight (prm_source='trained') still needs the OPSD answer-conditioned "
                        "teacher for the reverse-KL logits: teacher.kind='self' + condition_on='answer'. "
                        "The trained PRM only replaces the importance signal, not the teacher logits."
                    )
            if self.lam != 1.0:
                raise ValueError(
                    "prm_reweight is wired for lam==1 (pure PRM-reweighted OPSD); a 0<lam<1 blend would "
                    "reweight only the teacher branch and is left as a future extension."
                )
            if self.offpolicy_teacher_states is not None:
                raise ValueError("prm_reweight is on-policy OPSD only; not compatible with offpolicy_teacher_states.")
            if self.prm_weight_fn not in {"softmax", "linear"}:
                raise ValueError(f"prm_weight_fn must be 'softmax' or 'linear', got {self.prm_weight_fn!r}.")
        return self

    # convenience: a short label for the (alpha, lam, teacher) corner / point
    @property
    def recipe(self) -> str:
        a, l, tk = self.alpha, self.lam, self.teacher.kind
        if a == 0.0 and l == 1.0 and tk == "dataset":
            return "sft"
        if a == 1.0 and l == 0.0:
            return f"rl_{self.outcome_loss}"
        if a == 1.0 and l == 1.0 and tk == "same_family":
            return "opd"
        if a == 1.0 and l == 1.0 and tk == "self" and self.teacher.condition_on == "answer":
            return "opsd_prm" if self.prm_reweight else "opsd"
        if a == 1.0 and tk == "self" and self.teacher.condition_on == "fixed_hint":
            return "hint_opd" if l == 1.0 else f"hint_opd_l{l:g}"
        cond = f"+{self.teacher.condition_on}" if self.teacher.condition_on else ""
        return f"meta_a{a:g}_l{l:g}_{tk}{cond}"


def load_config(path: str) -> ResearchConfig:
    """Load a ResearchConfig from a YAML file (same shape as upstream `load_config`)."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return ResearchConfig(**raw)
