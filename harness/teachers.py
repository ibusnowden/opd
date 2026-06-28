"""Teacher registry — the pi_T side of the (alpha, lambda, pi_T) meta-algorithm.

Every teacher exposes ONE thing the trainer needs: per-token log-probabilities of the *student's*
sampled tokens under pi_T (so we can form the reverse-KL advantage  log pi_T - log pi_theta).
That uniformity is the whole point — OPD / OPSD / SDFT / hint-writer-OPD differ only in `pi_T`.

Implemented as scaffolding: the orchestration (loading the teacher model, conditioning its context
on privileged info, calling the hint-writer) is stubbed with TODO / NotImplementedError. The
`token_logprobs` signature and the small pieces that are pure tensor ops are real.

See: ../meta-algorithm-alpha-lambda.md, ../hint-writer-rl.md, ../hint-rewriter-distillation.md,
     ../per-token-kl-pivot-vs-style.md, ../cross-family-teacher-tax.md.
"""

from __future__ import annotations

import abc
from typing import Any

import torch

from . import _pg  # vendored policy_gradients helpers (load_model, compute_log_probs, ...)
from .config import TeacherSpec


class Teacher(abc.ABC):
    """Abstract teacher policy pi_T."""

    #: True if the teacher needs the student's *generated text* re-tokenized in another vocab
    #: (cross-family) — then token-level KL is not directly definable. See cross-family-teacher-tax.md.
    requires_tokenizer_match: bool = True

    @abc.abstractmethod
    def token_logprobs(
        self,
        sequence_ids: torch.Tensor,      # (B, S) prompt+completion in the STUDENT's vocab
        attention_mask: torch.Tensor,    # (B, S)
        action_mask: torch.Tensor,       # (B, S-1) which positions are generated tokens
        *,
        entries: list[dict[str, Any]] | None = None,   # raw dataset rows (for privileged-info conditioning)
        student_logprobs: torch.Tensor | None = None,  # (B, S-1) — some teachers need them (clipping etc.)
    ) -> torch.Tensor:
        """Return (B, S-1) log pi_T(token_t | prefix_<t) aligned with the student's targets."""
        raise NotImplementedError

    def maybe_refresh(self, student_state_dict: dict | None) -> None:
        """Hook for teachers that track the student (no-op by default; OPSD fixes at init)."""
        pass


# --- corners -----------------------------------------------------------------

class NoTeacher(Teacher):
    """lam == 0: there is no teacher term. Present only so the trainer can be uniform."""

    def token_logprobs(self, sequence_ids, attention_mask, action_mask, *, entries=None, student_logprobs=None):
        # Caller must not invoke this when lam == 0.
        raise RuntimeError("NoTeacher.token_logprobs called — teacher term is disabled (lam==0).")


class DatasetTeacher(Teacher):
    """pi_T = delta_{y_data}: the demonstration tokens get probability 1, everything else 0.

    Plugged in with alpha == 0 (the data is fixed, off-policy), this makes the reverse-KL term
    reduce to cross-entropy on the demonstration — i.e. plain SFT (see meta-algorithm-alpha-lambda.md,
    "SFT is distillation from a degenerate teacher"). In practice the trainer special-cases this
    (it's just NLL on the dataset), so `token_logprobs` here returns a one-hot-style log-prob
    proxy: 0.0 on the dataset token, a large negative elsewhere.
    """

    def __init__(self, neg_inf: float = -30.0) -> None:
        self._neg_inf = neg_inf

    def token_logprobs(self, sequence_ids, attention_mask, action_mask, *, entries=None, student_logprobs=None):
        # When the "rollout" IS the dataset completion, every target token is the demonstrated one,
        # so log pi_T(target) == 0 on generated positions. (The off-positions never enter the loss.)
        out = torch.zeros_like(action_mask, dtype=torch.float32)
        return out


class _HFTeacher(Teacher):
    """Base for teachers backed by a HuggingFace causal LM whose tokenizer matches the student's.

    Loads the model once on `spec.device_id`, runs a no-cache forward, and reads off the
    log-prob of the student's targets — exactly like `policy_gradients.train.compute_log_probs`,
    which we reuse. Subclasses override `_build_inputs` to inject privileged-info conditioning.
    """

    def __init__(self, spec: TeacherSpec, student_model_name: str) -> None:
        self.spec = spec
        self.model_name = spec.model_name or student_model_name
        self._model = None  # lazy-loaded in `_ensure_model`
        self._frozen_state = None

    # --- model lifecycle ---
    def _ensure_model(self):
        if self._model is None:
            device = torch.device(f"cuda:{self.spec.device_id}" if torch.cuda.is_available() else "cpu")
            # TODO: a teacher is inference-only — could load 8/4-bit (cfg.fit.teacher_load_in_8bit/4bit)
            #       or hit a vLLM endpoint (cfg.fit.teacher_vllm_url) to decouple its VRAM from the trainer.
            # NOTE: `_pg.load_model` forces `tie_word_embeddings = False`; that's fine for the OLMo-2
            #       family (already untied) but would randomly-init the lm_head of a *tied* checkpoint —
            #       respect the checkpoint's tying if you ever point this at one (cf. `_load_student`).
            self._model, _tok = _pg.load_model(self.model_name, device, gradient_checkpointing=False)
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
        return self._model

    # --- privileged-info conditioning (override in subclasses) ---
    def _build_inputs(self, sequence_ids, attention_mask, entries):
        """Default: feed the student's sequence as-is (plain OPD, no conditioning)."""
        return sequence_ids, attention_mask

    def token_logprobs(self, sequence_ids, attention_mask, action_mask, *, entries=None, student_logprobs=None):
        model = self._ensure_model()
        ids, mask = self._build_inputs(sequence_ids, attention_mask, entries)
        with torch.no_grad():
            lp = _pg.compute_log_probs(model, ids, mask)  # (B, S-1) aligned to targets
        # TODO: if `_build_inputs` prepended a conditioning prefix, slice `lp` back to the
        # student's completion positions so it aligns with `action_mask`.
        return lp.to(action_mask.device)

    def maybe_refresh(self, student_state_dict):
        if self.spec.kind == "self" and not self.spec.frozen_at_init and student_state_dict is not None:
            # "self" teacher that tracks the student: load the latest weights.
            # TODO: load_state_dict into self._model (careful with DDP wrappers / device).
            raise NotImplementedError("tracking 'self' teacher refresh not implemented yet")


class SameFamilyTeacher(_HFTeacher):
    """A separate, tokenizer-matched checkpoint (e.g. a bigger sibling). Plain OPD."""

    def _build_inputs(self, sequence_ids, attention_mask, entries):
        if self.spec.condition_on:
            # TODO: prepend a system/user turn carrying the privileged info, re-tokenize,
            # extend attention_mask; return the slice indices so token_logprobs can re-align.
            raise NotImplementedError("conditioning a same_family teacher not implemented yet")
        return sequence_ids, attention_mask


class PrivilegedInfoTeacher(_HFTeacher):
    """pi_T = a same-tokenizer model conditioned on privileged info the student doesn't see at sampling time.

    condition_on:
      "answer"      -> OPSD (Zhao et al. 2026) / PRM-as-teacher (this codebase's Exp 5):
                       the teacher sees the ground-truth answer prepended to the user message.
      "demo"        -> SDFT (Shenfeld et al. 2026): teacher sees an expert demonstration. [stubbed]
      "whitebox_rf" -> teacher sees the reward-function signature / unit tests / etc. [stubbed]

    `condition_on="answer"` is implemented (Exp 5, "prms-as-teachers"): we build new
    chat-templated prompts for the teacher with a "Hint: the answer is ..." appended to the user
    turn, retokenize, run a teacher forward, then slice the per-token log-probs back to align with
    the ORIGINAL action_mask positions. This is canonical OPSD per Zhao et al. 2026 and
    `prms-as-teachers.md`'s (a) interpretation — "a same-family model conditioned on
    partial-progress info (an OPSD-ish self-teacher)" — without needing a separate trained PRM.

    Tokenizer / chat-template assumption: the teacher uses the *student's* tokenizer (the OLMo-2
    family shares one across 1B/7B/13B), so the completion tokens decode identically and we can
    just concatenate `new_prompt_ids + completion_ids` without retokenizing the completion.
    """

    def __init__(self, spec: TeacherSpec, student_model_name: str, student_model=None) -> None:
        super().__init__(spec, student_model_name)
        # Option to share the student's weights at init instead of loading a copy:
        self._shared_student = student_model
        self._tokenizer = None
        self._system_prompt = None  # cached on first call

    def _ensure_model(self):
        # The "snapshot the student's init weights" raise applies only when the teacher IS the
        # student (spec.model_name unset → falls back to student_model_name in _HFTeacher.__init__).
        # Setting spec.model_name to a separate checkpoint (e.g. -7B-SFT as the OPSD teacher for
        # Exp 5's answer-conditioned PRM-as-teacher) sidesteps this: just load that model.
        if self._model is None and self._shared_student is not None and self.spec.frozen_at_init \
                and self.spec.model_name is None:
            # TODO: take a frozen snapshot of the student's initial weights rather than a fresh load.
            raise NotImplementedError("frozen-at-init snapshot of the student not implemented yet")
        model = super()._ensure_model()
        if self._tokenizer is None:
            # Late import to keep teachers.py importable without HF transformers in lighter-weight test runs.
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._system_prompt is None:
            from reasoning_gym.utils import SYSTEM_PROMPTS
            self._system_prompt = SYSTEM_PROMPTS["DeepSeekZero"]
        return model

    def _build_hint(self, entry: dict[str, Any]) -> str:
        """Format the privileged info as a user-message suffix. Per `condition_on`."""
        if self.spec.condition_on == "answer":
            ans = entry.get("answer", "")
            return f"\n\nHint: the final answer is `{ans}`. Use it to guide your reasoning."
        if self.spec.condition_on == "fixed_hint":
            # Exp 8 (per-task-hint-search-gepa.md): a FIXED task-level instruction from the GEPA-style
            # search (harness.hint_search). NOT privileged — entry-independent, no per-problem leak;
            # unlike "answer" it does not spend the §8.4 unbiasedness leg.
            return f"\n\n{self.spec.fixed_hint}"
        raise NotImplementedError(
            f"PrivilegedInfoTeacher(condition_on={self.spec.condition_on!r}) not implemented."
        )

    def token_logprobs(self, sequence_ids, attention_mask, action_mask, *, entries=None, student_logprobs=None):
        """log pi_T^answer per-token on the student's tokens — the OPSD answer-conditioned teacher term.
        Thin wrapper over `_conditioned_logprobs` with the privileged-info hint (`_build_hint`)."""
        if entries is None:
            raise RuntimeError(
                "PrivilegedInfoTeacher.token_logprobs needs `entries` to build the conditioning context. "
                "The trainer must pre-compute teacher_logprobs at rollout time (where entries are in scope) "
                "and cache them in Experience.teacher_logprobs — not call this from the training inner loop."
            )
        return self._conditioned_logprobs(sequence_ids, attention_mask, action_mask, entries, self._build_hint)

    def answer_info_gain(self, sequence_ids, attention_mask, action_mask, *, entries=None):
        """PRM-reweighted OPSD self-referential signal (prms-as-teachers.md variant (c)).

        Returns `(lp_answer, g)` where:
          * `lp_answer` = log pi_T^answer per-token (identical to `token_logprobs` — the OPSD teacher term);
          * `g` = log pi_T^answer - log pi_T^no-answer per-token: the "value of knowing the answer" at each
            token, computed from the SAME 7B teacher (answer-conditioned forward minus a matched no-answer
            forward — identical context except the appended hint). g is large on content/pivot tokens the
            ground-truth answer disambiguates and ~0 on stylistic tokens, giving a dense, reward-correlated
            per-token process-importance with NO separately-trained PRM.

        Two frozen-teacher forwards (answer + no-answer); called at rollout time and cached in
        Experience, so the training inner loop pays nothing extra (cf. token_logprobs caching).
        """
        if entries is None:
            raise RuntimeError("PrivilegedInfoTeacher.answer_info_gain needs `entries` (the answer builds the signal).")
        lp_answer = self._conditioned_logprobs(sequence_ids, attention_mask, action_mask, entries, self._build_hint)
        lp_noanswer = self._conditioned_logprobs(sequence_ids, attention_mask, action_mask, entries, lambda _e: "")
        return lp_answer, (lp_answer - lp_noanswer)

    def _conditioned_logprobs(self, sequence_ids, attention_mask, action_mask, entries, hint_fn):
        """Compute teacher log pi_T per-token on the STUDENT's generated tokens, with the teacher's
        context = chat-templated prompt + `hint_fn(entry)` suffix on the user turn + the student's
        completion. Shared by `token_logprobs` (hint = privileged info) and `answer_info_gain` (which
        also calls it with `hint_fn = lambda e: ""` for the matched no-answer baseline).

        Steps per row:
          1. Slice the student's completion tokens out of sequence_ids using action_mask.
          2. Build new prompt text via chat_template with `hint_fn(entry)` appended to the user message.
             Tokenize → `new_prompt_ids` (variable length per row).
          3. Concatenate `new_prompt_ids + completion_ids` per row, left-pad to a common length
             so every batch row's LAST prompt token sits at the same column (index = P-1).
          4. Forward the teacher in one shot. Extract log-probs at indices [P-1 : P-1 + comp_len]
             per row (these are log P(comp_t | new_prompt, comp_<t)).
          5. Scatter back into the original (B, S-1) action_mask positions.
        """
        model = self._ensure_model()
        tokenizer = self._tokenizer
        teacher_device = _pg.get_model_device(model)

        B, S = sequence_ids.shape
        # 1) extract completion ids per row + lengths
        am_cpu = action_mask.bool().cpu()
        comp_ids_list: list[torch.Tensor] = []
        comp_positions_list: list[torch.Tensor] = []
        for i in range(B):
            pos = am_cpu[i].nonzero(as_tuple=True)[0]  # action_mask is (S-1); position j → seq token j+1
            if pos.numel() == 0:
                comp_ids_list.append(torch.zeros(0, dtype=torch.long))
                comp_positions_list.append(pos)
                continue
            first = int(pos[0].item()) + 1
            last = int(pos[-1].item()) + 1
            comp_ids_list.append(sequence_ids[i, first : last + 1].detach().cpu())
            comp_positions_list.append(pos)

        # 2) build new chat-templated prompts (hint appended to user turn) + tokenize
        new_prompt_texts: list[str] = []
        for entry in entries:
            hint = hint_fn(entry)
            chat = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": entry["question"] + hint},
            ]
            new_prompt_texts.append(
                tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            )
        new_prompt_tok = tokenizer(new_prompt_texts, padding=True, padding_side="left", return_tensors="pt")
        new_prompt_ids = new_prompt_tok["input_ids"]                # (B, P) left-padded
        new_prompt_attn = new_prompt_tok["attention_mask"]          # (B, P)
        P = new_prompt_ids.shape[1]
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        # 3) right-pad completions to a common max length, build full ids
        max_comp = max((c.numel() for c in comp_ids_list), default=0)
        if max_comp == 0:
            return torch.zeros((B, S - 1), dtype=torch.float32, device=action_mask.device)
        comp_pad = torch.full((B, max_comp), pad_id, dtype=torch.long)
        comp_attn = torch.zeros((B, max_comp), dtype=torch.long)
        for i, c in enumerate(comp_ids_list):
            if c.numel() > 0:
                comp_pad[i, : c.numel()] = c
                comp_attn[i, : c.numel()] = 1
        full_ids = torch.cat([new_prompt_ids, comp_pad], dim=1)               # (B, P + max_comp)
        full_attn = torch.cat([new_prompt_attn, comp_attn], dim=1)            # (B, P + max_comp)

        # 4) teacher forward
        with torch.no_grad():
            full_logp = _pg.compute_log_probs(model, full_ids, full_attn)     # (B, P + max_comp - 1)

        # 5) scatter into output shape (B, S-1) at the original action_mask positions
        out = torch.zeros((B, S - 1), dtype=torch.float32, device=action_mask.device)
        for i in range(B):
            comp_len = comp_ids_list[i].numel()
            if comp_len == 0:
                continue
            # log_probs[P-1] = log P(comp[0] | new_prompt); log_probs[P-1 + j] = log P(comp[j] | ..., comp[:j])
            comp_logp = full_logp[i, P - 1 : P - 1 + comp_len].to(action_mask.device)
            out[i, comp_positions_list[i].to(action_mask.device)] = comp_logp
        return out


class HintWriterTeacher(_HFTeacher):
    """pi_T = the student model conditioned on a hint produced by a separate hint-writer model.

    The hint-writer is a learned artifact (see ../hint-writer-rl.md, ../hint-rewriter-distillation.md):
    given the problem (+ a 'bad' big hint like the answer/a demo/whitebox info), it emits a *minimal*
    hint that nudges the teacher distribution as little as possible while still raising reward.
    Here we just hold a reference to it and stub the call.
    """

    def __init__(self, spec: TeacherSpec, student_model_name: str) -> None:
        super().__init__(spec, student_model_name)
        self._hint_writer = None  # TODO: load spec.hint_writer_model (frozen, inference-only)

    def _build_inputs(self, sequence_ids, attention_mask, entries):
        # TODO: 1) decode the problem from `entries`; 2) hint = self._hint_writer.generate(problem, bad_hint);
        #       3) build the teacher context conditioned on `hint`; 4) tokenize + return (ids, mask, offset).
        raise NotImplementedError("HintWriterTeacher input construction not implemented yet")


class PRMTeacher(Teacher):
    """A SEPARATELY-TRAINED step-level process reward model (prms-as-teachers.md variant (b)).

    Unlike variant (c)'s self-referential answer-info-gain (g_t = log pi_T^answer - log pi_T^no-answer
    from the SAME 7B teacher), this loads a PRM trained on (trajectory, per-step-correctness) pairs
    (see harness/train_prm.py) and scores each token of the student's rollout with P(step correct |
    trajectory prefix). The per-token scalar g_t is then mapped to a mass-preserving weight w_t via
    harness.distill_losses.prm_importance_weights, exactly like variant (c).

    The PRM is a 1B-SFT base + a scalar head (PRMHead) trained with BCE on step-boundary positions.
    It is FROZEN at inference (inference-only, no grad). The teacher's reverse-KL term (log pi_T^answer)
    still comes from the 7B-SFT answer-conditioned teacher (PrivilegedInfoTeacher) — this class only
    provides the per-token IMPORTANCE signal, not the teacher logits. See the rollout-time PRM block
    in unified_trainer.py (the prm_source="trained" branch).

    Why this is variant (b) and not variant (c): the self-referential answer-info-gain concentrates
    KL mass on tokens the answer disambiguates (§7.11 — sharpens the heavy tail the clip exists to
    bound, collapsing harder). A trained PRM scores PROCESS CORRECTNESS — whether each step's
    arithmetic is right — which is a different signal and may have bounded mass. The roadmap
    prediction (§8.4 item 1): no-clip+trained-PRM collapses like (c); trained-PRM+clip matches or
    beats the clipped logit-teacher baseline (A=0.204).
    """

    def __init__(self, spec: TeacherSpec, student_model_name: str) -> None:
        # NOTE: we do NOT call _HFTeacher.__init__ because the PRM checkpoint is a directory, not an
        # HF model id — self.model_name resolves to the PRM checkpoint dir (e.g.
        # harness/checkpoints/prm_step_level_s4242/), and we load the base LM from a file inside it.
        self.spec = spec
        self.checkpoint_dir = spec.model_name  # directory with prm_head.pt + base_model_name.txt
        self._model = None
        self._prm_head = None
        self._tokenizer = None

    def _ensure_model(self):
        if self._model is None:
            from pathlib import Path
            import torch as _torch
            base_name_path = Path(self.checkpoint_dir) / "base_model_name.txt"
            if not base_name_path.exists():
                raise FileNotFoundError(
                    f"PRMTeacher: {self.checkpoint_dir}/base_model_name.txt not found — "
                    "run harness.train_prm stage=train first."
                )
            base_name = base_name_path.read_text().strip()
            device = _torch.device(f"cuda:{self.spec.device_id}" if _torch.cuda.is_available() else "cpu")
            model, tokenizer = _pg.load_model(base_name, device, gradient_checkpointing=False)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self._model = model
            self._tokenizer = tokenizer
            # Load the PRM head (scalar regression on the base LM's hidden state).
            head_path = Path(self.checkpoint_dir) / "prm_head.pt"
            if not head_path.exists():
                raise FileNotFoundError(
                    f"PRMTeacher: {head_path} not found — run harness.train_prm stage=train first."
                )
            head_state = _torch.load(head_path, map_location="cpu")
            self._prm_head = _PRMHead(head_state["hidden_size"]).to(device).to(_torch.bfloat16)
            self._prm_head.load_state_dict(head_state["prm_head_state"])
            self._prm_head.eval()
            for p in self._prm_head.parameters():
                p.requires_grad_(False)
        return self._model

    def step_scores(self, sequence_ids, attention_mask, action_mask, *, entries=None):
        """Per-token PRM score g_t in [0, 1] (sigmoid of the PRM head's logit at each token).

        Returns (g_t) shape (B, S-1), aligned to action_mask positions. g_t is the per-token
        process-correctness score — large on tokens that belong to CORRECT steps, ~0.5 (sigmoid(0))
        on ambiguous/unscoreable tokens, ~0 on incorrect steps. This is the variant (b) replacement
        for variant (c)'s answer_info_gain; it feeds into prm_importance_weights the same way.

        Inference: one forward pass over the full sequence (prompt + completion), apply the PRM
        head, sigmoid. No grad.
        """
        if entries is None:
            raise RuntimeError("PRMTeacher.step_scores needs `entries` (unused but kept for API symmetry).")
        model = self._ensure_model()
        with torch.no_grad():
            outputs = model(input_ids=sequence_ids, attention_mask=attention_mask,
                            output_hidden_states=False, use_cache=False)
            hidden = outputs.last_hidden_state  # (B, S, H)
            logits = self._prm_head(hidden)      # (B, S)
            g = torch.sigmoid(logits)            # (B, S) in [0, 1]
        # Align to action_mask positions (S-1): action_mask is (B, S-1) covering tokens 1..S-1.
        # g is (B, S) covering tokens 0..S-1; shift to match action_mask's convention (token t -> position t-1).
        g_action = g[:, 1:]  # (B, S-1)
        # Mask non-action positions to 0.
        return g_action * action_mask.to(g_action.dtype)

    def token_logprobs(self, sequence_ids, attention_mask, action_mask, *, entries=None, student_logprobs=None):
        """PRMTeacher does NOT provide teacher logits — it only provides the per-token importance
        signal (step_scores). The reverse-KL logits come from the PrivilegedInfoTeacher (7B-SFT
        +answer). Calling token_logprobs on a PRMTeacher is a bug.
        """
        raise NotImplementedError(
            "PRMTeacher.token_logprobs is not implemented — the PRM provides per-token importance "
            "(step_scores), not teacher logits. The reverse-KL logits should come from a "
            "PrivilegedInfoTeacher (teacher.kind='self' + condition_on='answer')."
        )


class _PRMHead(torch.nn.Module):
    """Scalar regression head on top of the base LM's hidden state (see harness/train_prm.py::PRMHead).

    Kept in sync with train_prm.py::PRMHead — same init (zero-mean, std=0.02) so a saved checkpoint
    loads cleanly into either. The training path (train_prm.py) uses PRMHead; the inference path
    (PRMTeacher) uses _PRMHead. Both classes have identical architecture and init.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, 1, bias=True)
        torch.nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_states).squeeze(-1)  # (B, S)


# --- factory -----------------------------------------------------------------

def build_teacher(spec: TeacherSpec, *, student_model_name: str, student_model=None) -> Teacher:
    """Construct the Teacher described by `spec`."""
    kind = spec.kind
    if kind == "none":
        return NoTeacher()
    if kind == "dataset":
        return DatasetTeacher()
    if kind == "same_family":
        return SameFamilyTeacher(spec, student_model_name)
    if kind == "self":
        return PrivilegedInfoTeacher(spec, student_model_name, student_model=student_model)
    if kind == "hint_writer":
        return HintWriterTeacher(spec, student_model_name)
    if kind == "prm_trained":
        return PRMTeacher(spec, student_model_name)
    raise ValueError(f"unknown teacher kind: {kind!r}")
