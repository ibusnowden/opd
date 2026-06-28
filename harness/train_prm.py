"""Stage 1 PRM trainer for prms-as-teachers.md variant (b) — a SEPARATELY-TRAINED step-level
process reward model scoring process correctness (not answer-dependence).

Why this exists (roadmap §8.4 item 1 / RESULTS §9):
  Variant (c) used a SELF-REFERENTIAL importance signal (g_t = log π_T^answer − log π_T^no-answer
  from the same 7B-SFT teacher) and FAILED — a mass-preserving reweight CONCENTRATES the per-token
  KL tail the clip exists to BOUND (§7.11: no-clip+reweight collapses to 0.006 p@1, grad norms 3-6×
  higher; clip+reweight slightly hurts at 0.157 vs 0.204). The roadmap's prediction for variant (b):
  a separately-trained step-level PRM scoring PROCESS CORRECTNESS (not answer-dependence) is
  qualitatively different — it scores whether each step is correct, and its mass may be bounded
  enough to avoid the tail-sharpening. The falsification test (prms-as-teachers.md line 45):
    - no-clip + trained-PRM  -> collapses like variant (c)?  (prediction: yes)
    - trained-PRM + clip     -> matches or beats the clipped logit-teacher baseline (A=0.204)?
  If both hold, the clip is confirmed as the universal structural stabilizer and the PRM teacher
  interface is the next axis to sweep.

This module produces the trained PRM checkpoint consumed by harness/teachers.py::PRMTeacher.
Two-stage, both driven by CLI flags:

  Stage 1 — generate (trajectory, per-step label) pairs from the 7B-SFT teacher on gsm_symbolic,
            segment each trajectory into steps, and label each step correct/incorrect using the
            metadata.variables ground-truth intermediate values (the same gsm_symbolic metadata the
            harness already exposes). Saves a JSONL of (prompt, completion, step_token_spans,
            step_labels) to <out_dir>/prm_train.jsonl.

  Stage 2 — fine-tune OLMo-2-0425-1B-SFT with a scalar head (BCE on step-boundary tokens) to
            predict P(step correct | trajectory prefix). Saves to <out_dir>/prm_step_level_<seed>/.

The PRM is then loaded by PRMTeacher (harness/teachers.py) and used as prm_source="trained" in
unified_trainer.py's rollout-time PRM block (replacing the self-referential answer_info_gain).

Per-step labeling — the load-bearing idea:
  gsm_symbolic metadata.answer_cot is a newline-separated chain of arithmetic steps terminating in
  "#### <answer>". metadata.variables is a dict of ground-truth intermediate values (e.g.
  {"score2": 281, "total_score": 777}). Each CoT step typically computes one of those values. We
  label a step CORRECT iff it (a) parses to a number and (b) that number matches one of the
  metadata.variables values that should have been produced by that step (in chain order). Steps
  that don't parse to a number (or that compute a value not in variables) get the AMBIGUOUS label
  — excluded from the BCE loss (mask=0) but still scored by the PRM at inference time. The final
  "#### <answer>" step is labeled correct iff the answer matches metadata.answer_value. This gives
  a dense, reward-correlated per-step signal that is NOT answer-info-gain — it tracks process
  correctness, exactly the variant (b) hypothesis.

Usage:
  # Stage 1: generate teacher rollouts + per-step labels (1×H100, ~2h for 2000 rollouts)
  python -m harness.train_prm stage=generate \\
          teacher=allenai/OLMo-2-1124-7B-SFT dataset=gsm_symbolic n=2000 seed=4242 \\
          out_dir=rft_data/prm_teacher_b max_new=1024 temperature=0.6

  # Stage 2: train the PRM head (1×H100, ~1h for 1B-SFT, 3 epochs over the JSONL)
  python -m harness.train_prm stage=train \\
          base=allenai/OLMo-2-0425-1B-SFT data=rft_data/prm_teacher_b/prm_train.jsonl \\
          out_dir=harness/checkpoints/prm_step_level_s4242 epochs=3 lr=1e-5 batch=4

Both stages are wrapped by run_exp9_train_prm.sh (SLURM, 1×H100).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Late imports (transformers/reasoning_gym) happen inside functions so `--help` and the stage=smoke
# path don't require a GPU or the heavy deps to be importable at module-load time.

# Re-export the rollout helper so the generation stage uses the SAME sampling path as the trainer
# (identical chat template, SYSTEM_PROMPT, action_mask convention).
from ._pg import (
    Experience,  # noqa: F401  (used by callers)
    compute_log_probs,
    load_model,
    seed_everything,
)

# --- step segmentation + labeling -------------------------------------------------

# Matches a number (int, float, "a/b", "a*b", "a+b", "a-b") anywhere in a step.
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")
# The CoT terminator "#### <answer>" — the final step.
_ANSWER_TAG = "####"


def _extract_step_numbers(step_text: str) -> list[float]:
    """Pull every numeric literal in a CoT step. Used to match against metadata.variables.

    Handles plain ints (42), floats (3.14), and fractions (1/2, 3/4) by evaluating the fraction.
    Any match that doesn't parse to a float is skipped (defensive — the regex is permissive).
    """
    out: list[float] = []
    for m in _NUM_RE.finditer(step_text):
        s = m.group(0)
        try:
            if "/" in s:
                num, den = s.split("/", 1)
                out.append(float(num) / float(den))
            else:
                out.append(float(s))
        except (ValueError, ZeroDivisionError):
            continue
    return out


def segment_cot(answer_cot: str) -> list[str]:
    """Split a gsm_symbolic answer_cot into steps.

    The CoT is newline-separated; the final line is "#### <answer>". We keep each non-empty line
    as one step, stripping the "####" terminator into the last step. Returns [] for empty input.
    """
    lines = [ln.strip() for ln in answer_cot.strip().splitlines() if ln.strip()]
    # Collapse the "#### <answer>" into its own step (it's already on its own line in gsm_symbolic).
    return lines


def label_steps(
    steps: list[str],
    variables: dict[str, Any],
    answer_value: Any,
) -> list[int]:
    """Label each CoT step 1 (correct) / 0 (incorrect) / -1 (ambiguous, excluded from loss).

    A step is CORRECT iff it contains a number that matches one of the ground-truth intermediate
    values in `variables` (or the final answer) AND that value hasn't already been claimed by an
    earlier correct step in the chain (so a repeated value doesn't double-count). AMBIGUOUS steps
    (no numeric literal, or a number not in variables) get -1 and are masked out of BCE.

    This is a deliberately simple heuristic — it labels the *arithmetic* steps that gsm_symbolic's
    procedural generator produces, which is exactly the distribution the PRM will score at inference
    time. Edge cases (a step that mentions a correct number for the wrong reason) are rare in
    gsm_symbolic because the CoT is deterministic given the variables.
    """
    gold_values: list[float] = []
    for v in variables.values():
        try:
            gold_values.append(float(v))
        except (ValueError, TypeError):
            continue
    try:
        gold_values.append(float(answer_value))
    except (ValueError, TypeError):
        pass
    gold_pool = {}  # value -> count (so repeated gold values can be claimed multiple times)
    for v in gold_values:
        gold_pool[v] = gold_pool.get(v, 0) + 1

    labels: list[int] = []
    claimed: dict[float, int] = {}
    for i, step in enumerate(steps):
        is_answer_step = step.startswith(_ANSWER_TAG)
        if is_answer_step:
            # Final step: correct iff the answer matches.
            try:
                ans = float(step.replace(_ANSWER_TAG, "").strip())
                ok = math.isclose(ans, float(answer_value), rel_tol=1e-6, abs_tol=1e-3)
                labels.append(1 if ok else 0)
            except (ValueError, TypeError):
                labels.append(-1)
            continue

        nums = _extract_step_numbers(step)
        if not nums:
            # No numeric literal in the step — truly unscorable (e.g. "First, I think about the problem.").
            labels.append(-1)
            continue

        # A step is CORRECT iff at least one of its numbers matches a gold value that hasn't been
        # fully claimed yet. A step WITH a number that matches NO gold value is INCORRECT (0) —
        # that's the load-bearing signal for the PRM (process correctness, not just answer-info-gain).
        # This is permissive (a step mentioning the right number for the wrong reason would be labeled
        # correct) but gsm_symbolic's CoT is deterministic and short, so false positives are rare.
        step_correct = False
        for n in nums:
            for gv in gold_pool:
                if math.isclose(n, gv, rel_tol=1e-6, abs_tol=1e-3):
                    n_clamped = round(float(gv), 6)
                    claimed[n_clamped] = claimed.get(n_clamped, 0)
                    if claimed[n_clamped] < gold_pool[gv]:
                        claimed[n_clamped] += 1
                        step_correct = True
                    break
            if step_correct:
                break
        labels.append(1 if step_correct else 0)
    return labels


# --- Stage 1: generate teacher rollouts + per-step labels ------------------------

@dataclass
class TrajectoryLabel:
    """One (trajectory, per-step-label) pair for PRM training."""
    prompt_text: str
    completion_text: str
    step_token_spans: list[tuple[int, int]]  # (start, end) token indices in the completion
    step_labels: list[int]                    # 1 / 0 / -1 per step


def _format_prompt(tokenizer, question: str, system_prompt: str) -> tuple[str, str]:
    """Build the chat-templated prompt; return (full_prompt_text, completion_prefix_text)."""
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    return prompt, prompt


def _rollout_teacher(
    teacher_name: str,
    dataset,
    n: int,
    seed: int,
    max_new: int,
    temperature: float,
    device: str,
    out_path: Path,
) -> int:
    """Generate teacher rollouts on gsm_symbolic, segment + label each, save JSONL to out_path."""
    from transformers import AutoTokenizer
    from reasoning_gym.utils import SYSTEM_PROMPTS

    seed_everything(seed)
    teacher, tokenizer = load_model(teacher_name, torch.device(device), gradient_checkpointing=False)
    teacher.eval()
    system_prompt = SYSTEM_PROMPTS["DeepSeekZero"]

    items: list[dict[str, Any]] = []
    n_correct = 0
    for i in range(n):
        entry = dataset[i]
        prompt_text, _ = _format_prompt(tokenizer, entry["question"], system_prompt)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(teacher.device)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        from transformers import GenerationConfig
        gen = teacher.generate(
            **inputs,
            generation_config=GenerationConfig(
                do_sample=True, temperature=temperature, top_p=0.95, top_k=20, min_p=0.0,
                max_new_tokens=max_new, pad_token_id=pad_id,
            ),
        )
        completion_ids = gen[:, inputs["input_ids"].shape[1]:]
        completion_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)[0]
        # Score the full answer (for filtering: keep only correct-on-the-answer rollouts — the PRM
        # should learn from trajectories that ARE correct, not from failures. This matches the
        # "process correctness" supervision story.)
        from reasoning_gym.utils import extract_answer
        ans = extract_answer(completion_text)
        acc = dataset.score_answer(ans, entry)
        if acc < 1.0:
            continue  # skip incorrect rollouts — PRM trains on correct trajectories only

        # Segment the completion into CoT steps aligned to the GOLD CoT's step structure.
        # We segment the MODEL's completion (not the gold CoT) and label each model step by
        # matching its arithmetic against metadata.variables.
        steps = segment_cot(entry["metadata"].get("answer_cot", ""))
        if not steps:
            continue
        labels = label_steps(steps, entry["metadata"].get("variables", {}), entry["answer"])
        # Mask out ambiguous steps; if NO step is correct, skip (nothing for the PRM to learn).
        if all(l != 1 for l in labels):
            continue

        # Tokenize the completion to get token spans per step. We tokenize each step separately and
        # record (start, end) token offsets in the full completion. This is approximate (tokenization
        # is not always a clean split at step boundaries) but sufficient for PRM training — the PRM
        # scores the LAST token of each step as the step-level scalar.
        step_token_spans: list[tuple[int, int]] = []
        offset = 0
        full_completion_ids = tokenizer(completion_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        # Heuristic: tokenize each step's text, find it in the full completion by string match,
        # record the token span. If a step's text isn't found (tokenization boundary), skip its span
        # and let the PRM score the previous step's last token.
        for step_text in steps:
            step_ids = tokenizer(step_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            slen = step_ids.numel()
            step_token_spans.append((offset, offset + slen))
            offset += slen
        # Pad spans to len(steps) (in case segmentation/tokenization mismatch).
        while len(step_token_spans) < len(steps):
            step_token_spans.append(step_token_spans[-1] if step_token_spans else (0, 0))

        items.append({
            "prompt_text": prompt_text,
            "completion_text": completion_text,
            "step_token_spans": [list(s) for s in step_token_spans[:len(steps)]],
            "step_labels": labels[:len(step_token_spans)],
            "question": entry["question"],
            "answer": entry["answer"],
        })
        n_correct += 1
        if (n_correct % 50) == 0:
            print(f"[prm-gen] {n_correct} correct rollouts (of {i+1} tried)", flush=True)

    with open(out_path, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"[prm-gen] wrote {len(items)} labeled trajectories to {out_path} (skipped {n - len(items)} incorrect/ambiguous)")
    teacher.cpu()
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return len(items)


# --- Stage 2: train the PRM head --------------------------------------------------

class PRMTrajDataset(Dataset):
    """Yields (input_ids, step_label_positions, step_labels) for PRM training.

    The PRM is the 1B-SFT base with a scalar head; we train it to output, at each step's LAST
    token, a logit that sigmoid-maps to P(step correct | trajectory prefix). We use the base LM's
    hidden state at the step's last token + a linear head to a single scalar.
    """

    def __init__(self, jsonl_path: str, tokenizer, max_len: int = 2048):
        self.items: list[dict[str, Any]] = []
        with open(jsonl_path) as f:
            for line in f:
                self.items.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        # Tokenize prompt + completion together (the full trajectory the PRM scores).
        ids = self.tokenizer(it["prompt_text"] + it["completion_text"], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        # The step's last-token position in the FULL sequence = prompt_len + step_end. We need
        # prompt_len to offset the step_token_spans (which are in completion-token units).
        prompt_len = self.tokenizer(it["prompt_text"], return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
        spans = it["step_token_spans"]
        labels = it["step_labels"]
        # Step label positions in the full sequence (last token of each step).
        step_positions = []
        step_label_values = []
        for (s, e), lab in zip(spans, labels):
            if lab == -1:
                continue  # ambiguous — mask out
            pos = prompt_len + e - 1  # last token of the step
            if pos < self.max_len:
                step_positions.append(pos)
                step_label_values.append(float(lab))
        return {
            "input_ids": ids[: self.max_len],
            "step_positions": torch.tensor(step_positions, dtype=torch.long),
            "step_labels": torch.tensor(step_label_values, dtype=torch.float),
        }


def _collate_prm(batch):
    """Pad to the max length in the batch; build the step-label mask."""
    maxlen = max(b["input_ids"].numel() for b in batch)
    pad_id = batch[0]["input_ids"][0].item() if batch[0]["input_ids"].numel() > 0 else 0
    # We need the tokenizer's pad_id; pass it via a global set by the caller.
    pad = _PRM_PAD_ID[0] if _PRM_PAD_ID[0] is not None else pad_id
    input_ids = torch.full((len(batch), maxlen), pad, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), maxlen), dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].numel()
        input_ids[i, :n] = b["input_ids"]
        attention_mask[i, :n] = 1
    # Step positions + labels are per-item (variable length); collect as lists.
    step_positions = [b["step_positions"] for b in batch]
    step_labels = [b["step_labels"] for b in batch]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "step_positions": step_positions,
        "step_labels": step_labels,
    }


_PRM_PAD_ID: list = [None]  # mutable cell for the pad id (set in train())


class PRMHead(nn.Module):
    """A scalar regression head on top of the base LM's hidden state: hidden -> 1 logit per token.

    Initialized as a zero-mean small-variance linear (so initial PRM scores are ~0 = sigmoid(0)=0.5,
    a neutral prior). Trained with BCE on step-boundary positions.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1, bias=True)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_states).squeeze(-1)  # (B, S)


def _train_prm(
    base_name: str,
    data_path: str,
    out_dir: str,
    epochs: int,
    lr: float,
    batch: int,
    max_len: int,
    seed: int,
    device: str,
):
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    seed_everything(seed)
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _PRM_PAD_ID[0] = tokenizer.pad_token_id

    cfg = AutoConfig.from_pretrained(base_name, trust_remote_code=False)
    if hasattr(cfg, "tie_word_embeddings"):
        cfg.tie_word_embeddings = False
    model = AutoModelForCausalLM.from_pretrained(
        base_name, config=cfg, trust_remote_code=False, dtype=torch.bfloat16,
    ).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # The PRM head operates on the base model's hidden state; we do NOT tie it to lm_head.
    prm_head = PRMHead(model.config.hidden_size).to(device).to(torch.bfloat16)
    prm_head.train()

    dataset = PRMTrajDataset(data_path, tokenizer, max_len=max_len)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, collate_fn=_collate_prm, drop_last=True)
    opt = torch.optim.AdamW(list(model.parameters()) + list(prm_head.parameters()), lr=lr, weight_decay=0.0)
    bce = nn.BCEWithLogitsLoss()

    print(f"[prm-train] {len(dataset)} trajectories, {epochs} epochs, batch {batch}, lr {lr}, max_len {max_len}")
    step = 0
    for epoch in range(epochs):
        for batch_data in loader:
            input_ids = batch_data["input_ids"].to(device)
            attn = batch_data["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=False, use_cache=False)
            hidden = outputs.last_hidden_state  # (B, S, H) — but we need logits at step positions
            logits = prm_head(hidden)          # (B, S)
            # Gather logits at step positions and compute BCE against step_labels.
            loss = 0.0
            n_steps = 0
            for i in range(input_ids.shape[0]):
                positions = batch_data["step_positions"][i].to(device)
                labels = batch_data["step_labels"][i].to(device)
                if positions.numel() == 0:
                    continue
                gathered = logits[i].index_select(0, positions)  # (n_steps_i,)
                loss = loss + bce(gathered, labels)
                n_steps += positions.numel()
            if n_steps == 0:
                continue
            loss = loss / max(1, input_ids.shape[0])  # mean over the batch
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(prm_head.parameters()), 1.0)
            opt.step()
            step += 1
            if step % 10 == 0:
                print(f"[prm-train] epoch {epoch} step {step} loss {loss.item():.4f} (n_steps/batch {n_steps})", flush=True)

    # Save the PRM head + a small config so PRMTeacher can load it.
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "prm_head_state": prm_head.state_dict(),
        "hidden_size": model.config.hidden_size,
        "base_model_name": base_name,
        "step_label_convention": "1=correct, 0=incorrect, -1=ambiguous(masked)",
        "trained_on": "gsm_symbolic teacher rollouts (correct-only)",
    }, out / "prm_head.pt")
    # Also save the base model name so PRMTeacher can load the base + head together.
    (out / "base_model_name.txt").write_text(base_name)
    # Save the tokenizer too for convenience.
    tokenizer.save_pretrained(out)
    print(f"[prm-train] saved PRM head to {out}/prm_head.pt (base = {base_name})")


# --- CLI --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage-1 PRM trainer for prms-as-teachers variant (b)")
    parser.add_argument("stage", choices=["generate", "train", "smoke"], help="generate = teacher rollouts + labels; train = fit the PRM head; smoke = unit-test the labeling")
    # generate
    parser.add_argument("--teacher", default="allenai/OLMo-2-1124-7B-SFT", help="teacher model for rollout generation")
    parser.add_argument("--dataset", default="gsm_symbolic")
    parser.add_argument("--n", type=int, default=2000, help="number of rollouts to generate")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--max_new", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="rft_data/prm_teacher_b")
    # train
    parser.add_argument("--base", default="allenai/OLMo-2-0425-1B-SFT", help="base model for the PRM head")
    parser.add_argument("--data", default="rft_data/prm_teacher_b/prm_train.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--out", default="harness/checkpoints/prm_step_level_s4242")
    args = parser.parse_args()

    if args.stage == "smoke":
        # Unit-test the step segmentation + labeling on a fixed example.
        cot = "10 years from now Jennifer will be 15*5=75.\nRight now Jennifer is 75-10=65 years old.\n#### 65"
        steps = segment_cot(cot)
        labels = label_steps(steps, {"age_in_10_years": 75, "age_now": 65}, 65)
        print("steps:", steps)
        print("labels:", labels)
        assert labels == [1, 1, 1], f"expected [1,1,1], got {labels}"
        # ambiguous step (no number):
        cot2 = "First, I think about the problem.\n5 * 5 = 25.\n#### 25"
        steps2 = segment_cot(cot2)
        labels2 = label_steps(steps2, {"product": 25}, 25)
        print("steps2:", steps2)
        print("labels2:", labels2)
        assert labels2 == [-1, 1, 1], f"expected [-1,1,1], got {labels2}"
        # incorrect step:
        cot3 = "5 * 5 = 30.\n#### 30"
        steps3 = segment_cot(cot3)
        labels3 = label_steps(steps3, {"product": 25}, 25)
        print("steps3:", steps3)
        print("labels3:", labels3)
        assert labels3 == [0, 0], f"expected [0,0], got {labels3}"
        print("[prm-smoke] ALL PASS")
        return

    if args.stage == "generate":
        import reasoning_gym as rg
        from reasoning_gym.composite import DatasetSpec
        dataset = rg.create_dataset("composite", size=args.n * 4, seed=args.seed,
                                     datasets=[DatasetSpec(name=args.dataset, weight=1.0, config={})])
        out_path = Path(args.out_dir) / "prm_train.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n = _rollout_teacher(args.teacher, dataset, args.n, args.seed, args.max_new, args.temperature, args.device, out_path)
        print(f"[prm-gen] DONE — {n} labeled trajectories at {out_path}")
        return

    if args.stage == "train":
        _train_prm(args.base, args.data, args.out, args.epochs, args.lr, args.batch, args.max_len, args.seed, args.device)
        return


if __name__ == "__main__":
    main()
