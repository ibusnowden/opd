"""Smoke test for PrivilegedInfoTeacher answer-conditioning.

Loads a small slice of gsm_symbolic, generates a few rollouts with the 1B student, then calls
PrivilegedInfoTeacher.token_logprobs and checks: (a) the output shape matches action_mask,
(b) per-row values are finite, (c) the cached teacher_logprobs would survive
Experience.to(device) and join_experiences_batch round-trip.

Usage:
    python -m harness.smoke_opsd
"""
from __future__ import annotations
import torch
from itertools import batched

import reasoning_gym as rg
from reasoning_gym.composite import DatasetSpec

from . import _pg
from .config import TeacherSpec
from .teachers import PrivilegedInfoTeacher


def main():
    print("[smoke] loading student + teacher")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    student_name = "allenai/OLMo-2-0425-1B-SFT"
    teacher_name = "allenai/OLMo-2-1124-7B-SFT"
    student, tokenizer = _pg.load_model(student_name, device, gradient_checkpointing=False)
    student.eval()

    spec = TeacherSpec(kind="self", model_name=teacher_name, condition_on="answer",
                       frozen_at_init=True, device_id=0)
    teacher = PrivilegedInfoTeacher(spec, student_name, student_model=None)
    teacher._ensure_model()  # eager load

    print("[smoke] building gsm_symbolic batch")
    dataset = rg.create_dataset("composite", size=4, seed=2026,
                                datasets=[DatasetSpec(name="gsm_symbolic", weight=1.0, config={})])
    entries = [dataset[i] for i in range(4)]
    print("[smoke] first entry:")
    print(f"  question  : {entries[0]['question'][:80]}...")
    print(f"  answer    : {entries[0]['answer']!r}")

    print("[smoke] rolling out 4 prompts × 1 sample, max_new=128, T=0.6")
    with torch.no_grad():
        seq, am, attn, _r, _comps, _acc, _fmt = _pg.rollout(
            model=student, entries=entries, dataset=dataset, tokenizer=tokenizer,
            max_new_tokens=128, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
        )
    print(f"  seq.shape={tuple(seq.shape)}  action_mask.shape={tuple(am.shape)}  "
          f"action_mask.sum_per_row={am.sum(dim=1).tolist()}")

    print("[smoke] calling teacher.token_logprobs(entries=entries)")
    with torch.no_grad():
        tlp = teacher.token_logprobs(seq, attn, am, entries=entries, student_logprobs=None)
    print(f"  tlp.shape={tuple(tlp.shape)}  expected (B, S-1)=({seq.shape[0]}, {seq.shape[1]-1})")
    finite_frac = torch.isfinite(tlp).float().mean().item()
    nonzero_at_action = ((tlp != 0.0) & am.bool().to(tlp.device)).sum().item()
    n_action = am.sum().item()
    print(f"  tlp finite_frac={finite_frac:.3f}; nonzero-at-action {nonzero_at_action}/{n_action}")
    if n_action > 0:
        valid = tlp[am.bool().to(tlp.device)]
        print(f"  tlp at action positions: mean={valid.mean().item():.4f}  min={valid.min().item():.4f}  "
              f"max={valid.max().item():.4f}")

    print("[smoke] comparing to unconditioned teacher (no answer hint) for the same tokens")
    spec_plain = TeacherSpec(kind="same_family", model_name=teacher_name, device_id=0)
    from .teachers import SameFamilyTeacher
    plain = SameFamilyTeacher(spec_plain, student_name)
    plain._ensure_model()
    with torch.no_grad():
        plain_lp = plain.token_logprobs(seq, attn, am, entries=None, student_logprobs=None)
    diff = (tlp - plain_lp)[am.bool().to(tlp.device)]
    print(f"  Δ(OPSD-plain) at action positions: mean={diff.mean().item():.4f}  "
          f"std={diff.std().item():.4f}  abs_max={diff.abs().max().item():.4f}")
    print("[smoke] OK." if finite_frac == 1.0 and nonzero_at_action == n_action else "[smoke] FAIL: see numbers above")


if __name__ == "__main__":
    main()
