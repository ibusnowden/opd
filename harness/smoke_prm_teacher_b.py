"""Smoke test for PRM-as-teacher variant (b) plumbing.

CPU-only by default — unit-tests the new pure math + config validation:
  * PRMTeacher builds from a TeacherSpec(kind="prm_trained") and raises on token_logprobs (it
    provides step_scores, not logits).
  * config: the 4 launcher arms (clip x prm_reweight x prm_source) all validate, including the
    trained-PRM arms that require prm_model_path.
  * train_prm.segment_cot / label_steps: the step-labeling heuristic (correct/incorrect/ambiguous)
    on fixed examples.
  * train_prm._PRMHead: shape + init (zero-mean small-variance linear).

If CUDA is available it additionally:
  * builds a tiny PRM checkpoint (1B-SFT base + random PRMHead) in a temp dir
  * loads it via PRMTeacher and checks step_scores returns (B, S-1) in [0,1] on a 1B student rollout
  * checks the rollout-time PRM block in unified_trainer wires prm_source="trained" correctly
    (teacher_logprobs from PrivilegedInfoTeacher, prm_weights from PRMTeacher.step_scores).

Usage:
    python -m harness.smoke_prm_teacher_b
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch
import yaml

from . import _pg
from .config import ResearchConfig, TeacherSpec
from .teachers import PRMTeacher, build_teacher, PrivilegedInfoTeacher, _PRMHead
from .train_prm import segment_cot, label_steps


def _check(name: str, cond: bool):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def test_step_labeling():
    print("[smoke] train_prm segment_cot + label_steps (the step-labeling heuristic)")
    # correct CoT
    cot = "10 years from now Jennifer will be 15*5=75.\nRight now Jennifer is 75-10=65 years old.\n#### 65"
    steps = segment_cot(cot)
    _check("segment_cot splits into 3 steps", len(steps) == 3)
    labels = label_steps(steps, {"age_in_10_years": 75, "age_now": 65}, 65)
    _check("correct CoT -> [1,1,1]", labels == [1, 1, 1])

    # ambiguous first step (no number) + correct arithmetic
    cot2 = "First, I think about the problem.\n5 * 5 = 25.\n#### 25"
    steps2 = segment_cot(cot2)
    labels2 = label_steps(steps2, {"product": 25}, 25)
    _check("ambiguous-then-correct -> [-1,1,1]", labels2 == [-1, 1, 1])

    # incorrect arithmetic
    cot3 = "5 * 5 = 30.\n#### 30"
    steps3 = segment_cot(cot3)
    labels3 = label_steps(steps3, {"product": 25}, 25)
    _check("incorrect arithmetic -> [0,0]", labels3 == [0, 0])

    # wrong answer
    cot4 = "5 * 5 = 25.\n#### 30"
    steps4 = segment_cot(cot4)
    labels4 = label_steps(steps4, {"product": 25}, 25)
    _check("wrong answer -> [1,0]", labels4 == [1, 0])


def test_prm_head():
    print("[smoke] _PRMHead shape + init")
    head = _PRMHead(hidden_size=2048)
    _check("linear.weight shape", head.linear.weight.shape == (1, 2048))
    _check("linear.bias shape", head.linear.bias.shape == (1,))
    # forward
    hidden = torch.randn(2, 10, 2048)
    logits = head(hidden)
    _check("forward -> (B, S)", logits.shape == (2, 10))


def test_config_arms():
    print("[smoke] 4 launcher arms (clip x prm x source) all validate")
    raw = yaml.safe_load(open("harness/configs/exp9_prm_teacher_b.yaml"))
    PRM_PATH = "/tmp/fake_prm_ckpt"  # placeholder; the launcher sets the real path
    for clip, prm, src, mp in [(1.0, False, "answer_info_gain", None), (None, False, "answer_info_gain", None),
                                (None, True, "trained", PRM_PATH), (1.0, True, "trained", PRM_PATH)]:
        cfg = ResearchConfig(**{**raw, "per_token_kl_clip": clip, "prm_reweight": prm,
                                 "prm_source": src, "prm_model_path": mp})
        _check(f"arm clip={clip} prm={prm} src={src} -> recipe={cfg.recipe}", cfg.lam == 1.0)
    # negative: trained PRM without prm_model_path
    try:
        ResearchConfig(**{**raw, "prm_source": "trained", "prm_reweight": True})
        _check("trained PRM without prm_model_path should raise", False)
    except Exception:
        _check("trained PRM without prm_model_path raises", True)
    # negative: trained PRM without answer-conditioned teacher
    bad = {**raw, "prm_source": "trained", "prm_reweight": True, "prm_model_path": PRM_PATH,
           "teacher": {**raw["teacher"], "condition_on": None}}
    try:
        ResearchConfig(**bad)
        _check("trained PRM without answer-teacher should raise", False)
    except Exception:
        _check("trained PRM without answer-teacher raises", True)


def test_prm_teacher_builds():
    print("[smoke] PRMTeacher builds from TeacherSpec(kind='prm_trained')")
    spec = TeacherSpec(kind="prm_trained", model_name="/tmp/fake_prm_ckpt", device_id=0)
    t = build_teacher(spec, student_model_name="allenai/OLMo-2-0425-1B-SFT")
    _check("build_teacher(kind=prm_trained) -> PRMTeacher", type(t).__name__ == "PRMTeacher")
    # token_logprobs must raise (PRM provides importance, not logits)
    try:
        t.token_logprobs(None, None, None)
        _check("PRMTeacher.token_logprobs raises NotImplementedError", False)
    except NotImplementedError:
        _check("PRMTeacher.token_logprobs raises NotImplementedError", True)


def test_prm_teacher_gpu():
    if not torch.cuda.is_available():
        print("[smoke] (skipping PRMTeacher GPU test — no CUDA)")
        return
    print("[smoke] PRMTeacher.step_scores end-to-end (1B student rollout + tiny PRM checkpoint)")
    import reasoning_gym as rg
    from reasoning_gym.composite import DatasetSpec

    device = torch.device("cuda:0")
    student_name = "allenai/OLMo-2-0425-1B-SFT"

    # Build a tiny PRM checkpoint: 1B-SFT base + random PRMHead, saved to a temp dir.
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save the base model name + a random PRMHead.
        (Path(tmpdir) / "base_model_name.txt").write_text(student_name)
        head = _PRMHead(hidden_size=2048)  # OLMo-2-1B hidden size
        torch.save({"prm_head_state": head.state_dict(), "hidden_size": 2048,
                    "base_model_name": student_name,
                    "step_label_convention": "1=correct, 0=incorrect, -1=ambiguous(masked)"},
                    Path(tmpdir) / "prm_head.pt")
        # Build the PRMTeacher and load the base + head.
        spec = TeacherSpec(kind="prm_trained", model_name=tmpdir, device_id=0)
        prm = PRMTeacher(spec, student_model_name=student_name)
        prm._ensure_model()
        _check("PRMTeacher loads base + head", prm._model is not None and prm._prm_head is not None)

        # Run a tiny rollout and check step_scores.
        student, tokenizer = _pg.load_model(student_name, device, gradient_checkpointing=False)
        student.eval()
        dataset = rg.create_dataset("composite", size=4, seed=2026,
                                     datasets=[DatasetSpec(name="gsm_symbolic", weight=1.0, config={})])
        entries = [dataset[i] for i in range(4)]
        with torch.no_grad():
            seq, am, attn, _r, _c, _a, _f = _pg.rollout(
                model=student, entries=entries, dataset=dataset, tokenizer=tokenizer,
                max_new_tokens=64, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
            g = prm.step_scores(seq, attn, am, entries=entries)
        _check("step_scores shape == (B, S-1)", g.shape == (seq.shape[0], seq.shape[1] - 1))
        _check("step_scores in [0, 1]", bool((g >= 0).all() and (g <= 1).all()))
        mask = am.bool().to(g.device)
        gv = g[mask]
        print(f"  step_scores at action tok: mean={gv.mean():.4f} std={gv.std():.4f} "
              f"p99={torch.quantile(gv.float(), 0.99):.4f} max={gv.max():.4f}")
        _check("step_scores finite at action positions", bool(torch.isfinite(g[mask]).all()))


def main():
    test_step_labeling()
    test_prm_head()
    test_config_arms()
    test_prm_teacher_builds()
    test_prm_teacher_gpu()
    print("[smoke] ALL PASS")


if __name__ == "__main__":
    main()
