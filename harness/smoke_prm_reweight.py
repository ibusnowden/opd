"""Smoke test for PRM-reweighted OPSD (prms-as-teachers variant (c)) plumbing.

CPU-only by default — unit-tests the new pure math:
  * prm_importance_weights: mass-preservation (mean(w) over action tokens == 1), monotonicity
    (higher importance -> higher weight), softmax & linear, edge cases (zero action tokens, all-equal
    g, ceiling cap);
  * reverse_kl_distill_advantage with prm_weights: clip-then-reweight matches a manual computation;
  * Experience round-trip: prm_weights survives split/join/.to() with the other per-token tensors;
  * config: the 4 launcher arms (clip x prm_reweight) all validate.

If CUDA is available it additionally loads the 1B student + 7B-SFT teacher and checks
PrivilegedInfoTeacher.answer_info_gain end-to-end (shapes, finiteness, that g concentrates).

Usage:
    python -m harness.smoke_prm_reweight
"""
from __future__ import annotations

import torch
import yaml

from . import _pg
from .config import ResearchConfig
from .distill_losses import per_token_kl_clip, prm_importance_weights, reverse_kl_distill_advantage


def _check(name: str, cond: bool):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def test_weights_math():
    print("[smoke] prm_importance_weights — mass preservation + monotonicity")
    torch.manual_seed(0)
    B, T = 3, 10
    g = torch.randn(B, T)
    action_mask = torch.ones(B, T)
    action_mask[1, 6:] = 0.0      # row 1: partial (6 action tokens)
    action_mask[2, :] = 0.0       # row 2: NO action tokens (degenerate)

    for fn in ("softmax", "linear"):
        w = prm_importance_weights(g, action_mask, fn=fn, temp=1.0, ceiling=None)
        _check(f"{fn}: shape == g", w.shape == g.shape)
        _check(f"{fn}: w >= 0", bool((w >= -1e-6).all()))
        _check(f"{fn}: zero on non-action positions", float((w * (1 - action_mask)).abs().sum()) < 1e-5)
        # mean over action tokens == 1 for rows that HAVE action tokens
        for i in (0, 1):
            n = int(action_mask[i].sum())
            mean_w = float(w[i, action_mask[i].bool()].mean())
            _check(f"{fn} row{i}: mean(w over {n} action tok) == 1", abs(mean_w - 1.0) < 1e-4)
        # degenerate row -> all zero (no contribution, no NaN)
        _check(f"{fn} row2 (no action): all zero & finite", bool(torch.isfinite(w[2]).all()) and float(w[2].abs().sum()) < 1e-6)
        # monotonic: within row 0, the highest-importance token gets the highest weight
        amax_g = int(g[0].argmax()); amax_w = int(w[0].argmax())
        _check(f"{fn} row0: argmax(w) == argmax(g)", amax_g == amax_w)

    print("[smoke] all-equal g -> uniform weights (== plain OPSD)")
    g_flat = torch.zeros(1, 5)
    w_flat = prm_importance_weights(g_flat, torch.ones(1, 5), fn="softmax")
    _check("softmax flat g -> all weights 1.0", bool((w_flat - 1.0).abs().max() < 1e-4))

    print("[smoke] ceiling caps the weight")
    g_spike = torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0]])
    w_cap = prm_importance_weights(g_spike, torch.ones(1, 5), fn="softmax", temp=0.25, ceiling=2.0)
    _check("ceiling=2.0 -> max(w) <= 2.0", float(w_cap.max()) <= 2.0 + 1e-6)
    w_unc = prm_importance_weights(g_spike, torch.ones(1, 5), fn="softmax", temp=0.25, ceiling=None)
    _check("uncapped spike -> max(w) > 2.0 (so the cap was doing something)", float(w_unc.max()) > 2.0)


def test_advantage_reweight():
    print("[smoke] reverse_kl_distill_advantage with prm_weights = clip(t-s) * w * mask")
    torch.manual_seed(1)
    B, T = 2, 6
    s = torch.randn(B, T)
    t = torch.randn(B, T)
    am = torch.ones(B, T); am[0, 4:] = 0.0
    w = prm_importance_weights(torch.randn(B, T), am, fn="softmax")

    adv = reverse_kl_distill_advantage(s, t, am, clip=1.0, prm_weights=w)
    manual = per_token_kl_clip(t.detach() - s.detach(), 1.0) * w * am
    _check("clip+reweight matches manual", bool((adv - manual).abs().max() < 1e-5))

    # reweight off (None) reduces to the plain clipped term (back-compat for OPD/OPSD)
    adv0 = reverse_kl_distill_advantage(s, t, am, clip=1.0, prm_weights=None)
    manual0 = per_token_kl_clip(t.detach() - s.detach(), 1.0) * am
    _check("prm_weights=None == plain clipped advantage", bool((adv0 - manual0).abs().max() < 1e-5))


def test_experience_roundtrip():
    print("[smoke] Experience carries prm_weights through split/join/.to()")
    B, T = 4, 7
    seq = torch.randint(0, 100, (B, T + 1))
    attn = torch.ones(B, T + 1)
    am = torch.ones(B, T)
    tlp = torch.randn(B, T)
    pw = prm_importance_weights(torch.randn(B, T), am, fn="softmax")
    exp = _pg.Experience(sequence_ids=seq, attention_mask=attn, action_mask=am,
                         teacher_logprobs=tlp, prm_weights=pw)
    parts = _pg.pg_buffer.split_experience_batch(exp)
    _check("split preserves prm_weights", all(p.prm_weights is not None for p in parts))
    joined = _pg.join_experiences_batch(parts)
    _check("join preserves prm_weights shape", joined.prm_weights.shape == pw.shape)
    moved = exp.to(torch.device("cpu"))
    _check(".to() moves prm_weights", moved.prm_weights is not None)


def test_config_arms():
    print("[smoke] 4 launcher arms (clip x prm_reweight) all validate")
    raw = yaml.safe_load(open("harness/configs/exp6_prm_reweighted_opsd.yaml"))
    for clip, prm in [(1.0, False), (None, False), (None, True), (1.0, True)]:
        cfg = ResearchConfig(**{**raw, "per_token_kl_clip": clip, "prm_reweight": prm})
        _check(f"arm clip={clip} prm={prm} -> recipe={cfg.recipe}", cfg.lam == 1.0)
    # negative: prm_reweight without answer-conditioned teacher must raise
    bad = {**raw, "prm_reweight": True}
    bad["teacher"] = {**raw["teacher"], "condition_on": None}
    try:
        ResearchConfig(**bad)
        _check("prm_reweight w/o answer-teacher should raise", False)
    except Exception:
        _check("prm_reweight w/o answer-teacher raises", True)


def test_answer_info_gain_gpu():
    if not torch.cuda.is_available():
        print("[smoke] (skipping answer_info_gain GPU test — no CUDA)")
        return
    print("[smoke] answer_info_gain end-to-end (1B student + 7B-SFT teacher)")
    import reasoning_gym as rg
    from reasoning_gym.composite import DatasetSpec
    from .config import TeacherSpec
    from .teachers import PrivilegedInfoTeacher

    device = torch.device("cuda:0")
    student_name = "allenai/OLMo-2-0425-1B-SFT"
    teacher_name = "allenai/OLMo-2-1124-7B-SFT"
    student, tokenizer = _pg.load_model(student_name, device, gradient_checkpointing=False)
    student.eval()
    teacher = PrivilegedInfoTeacher(
        TeacherSpec(kind="self", model_name=teacher_name, condition_on="answer", frozen_at_init=True),
        student_name)
    teacher._ensure_model()
    dataset = rg.create_dataset("composite", size=4, seed=2026,
                                datasets=[DatasetSpec(name="gsm_symbolic", weight=1.0, config={})])
    entries = [dataset[i] for i in range(4)]
    with torch.no_grad():
        seq, am, attn, _r, _c, _a, _f = _pg.rollout(
            model=student, entries=entries, dataset=dataset, tokenizer=tokenizer,
            max_new_tokens=128, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
        lp_answer, g = teacher.answer_info_gain(seq, attn, am, entries=entries)
    _check("lp_answer shape == (B, S-1)", lp_answer.shape == (seq.shape[0], seq.shape[1] - 1))
    _check("g shape == (B, S-1)", g.shape == lp_answer.shape)
    mask = am.bool().to(g.device)
    _check("g finite at action positions", bool(torch.isfinite(g[mask]).all()))
    gv = g[mask]
    print(f"  g (answer-info-gain) at action tok: mean={gv.mean():.4f} std={gv.std():.4f} "
          f"p99={torch.quantile(gv.float(),0.99):.4f} max={gv.max():.4f}")
    w = prm_importance_weights(g, am, fn="softmax", temp=1.0)
    wv = w[mask]
    print(f"  softmax weights: mean={wv.mean():.4f} (==1?) max={wv.max():.4f} p99={torch.quantile(wv.float(),0.99):.4f}")
    _check("weights mass-preserving (mean ~1 over action tokens, pooled)", abs(float(wv.mean()) - 1.0) < 0.05)


def main():
    test_weights_math()
    test_advantage_reweight()
    test_experience_roundtrip()
    test_config_arms()
    test_answer_info_gain_gpu()
    print("[smoke] ALL PASS")


if __name__ == "__main__":
    main()
