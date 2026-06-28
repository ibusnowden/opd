"""diagnose_per_token_kl — given a (student, teacher) pair, roll out N held-out prompts at training
temperature, score correctness, and report **where the per-token reverse-KL signal landed**.

This is the per-token-KL pivot/style question (proposal #4) restricted to the failure case from
Exp 1: does the teacher's `log π_T − log π_θ` signal concentrate on tokens that distinguish correct
from incorrect completions, or on formatting/style tokens that don't?

Outputs (one row per generated token, JSON-lines, plus a small summary):
  prompt_idx, sample_idx, position, token_id, token_str,
  log_p_student, log_p_teacher, kl_signal = log_p_teacher - log_p_student,
  is_correct (per-completion), is_format_token (a heuristic),
  ent_student (per-position entropy of π_θ).

Then prints a small table:
  - mean / quantiles of `kl_signal` overall
  - mean `kl_signal` on correct vs. incorrect completions
  - fraction of top-q% |kl_signal| tokens that land on format tokens vs content tokens
  - fraction of top-q% kl_signal tokens whose completions ended up correct

Lightweight; single GPU; ~5 min for 32 prompts × 4 samples × 1024 max_new on an H100.

Usage:
  python -m harness.diagnose_per_token_kl \
      --student harness/checkpoints/opd-sft7b-s42 \
      --teacher allenai/OLMo-2-1124-7B-SFT \
      --task gsm_symbolic --n-prompts 32 --n-samples 4 --out results/diag_opd_sft7b_s42.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import batched   # py3.12+

import torch
import torch.nn.functional as F

import reasoning_gym as rg
from reasoning_gym.composite import DatasetSpec
from reasoning_gym.utils import extract_answer

from . import _pg


# Lightweight heuristic for "format/style" tokens — DeepSeekZero-style tags + common boilerplate
# whitespace + a small set of punctuation/special token strings. Conservative: misses most content.
_FORMAT_SUBSTRINGS = {
    "<think>", "</think>", "<answer>", "</answer>",
    "\n\n", "\n", "  ",
    " The", " So", " Therefore", " Thus", " Hence", " Now",
    " First", " Second", " Then", " Finally", " Next",
    " step", " Step", " is", " are",
}


def _is_format_token(tok_str: str) -> bool:
    s = tok_str
    if not s.strip():           # pure whitespace
        return True
    for needle in _FORMAT_SUBSTRINGS:
        if needle in s:
            return True
    return False


# Taxonomy (Exp 2 / proposal #4): which kind of token does each per-token-KL
# value correspond to? Buckets are mutually exclusive in this order:
#   1. format   — DeepSeekZero tag / whitespace / boilerplate (always wins; cheap)
#   2. uncertain — student entropy > UNCERTAIN_ENT (the student doesn't know)
#   3. wrong_confident — student entropy < CONFIDENT_ENT AND kl < -WRONG_KL
#                        (student is sure, teacher disagrees strongly; the OPSD
#                         "confidently wrong" failure mode)
#   4. content  — everything else (the meat — math/reasoning tokens where the
#                 student has a definite prediction the teacher mostly agrees with;
#                 the "pivot" candidates live here)
_UNCERTAIN_ENT = 1.0    # nats; > 1.0 ~ a near-uniform distribution over a few tokens
_CONFIDENT_ENT = 0.2    # nats; < 0.2 ~ p_max > 0.85
_WRONG_KL      = 1.5    # log-prob gap; teacher would have placed >4× more mass


def categorize_token(tok_str: str, ent: float, kl: float) -> str:
    if _is_format_token(tok_str):
        return "format"
    if ent > _UNCERTAIN_ENT:
        return "uncertain"
    if ent < _CONFIDENT_ENT and kl < -_WRONG_KL:
        return "wrong_confident"
    return "content"


def taxonomy_summary(rows: list[dict], clip: float = 1.0) -> dict:
    """Per-bucket counts and |kl| mass. Also reports what `per_token_kl_clip = clip`
    would do to each bucket's mass: fraction of mass removed by clipping = how much
    of the destabilising signal §7.7 was actually disarming."""
    buckets = ("format", "uncertain", "wrong_confident", "content")
    by_b: dict[str, dict] = {b: {"n": 0, "sum_abs_kl": 0.0, "sum_kl": 0.0,
                                  "sum_abs_kl_clipped": 0.0, "n_correct": 0,
                                  "sum_ent": 0.0, "sum_clipped_off": 0.0} for b in buckets}
    n_total = len(rows)
    total_abs = 0.0
    total_clip_removed = 0.0
    for r in rows:
        b = categorize_token(r["tok"], r["ent_s"], r["kl"])
        akl = abs(r["kl"])
        akl_c = min(akl, clip)
        d = by_b[b]
        d["n"] += 1
        d["sum_abs_kl"] += akl
        d["sum_kl"] += r["kl"]
        d["sum_abs_kl_clipped"] += akl_c
        d["sum_ent"] += r["ent_s"]
        if r.get("is_correct"):
            d["n_correct"] += 1
        if akl > clip:
            d["sum_clipped_off"] += (akl - akl_c)
        total_abs += akl
        total_clip_removed += (akl - akl_c)
    out = {"clip_threshold": clip, "n_total": n_total,
           "total_abs_kl": total_abs,
           "total_clip_removed_abs_kl": total_clip_removed,
           "buckets": {}}
    for b in buckets:
        d = by_b[b]
        out["buckets"][b] = {
            "n_tokens": d["n"],
            "frac_of_tokens": d["n"] / n_total if n_total else 0.0,
            "mass_frac": d["sum_abs_kl"] / total_abs if total_abs > 0 else 0.0,
            "mean_kl": d["sum_kl"] / d["n"] if d["n"] else 0.0,
            "mean_abs_kl": d["sum_abs_kl"] / d["n"] if d["n"] else 0.0,
            "mean_entropy": d["sum_ent"] / d["n"] if d["n"] else 0.0,
            "corr_frac": d["n_correct"] / d["n"] if d["n"] else 0.0,
            "frac_of_clip_removed": (d["sum_clipped_off"] / total_clip_removed) if total_clip_removed > 0 else 0.0,
            "mass_frac_after_clip": (d["sum_abs_kl_clipped"] / (total_abs - total_clip_removed)) if (total_abs - total_clip_removed) > 0 else 0.0,
        }
    return out


def summarize_kl_signal(
    kl: list[float],
    *,
    is_correct: list[bool] | None = None,
    is_format: list[bool] | None = None,
    abs_top_q: float = 0.10,
    heavy_tail: float = 5.0,
) -> dict:
    """Aggregate per-token reverse-KL stats. Shared by `diagnose_per_token_kl` (offline,
    per-checkpoint with correctness/format flags) and `_run_distill_loop` (online, per-step,
    no flags). Pass only the lists you have; missing ones become NaN in the result."""
    import statistics as stat
    n = len(kl)
    if n == 0:
        return {"n_tokens": 0}

    def q(vs: list[float], p: float) -> float:
        return sorted(vs)[int(p * (len(vs) - 1))] if vs else float("nan")

    abs_kl_sorted = sorted(((abs(v), i) for i, v in enumerate(kl)), reverse=True)
    cut = max(1, int(abs_top_q * n))
    top_idx = {i for _, i in abs_kl_sorted[:cut]}

    out = {
        "n_tokens": n,
        "kl_mean": stat.mean(kl),
        "kl_p50": q(kl, 0.5),
        "kl_p90": q(kl, 0.9),
        "kl_p99": q(kl, 0.99),
        "kl_max": max(kl),
        "kl_heavy_tail_frac": sum(1 for v in kl if abs(v) > heavy_tail) / n,
        "abs_top_q": abs_top_q,
        "heavy_tail": heavy_tail,
    }

    if is_correct is not None and len(is_correct) == n:
        kl_corr = [kl[i] for i in range(n) if is_correct[i]]
        kl_inc = [kl[i] for i in range(n) if not is_correct[i]]
        out["kl_mean_on_correct"] = stat.mean(kl_corr) if kl_corr else float("nan")
        out["kl_mean_on_incorrect"] = stat.mean(kl_inc) if kl_inc else float("nan")
        top_corr_frac = sum(1 for i in top_idx if is_correct[i]) / len(top_idx)
        overall_corr_frac = sum(is_correct) / n
        out["topq_correct_frac"] = top_corr_frac
        out["overall_correct_frac"] = overall_corr_frac
        out["topq_correct_lift"] = (top_corr_frac / overall_corr_frac) if overall_corr_frac > 0 else float("nan")

    if is_format is not None and len(is_format) == n:
        top_fmt_frac = sum(1 for i in top_idx if is_format[i]) / len(top_idx)
        overall_fmt_frac = sum(is_format) / n
        out["topq_format_frac"] = top_fmt_frac
        out["overall_format_frac"] = overall_fmt_frac
        out["topq_format_lift"] = (top_fmt_frac / overall_fmt_frac) if overall_fmt_frac > 0 else float("nan")

    return out


@torch.no_grad()
def _per_token_logp_and_entropy(model, sequence_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Return (target_logp, full_entropy_per_pos) shaped (B, S-1). Mirrors `_pg.compute_log_probs`
    but also returns entropy = -sum_v p_v log p_v, which we need for the diagnostic."""
    dev = _pg.get_model_device(model)
    out = _pg.unwrap_model(model)(input_ids=sequence_ids.to(dev), attention_mask=attention_mask.to(dev), use_cache=False)
    logits = out.logits[:, :-1, :].to(torch.float32)
    logp_full = F.log_softmax(logits, dim=-1)
    targets = sequence_ids[:, 1:].unsqueeze(-1).to(dev)
    target_logp = torch.gather(logp_full, dim=-1, index=targets).squeeze(-1)
    ent = -(logp_full.exp() * logp_full).sum(dim=-1)
    return target_logp, ent


def main_cli() -> None:
    ap = argparse.ArgumentParser(description="per-token KL × correctness diagnostic (proposal #4 on Exp-1 failure case)")
    ap.add_argument("--student", required=True, help="HF hub name or local path of the (trained or init) student")
    ap.add_argument("--teacher", required=True, help="HF hub name or local path of the (frozen) teacher")
    ap.add_argument("--task", default="gsm_symbolic")
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--gen-batch-size", type=int, default=8)
    ap.add_argument("--eval-seed", type=int, default=1_000_000)
    ap.add_argument("--student-device", default="cuda:0")
    ap.add_argument("--teacher-device", default="cuda:0")
    ap.add_argument("--out", required=True, help="path to write the JSON summary")
    ap.add_argument("--tokens-out", default=None, help="optional path to dump per-token JSONL (large)")
    ap.add_argument("--top-q", type=float, default=0.10, help="quantile cut for the 'top-q% KL' bucket")
    ap.add_argument("--clip-thresh", type=float, default=1.0, help="per-token-KL clip threshold used for the taxonomy's clip-removal view (matches §7.7's per_token_kl_clip)")
    args = ap.parse_args()

    s_dev = torch.device(args.student_device if torch.cuda.is_available() else "cpu")
    t_dev = torch.device(args.teacher_device if torch.cuda.is_available() else "cpu")

    # build dataset (same shape as eval_passk._eval_dataset)
    dataset = rg.create_dataset("composite", size=args.n_prompts, seed=args.eval_seed,
                                datasets=[DatasetSpec(name=args.task, weight=1.0, config={})])

    print(f"[diag_kl] student={args.student}  teacher={args.teacher}  task={args.task}  "
          f"n_prompts={args.n_prompts} n_samples={args.n_samples} T={args.temperature}")
    student, tokenizer = _pg.load_model(args.student, s_dev, gradient_checkpointing=False)
    student.eval()
    teacher, _ = _pg.load_model(args.teacher, t_dev, gradient_checkpointing=False)
    teacher.eval()
    if getattr(tokenizer, "chat_template", None) is None:
        raise RuntimeError(f"{args.student!r} has no chat_template")

    entries = [dataset[i] for i in range(args.n_prompts)]
    expanded = [e for e in entries for _ in range(args.n_samples)]

    # rollout (on the student, like training), score, and grab per-token logp for both models.
    rows: list[dict] = []
    n_correct = 0
    t0 = time.time()
    for chunk_idx, chunk in enumerate(batched(expanded, args.gen_batch_size)):
        chunk = list(chunk)
        seq_ids, action_mask, attn_mask, _r, comps, _acc, _fmt = _pg.rollout(
            model=student, entries=chunk, dataset=dataset, tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k, min_p=args.min_p,
        )
        # student per-token logp + entropy
        s_logp, s_ent = _per_token_logp_and_entropy(student, seq_ids, attn_mask)
        # teacher per-token logp (computed via teacher forward over the SAME tokens — that's the OPD signal)
        t_logp_only = _pg.compute_log_probs(teacher, seq_ids, attn_mask)

        # correctness per completion
        correct = [bool(dataset.score_answer(extract_answer(c), e) >= 1.0) for c, e in zip(comps, chunk)]
        n_correct += sum(correct)

        # iterate the generated positions only (action_mask is shape (B, S-1))
        seq_cpu = seq_ids[:, 1:].cpu()                          # target tokens
        am = action_mask.bool().cpu()
        slp = s_logp.detach().cpu()
        sent = s_ent.detach().cpu()
        tlp = t_logp_only.detach().cpu()
        for b in range(seq_cpu.size(0)):
            global_idx = chunk_idx * args.gen_batch_size + b
            prompt_idx = global_idx // args.n_samples
            sample_idx = global_idx %  args.n_samples
            mask_b = am[b]
            if not mask_b.any():
                continue
            ids = seq_cpu[b][mask_b].tolist()
            sl  = slp[b][mask_b].tolist()
            tl  = tlp[b][mask_b].tolist()
            se  = sent[b][mask_b].tolist()
            # decode tokens individually (cheap; one-tok strings)
            tok_strs = tokenizer.batch_decode([[tid] for tid in ids], skip_special_tokens=False)
            for pos_in_gen, (tid, ts, slv, tlv, sev) in enumerate(zip(ids, tok_strs, sl, tl, se)):
                rows.append({
                    "prompt": prompt_idx, "sample": sample_idx, "pos": pos_in_gen,
                    "tok_id": int(tid), "tok": ts,
                    "lp_s": float(slv), "lp_t": float(tlv),
                    "kl": float(tlv - slv), "ent_s": float(sev),
                    "is_correct": bool(correct[b]),
                    "is_format": _is_format_token(ts),
                })

    # --- aggregate stats --------------------------------------------------------
    n_rows = len(rows)
    if n_rows == 0:
        raise SystemExit("[diag_kl] no generated tokens collected (action_mask all-zero?)")

    agg = summarize_kl_signal(
        [r["kl"] for r in rows],
        is_correct=[r["is_correct"] for r in rows],
        is_format=[r["is_format"] for r in rows],
        abs_top_q=args.top_q,
    )
    tax = taxonomy_summary(rows, clip=args.clip_thresh)
    summary = {
        "student": args.student, "teacher": args.teacher, "task": args.task,
        "n_prompts": args.n_prompts, "n_samples": args.n_samples, "T": args.temperature,
        "n_tokens_total": n_rows,
        "completion_accuracy": n_correct / len(expanded),
        "top_q": args.top_q,
        **agg,
        "taxonomy": tax,
        "wall_seconds": time.time() - t0,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    if args.tokens_out:
        with open(args.tokens_out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[diag_kl] wrote {args.tokens_out} ({n_rows} tokens)")

    print("[diag_kl] === summary ===")
    for k, v in summary.items():
        if isinstance(v, (dict, list)):
            continue
        if isinstance(v, float):
            print(f"  {k:<24s} = {v:.4f}")
        else:
            print(f"  {k:<24s} = {v}")
    print("[diag_kl] === taxonomy (per-bucket KL mass; clip = "
          f"{tax['clip_threshold']:.2f}) ===")
    print(f"  {'bucket':<16} {'n_tok':>8} {'frac':>7} {'mass_f':>7} {'mean_kl':>8} {'<|kl|>':>7} {'<ent>':>7} {'corr_f':>7} {'clip_rm':>8}")
    for b, d in tax["buckets"].items():
        print(f"  {b:<16} {d['n_tokens']:>8d} {d['frac_of_tokens']:>7.3f} "
              f"{d['mass_frac']:>7.3f} {d['mean_kl']:>8.3f} {d['mean_abs_kl']:>7.3f} "
              f"{d['mean_entropy']:>7.3f} {d['corr_frac']:>7.3f} {d['frac_of_clip_removed']:>8.3f}")
    print(f"  total_abs_kl                       = {tax['total_abs_kl']:.1f}")
    print(f"  total_clip_removed_abs_kl (>{tax['clip_threshold']:.1f})  = {tax['total_clip_removed_abs_kl']:.1f}")
    if tax['total_abs_kl'] > 0:
        print(f"  → clip removes {tax['total_clip_removed_abs_kl']/tax['total_abs_kl']*100:.1f}% of total |KL| mass")
    print(f"[diag_kl] wrote {args.out}")


if __name__ == "__main__":
    main_cli()
