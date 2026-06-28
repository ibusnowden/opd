"""eval_passk — pass@k / accuracy / diversity evaluation on a held-out reasoning_gym set.

The harness training loops log only `reward` (one greedy-ish rollout per prompt, scored by reasoning_gym
— a pass@1-flavored / average-accuracy metric). Several roadmap proposals need more: pass@k (does RL/OPD
sharpen the policy onto modes the init already had? Yue et al. 2025 vs ProRL — see `../pass-at-k-vs-pass-at-1.md`),
and a coverage/diversity proxy on the *correct* completions (distinct-n, per-position token-entropy, self-BLEU
if available). This module provides that, reusing the vendored rollout/scoring so "correct" means exactly
what it means in training.

  evaluate_passk(model, tokenizer, dataset, ...) -> dict of  eval/*  metrics  (the in-loop hook calls this)
  CLI:  python -m harness.eval_passk --ckpt allenai/OLMo-2-0425-1B-SFT --task gsm_symbolic \
            --n-prompts 256 --n-samples 64 --k 1,2,4,8,16,32,64 --temps 0.6,1.0

pass@k uses the unbiased estimator (Chen et al. 2021, "Evaluating Large Language Models Trained on Code"):
for a prompt with c correct out of n samples,  pass@k = 1 - C(n-c, k) / C(n, k),  evaluated in the
numerically-stable product form  1 - prod_{i=0}^{k-1} (n-c-i)/(n-i)  ;  then averaged over prompts.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from itertools import batched  # py3.12+

import torch

import reasoning_gym as rg
from reasoning_gym.composite import DatasetSpec
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.utils import extract_answer

from . import _pg


# --- pass@k estimator --------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one prompt: c correct out of n samples (Chen et al. 2021)."""
    if k <= 0 or n <= 0:
        return 0.0
    if c <= 0:
        return 0.0
    if n - c < k:        # fewer than k incorrect → every k-subset hits a correct one
        return 1.0
    # 1 - prod_{i=0}^{k-1} (n-c-i)/(n-i)
    prob_all_wrong = 1.0
    for i in range(k):
        prob_all_wrong *= (n - c - i) / (n - i)
    return 1.0 - prob_all_wrong


# --- diversity proxies (over the CORRECT completions per prompt) -------------

def _distinct_n(texts: list[str], n: int) -> float:
    """distinct-n: unique n-grams / total n-grams, over a pool of texts (token = whitespace word)."""
    total, uniq = 0, set()
    for t in texts:
        toks = t.split()
        grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
        total += len(grams)
        uniq.update(grams)
    return (len(uniq) / total) if total else 0.0


def _self_bleu(texts: list[str]) -> float | None:
    """Mean pairwise BLEU among texts (lower = more diverse). None if no BLEU lib available."""
    if len(texts) < 2:
        return None
    try:
        from sacrebleu import sentence_bleu  # type: ignore
    except Exception:
        try:
            from nltk.translate.bleu_score import sentence_bleu as _nb, SmoothingFunction  # type: ignore
            sf = SmoothingFunction().method1

            def sentence_bleu(hyp, refs):  # noqa: N802 - shadow sacrebleu's signature
                class _S:  # mimic sacrebleu's .score (0-100)
                    score = 100.0 * _nb([r.split() for r in refs], hyp.split(), smoothing_function=sf)
                return _S()
        except Exception:
            return None
    cap = 16  # cap pairs for speed: sample up to `cap` texts
    pool = texts[:cap]
    scores = []
    for i, h in enumerate(pool):
        refs = [pool[j] for j in range(len(pool)) if j != i]
        if refs:
            scores.append(sentence_bleu(h, refs).score / 100.0)
    return (sum(scores) / len(scores)) if scores else None


# --- per-position token entropy along generated tokens -----------------------

@torch.no_grad()
def _mean_token_entropy(model, sequence_ids: torch.Tensor, attention_mask: torch.Tensor,
                        action_mask: torch.Tensor) -> float:
    """Mean (over generated positions) of the next-token-distribution entropy H = -sum_v p_v log p_v."""
    dev = _pg.get_model_device(model)
    out = _pg.unwrap_model(model)(input_ids=sequence_ids.to(dev), attention_mask=attention_mask.to(dev), use_cache=False)
    logits = out.logits[:, :-1, :].to(torch.float32)               # (B, S-1, V)
    logp = torch.log_softmax(logits, dim=-1)
    ent = -(logp.exp() * logp).sum(dim=-1)                          # (B, S-1)
    m = action_mask.to(ent.device).float()
    denom = m.sum().clamp_min(1.0)
    return float((ent * m).sum() / denom)


# --- main eval ---------------------------------------------------------------

@torch.no_grad()
def evaluate_passk(
    model,
    tokenizer,
    dataset: ProceduralDataset,
    *,
    n_prompts: int,
    n_samples: int,
    k_values: list[int],
    temperature: float,
    top_p: float = 0.95,
    top_k: int = 20,
    min_p: float = 0.0,
    max_new_tokens: int = 1024,
    gen_batch_size: int = 32,
    compute_entropy: bool = True,
    compute_self_bleu: bool = True,
    tag: str = "",
) -> dict[str, float]:
    """Sample n_samples completions for each of the first n_prompts in `dataset` at `temperature`,
    score correctness with the reasoning_gym verifier, return pass@k + accuracy + diversity proxies
    as a flat `eval/...` metrics dict.  `model` may be DDP-wrapped (we unwrap for generation)."""
    was_training = _pg.unwrap_model(model).training
    model.eval()
    entries = [dataset[i] for i in range(min(n_prompts, len(dataset)))]
    n_prompts = len(entries)
    expanded = [e for e in entries for _ in range(n_samples)]

    completions: list[str] = []
    ent_sum, ent_n = 0.0, 0
    t0 = time.time()
    for chunk in batched(expanded, gen_batch_size):
        chunk = list(chunk)
        seq_ids, action_mask, attn_mask, _rewards, comps, _accuracy, _format = _pg.rollout(
            model=model, entries=chunk, dataset=dataset, tokenizer=tokenizer,
            max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
        )
        completions.extend(comps)
        if compute_entropy:
            ent_sum += _mean_token_entropy(model, seq_ids, attn_mask, action_mask) * action_mask.size(0)
            ent_n += action_mask.size(0)

    # correctness = the reasoning_gym answer verifier (0/1), reshaped to (n_prompts, n_samples)
    correct = [bool(dataset.score_answer(extract_answer(c), e) >= 1.0) for c, e in zip(completions, expanded)]
    per_prompt_c = [sum(correct[p * n_samples:(p + 1) * n_samples]) for p in range(n_prompts)]

    metrics: dict[str, float] = {}
    for k in k_values:
        if k > n_samples:
            continue
        metrics[f"eval/pass@{k}"] = float(sum(pass_at_k(n_samples, c, k) for c in per_prompt_c) / n_prompts)
    metrics["eval/accuracy_mean"] = float(sum(per_prompt_c) / (n_prompts * n_samples))         # = pass@1 estimate
    metrics["eval/solved_any"] = float(sum(1 for c in per_prompt_c if c > 0) / n_prompts)       # pass@n_samples (≥1 correct)
    metrics["eval/n_prompts"] = float(n_prompts)
    metrics["eval/n_samples"] = float(n_samples)
    metrics["eval/temperature"] = float(temperature)
    metrics["eval/gen_seconds"] = float(time.time() - t0)
    if compute_entropy and ent_n:
        metrics["eval/token_entropy"] = float(ent_sum / ent_n)

    # diversity proxies over the CORRECT completions only (pooled across prompts; cap for speed)
    correct_texts = [c for c, ok in zip(completions, correct) if ok]
    if correct_texts:
        for n in (1, 2, 3, 4):
            metrics[f"eval/distinct_{n}"] = _distinct_n(correct_texts[:4096], n)
        if compute_self_bleu:
            sb = _self_bleu(correct_texts[:64])
            if sb is not None:
                metrics["eval/self_bleu"] = float(sb)

    if tag:
        metrics = {f"{k}/{tag}" if k.startswith("eval/") else k: v for k, v in metrics.items()}
    if was_training:
        model.train()
    return metrics


# --- CLI ---------------------------------------------------------------------

def _eval_dataset(task: str, n_prompts: int, seed: int) -> ProceduralDataset:
    # reasoning_gym's DatasetSpec needs (name, weight, config); config={} -> use the task's defaults.
    return rg.create_dataset("composite", size=n_prompts, seed=seed,
                             datasets=[DatasetSpec(name=task, weight=1.0, config={})])


def main_cli() -> None:
    ap = argparse.ArgumentParser(description="pass@k / accuracy / diversity eval on a held-out reasoning_gym set.")
    ap.add_argument("--ckpt", required=True, help="HF hub name or local path of the model to evaluate")
    ap.add_argument("--task", default="gsm_symbolic", help="reasoning_gym dataset name (default: gsm_symbolic)")
    ap.add_argument("--n-prompts", type=int, default=256)
    ap.add_argument("--n-samples", type=int, default=64, help="completions sampled per prompt (>= max k)")
    ap.add_argument("--k", default="1,2,4,8,16,32,64", help="comma-separated k values for pass@k")
    ap.add_argument("--temps", default="0.6,1.0", help="comma-separated decoding temperatures")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--gen-batch-size", type=int, default=32)
    ap.add_argument("--eval-seed", type=int, default=1_000_000, help="seed for the held-out prompt set (≠ training seed)")
    ap.add_argument("--device", default=None, help="cuda:N or cpu (default: cuda:0 if available)")
    ap.add_argument("--out", default=None, help="path for a JSON dump of the metrics (default: harness/logs/eval_passk_<ckpt>_<task>.json)")
    ap.add_argument("--no-self-bleu", action="store_true")
    args = ap.parse_args()

    k_values = sorted({int(x) for x in args.k.split(",") if x.strip()})
    temps = [float(x) for x in args.temps.split(",") if x.strip()]
    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[eval_passk] ckpt={args.ckpt} task={args.task} device={device} n_prompts={args.n_prompts} "
          f"n_samples={args.n_samples} k={k_values} temps={temps}")
    model, tokenizer = _pg.load_model(args.ckpt, device, gradient_checkpointing=False)
    model.eval()
    if getattr(tokenizer, "chat_template", None) is None:
        raise RuntimeError(f"{args.ckpt!r} has no chat_template — the reasoning_gym rollout needs apply_chat_template "
                           "(use an SFT/Instruct checkpoint, not the bare base model).")

    all_metrics: dict[str, dict[str, float]] = {}
    for T in temps:
        ds = _eval_dataset(args.task, args.n_prompts, args.eval_seed)  # fresh dataset per temp (same seed → same prompts)
        m = evaluate_passk(model, tokenizer, ds, n_prompts=args.n_prompts, n_samples=args.n_samples,
                           k_values=k_values, temperature=T, max_new_tokens=args.max_new_tokens,
                           gen_batch_size=args.gen_batch_size, compute_self_bleu=not args.no_self_bleu)
        all_metrics[f"T={T}"] = m
        line = "  ".join(f"{kk.split('/')[-1]}={vv:.4f}" for kk, vv in sorted(m.items()) if kk.startswith("eval/"))
        print(f"[eval_passk] T={T}: {line}")

    out = args.out or f"harness/logs/eval_passk_{args.ckpt.replace('/', '_')}_{args.task}.json"
    import os
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"ckpt": args.ckpt, "task": args.task, "n_prompts": args.n_prompts, "n_samples": args.n_samples,
                   "k_values": k_values, "metrics_by_temp": all_metrics}, f, indent=2)
    print(f"[eval_passk] wrote {out}")


if __name__ == "__main__":
    main_cli()
