"""RFT generation: roll a teacher on `reasoning_gym` prompts, filter by verifier, save accepted
(prompt, completion, answer) tuples to JSONL for the downstream SFT phase.

This is step 1 of the §7.2 positive-control ablation in `RESULTS.md`: produce a task-specialized
same-base teacher (`-7B-SFT-gsm`) by rejection-sampling solutions from a strong teacher
(`-7B-Instruct` by default), then SFT `-7B-SFT` on those.  If state-coverage is the bottleneck
in OPD (§7.1), the task-specialized teacher should overlap the student's reachable trajectories
better than the off-the-shelf `-7B-SFT` — and pass@1 on the student should climb meaningfully.

Reuses `_pg.rollout` (same generation/scoring path as training), so the verifier scoring is
identical to what Exp 1's RL/OPD arms saw.  Output is one JSONL line per accepted (prompt,
completion); the `templated_prompt` field stores the full chat-template-applied prompt so the
SFT trainer can feed it verbatim.

Disjoint seed.  The dataset seed (`--seed`, default 4242) is disjoint from training (42 / 43) and
held-out eval (1_000_042 / 1_000_043) — so the SFT data and the eval set don't overlap.

Usage:
    python -m harness.rft_generate \
        --teacher allenai/OLMo-2-1124-7B-Instruct \
        --task gsm_symbolic \
        --n_prompts 1500 \
        --n_samples 4 \
        --temperature 1.0 \
        --max_new_tokens 1024 \
        --seed 4242 \
        --output rft_data/gsm_symbolic_from_7B-Instruct_seed4242.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import reasoning_gym as rg
import torch
from reasoning_gym.composite import DatasetSpec
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import _pg


def main() -> None:
    ap = argparse.ArgumentParser(description="RFT generation: teacher rollouts filtered by verifier.")
    ap.add_argument("--teacher", default="allenai/OLMo-2-1124-7B-Instruct",
                    help="HF model id of the teacher whose rollouts we rejection-sample.")
    ap.add_argument("--task", default="gsm_symbolic",
                    help="reasoning_gym task name (single-spec composite).")
    ap.add_argument("--n_prompts", type=int, default=1500,
                    help="Number of distinct dataset prompts to roll.")
    ap.add_argument("--n_samples", type=int, default=4,
                    help="Completions per prompt (higher = more accepted per prompt; teacher pass@1≈0.46 on gsm_symbolic so n=4 yields ~85%% prompts with ≥1 accept).")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Higher temp = more diverse SFT data; 1.0 is the usual RFT default.")
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--min_p", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=4242,
                    help="Dataset seed — keep disjoint from training (42/43) and eval (1_000_042/3) sets.")
    ap.add_argument("--gen_batch_size", type=int, default=16,
                    help="Generations per .generate() call; 7B bf16 + max_new=1024 fits 16-32 on 80 GB H100.")
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument("--output", required=True,
                    help="Output JSONL path (one accepted (prompt, completion) per line).")
    ap.add_argument("--keep_all", action="store_true",
                    help="Save EVERY completion (correct + incorrect) with its accuracy/format labels, "
                         "not just verifier-accepted ones. Used for the unfiltered off-policy-SFT control "
                         "(§8.1): isolates the correctness filter from on-policy-ness by SFT-ing on the same "
                         "teacher rollouts WITHOUT the acc>=1.0 gate. Default off → unchanged RFT behavior.")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print(f"[rft-gen] loading teacher: {args.teacher}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16, attn_implementation=_pg.get_attn_implementation(),
    ).to(device).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[rft-gen] building dataset: task={args.task} n_prompts={args.n_prompts} seed={args.seed}", flush=True)
    dataset = rg.create_dataset(
        "composite", size=args.n_prompts, seed=args.seed,
        datasets=[DatasetSpec(name=args.task, weight=1.0, config={})],
    )
    entries = [entry for entry in dataset for _ in range(args.n_samples)]
    total = len(entries)
    print(f"[rft-gen] {args.n_prompts} prompts × {args.n_samples} samples = {total} generations; "
          f"batch={args.gen_batch_size} → {(total + args.gen_batch_size - 1) // args.gen_batch_size} batches", flush=True)

    accepted: list[dict] = []
    n_total, n_correct, n_kept = 0, 0, 0
    t0 = time.time()
    for chunk_start in range(0, total, args.gen_batch_size):
        chunk = entries[chunk_start:chunk_start + args.gen_batch_size]
        seq_ids, action_mask, attn_mask, rewards, completions, accuracy, format_score = _pg.rollout(
            model=model, entries=chunk, dataset=dataset, tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k, min_p=args.min_p,
        )
        acc_list = accuracy.squeeze(-1).tolist()
        fmt_list = format_score.squeeze(-1).tolist()
        for entry, completion, acc, fmt in zip(chunk, completions, acc_list, fmt_list, strict=True):
            n_total += 1
            if acc >= 1.0:
                n_correct += 1
            if args.keep_all or acc >= 1.0:
                # Re-apply the chat template so the SFT trainer can use `templated_prompt` verbatim.
                from reasoning_gym.utils import SYSTEM_PROMPTS
                templated = tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPTS["DeepSeekZero"]},
                     {"role": "user", "content": entry["question"]}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=True,
                )
                accepted.append({
                    "prompt": entry["question"],
                    "templated_prompt": templated,
                    "completion": completion,
                    "answer": entry.get("answer"),
                    "accuracy": float(acc),
                    "format": float(fmt),
                })
                n_kept += 1
        if (chunk_start // args.gen_batch_size) % 5 == 0:
            elapsed = time.time() - t0
            rate = n_total / max(elapsed, 1e-6)
            eta = (total - n_total) / max(rate, 1e-6) / 60.0
            print(f"[rft-gen] batch {chunk_start // args.gen_batch_size}: "
                  f"{n_total}/{total} done, kept={n_kept}, correct={n_correct} "
                  f"({100 * n_correct / max(n_total, 1):.1f}%), "
                  f"{rate:.1f} gen/s, ETA {eta:.1f} min", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for line in accepted:
            f.write(json.dumps(line) + "\n")

    elapsed = (time.time() - t0) / 60.0
    mode = "keep_all" if args.keep_all else "verifier-filtered"
    print(f"[rft-gen] DONE [{mode}]: kept {n_kept} of {n_total} completions "
          f"(verifier-correct {n_correct}/{n_total} = {100 * n_correct / max(n_total, 1):.1f}%) "
          f"in {elapsed:.1f} min → {output_path}", flush=True)


if __name__ == "__main__":
    main()
