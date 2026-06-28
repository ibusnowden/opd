"""hint_search — Exp 8 stage 1: GEPA-style per-task hint search over the Lagrangian.

Implements per-task-hint-search-gepa.md (roadmap tier 2, the "best near-term intervention idea"):
search the space of FIXED task-level conditioning strings `c_T` for the same-family teacher, scoring
each candidate on the Lagrangian  E[Δreward] − β·KL  — *no training in the loop*. The selected hint
feeds stage 2 (a real OPD run with `teacher.condition_on=fixed_hint`, config exp8_fixed_hint_opd.yaml).

The post-§7.10/§7.12 framing sharpens the original question: Exp 5 showed *per-problem answer*
conditioning rescues pure OPD from the dead corner (0.029 → 0.188/0.252) but spends the §8.4
unbiasedness leg (privileged info). A searched *task-level* hint carries NO per-problem info — if it
rescues pure OPD too, the §7.10 mechanism is "shift the teacher's distribution onto answer-shaped
paths" generally, not privileged information per se; if it doesn't, OPSD's rescue really is about
per-problem privilege.

Scoring a candidate hint `c` (everything against ONE fixed set of student rollouts, generated once):
  * `teacher_acc` — Δreward proxy: verifier accuracy of the *hinted teacher's own generations* on a
    dev-pool subset. OPD pulls the student toward the hinted teacher's distribution, so the hinted
    teacher's accuracy upper-bounds what imitation can buy (cf. §7.2 teacher competence).
  * `kl_mean` / `kl_p99` — KL pull: |log π_T^c − log π_θ| per-token on the FIXED student rollouts,
    mean and 99th percentile. p99 is the §8.2-relevant statistic: the heavy tail of per-token pushes
    is what drives the §7.6 collapse, so a "surgical" hint is one with high acc and a SMALL tail.
  * `disc` — secondary: mean per-token log π_T^c on verifier-correct student rollouts minus on
    incorrect ones (does the hinted teacher selectively prefer reward-bearing continuations?).
    None when the cold student produces too few correct rollouts to estimate it.
  * `score` = teacher_acc − β·kl_p99 (the Lagrangian; β re-pickable offline — all components saved).

Outer loop (GEPA-ish): seed population → score → reflective mutation by an Instruct sibling shown the
scored table (asks for K new candidate hints as a JSON array) → keep elites → repeat. The final JSON
carries every candidate ever scored, the Pareto front (acc vs kl_p99), and the β-ranking.

  python -m harness.hint_search --out results/exp8_hint_search/search.json          # full search
  python -m harness.hint_search --smoke                                              # tiny plumbing run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from itertools import batched  # py3.12+

import torch
from transformers import GenerationConfig

import reasoning_gym as rg
from reasoning_gym.composite import DatasetSpec
from reasoning_gym.utils import SYSTEM_PROMPTS, extract_answer

from . import _pg
from .config import TeacherSpec
from .teachers import PrivilegedInfoTeacher


# --- seed population (generation 0) ------------------------------------------
# "" = the no-hint reference (≈ plain OPD teacher); the rest span the obvious axes:
# CoT, structure, verification, brevity, persona, format.
SEED_HINTS = [
    "",
    "Think step by step and show every arithmetic operation explicitly.",
    "First restate the given quantities, then define what is asked, compute intermediate "
    "results one at a time, and verify the final answer by substituting it back.",
    "Be concise: give only the minimal chain of calculations needed, then the final answer.",
    "Identify the type of word problem and the operations it needs before doing any arithmetic.",
    "You are a meticulous grade-school math teacher demonstrating a clean worked solution.",
    "Double-check each intermediate result before using it in the next step.",
    "Work the problem in short numbered steps, one calculation per step.",
]

REFLECT_TEMPLATE = """You are optimizing a short instruction ("hint") that will be appended to grade-school math word problems given to a teacher language model. The teacher's hinted token distribution is then distilled into a weaker student, so a GOOD hint must:
1. make the teacher solve the problems CORRECTLY (high accuracy), and
2. keep the teacher's wording/style CLOSE to an unhinted model (low KL pull) — surgical, not disruptive.

Here are hints tried so far, with their measured teacher accuracy (higher is better) and KL tail kl_p99 (lower is better):

{table}

Propose {k} NEW hint strings that could beat the best ones above. Make them diverse: vary structure, specificity, and length (under 280 characters each). Do not repeat or trivially rephrase the hints above.

Output ONLY a JSON array of {k} strings, nothing else."""


# --- dataset / rollouts --------------------------------------------------------

def _make_dataset(task: str, size: int, seed: int):
    return rg.create_dataset("composite", size=size, seed=seed,
                             datasets=[DatasetSpec(name=task, weight=1.0, config={})])


@torch.no_grad()
def collect_student_rollouts(student, tokenizer, dataset, entries, *, n_rollouts: int,
                             max_new_tokens: int, temperature: float, gen_batch_size: int):
    """Generate the FIXED student rollout set once: returns a list of per-chunk dicts holding the
    padded tensors (CPU), the matching entries, per-seq verifier correctness, and the student's own
    per-token logprobs — everything every candidate hint is scored against."""
    expanded = [e for e in entries for _ in range(n_rollouts)]
    chunks = []
    n_correct = 0
    for chunk in batched(expanded, gen_batch_size):
        chunk = list(chunk)
        seq_ids, action_mask, attn_mask, _r, comps, _acc, _fmt = _pg.rollout(
            model=student, entries=chunk, dataset=dataset, tokenizer=tokenizer,
            max_new_tokens=max_new_tokens, temperature=temperature, top_p=0.95, top_k=20, min_p=0.0,
        )
        lp_student = _pg.compute_log_probs(student, seq_ids, attn_mask)  # (B, S-1)
        correct = [bool(dataset.score_answer(extract_answer(c), e) >= 1.0) for c, e in zip(comps, chunk)]
        n_correct += sum(correct)
        chunks.append({
            "seq_ids": seq_ids.cpu(), "attn": attn_mask.cpu(), "amask": action_mask.cpu(),
            "lp_student": lp_student.float().cpu(), "entries": chunk, "correct": correct,
        })
    return chunks, n_correct, len(expanded)


# --- per-candidate scoring -----------------------------------------------------

@torch.no_grad()
def teacher_gen_accuracy(teacher_model, tokenizer, dataset, entries, hint: str, *,
                         n_samples: int, max_new_tokens: int, temperature: float,
                         gen_batch_size: int) -> float:
    """Verifier accuracy of the hinted teacher's own generations (the Δreward proxy).
    Prompt construction mirrors `PrivilegedInfoTeacher._conditioned_logprobs` exactly."""
    system_prompt = SYSTEM_PROMPTS["DeepSeekZero"]
    suffix = f"\n\n{hint}" if hint else ""
    expanded = [e for e in entries for _ in range(n_samples)]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gcfg = GenerationConfig(temperature=temperature, top_p=0.95, top_k=20, do_sample=True,
                            max_new_tokens=max_new_tokens, pad_token_id=pad_id)
    dev = _pg.get_model_device(teacher_model)
    n_ok = 0
    for chunk in batched(expanded, gen_batch_size):
        chunk = list(chunk)
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": e["question"] + suffix}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True)
            for e in chunk
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left",
                           return_attention_mask=True).to(dev)
        out = teacher_model.generate(**inputs, generation_config=gcfg)
        comps = tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        n_ok += sum(bool(dataset.score_answer(extract_answer(c), e) >= 1.0) for c, e in zip(comps, chunk))
    return n_ok / max(1, len(expanded))


@torch.no_grad()
def score_hint(hint: str, teacher: PrivilegedInfoTeacher, rollout_chunks: list[dict],
               dataset, gen_entries, args) -> dict:
    """Score one candidate against the fixed student rollouts + a hinted-teacher generation pass."""
    t0 = time.time()
    hint_fn = (lambda _e: f"\n\n{hint}") if hint else (lambda _e: "")

    abs_d_all: list[torch.Tensor] = []
    per_seq_lpT: list[float] = []
    per_seq_correct: list[bool] = []
    for ch in rollout_chunks:
        lp_T = teacher._conditioned_logprobs(ch["seq_ids"], ch["attn"], ch["amask"], ch["entries"], hint_fn)
        m = ch["amask"].bool()
        d = (lp_T - ch["lp_student"])[m]                      # per-token reverse-KL push the OPD update would apply
        abs_d_all.append(d.abs())
        for i in range(m.shape[0]):
            row = m[i]
            if row.any():
                per_seq_lpT.append(float(lp_T[i][row].mean()))
                per_seq_correct.append(ch["correct"][i])

    abs_d = torch.cat(abs_d_all) if abs_d_all else torch.zeros(1)
    kl_mean = float(abs_d.mean())
    kl_p99 = float(torch.quantile(abs_d, 0.99)) if abs_d.numel() > 1 else float(abs_d.max())

    lp_ok = [v for v, c in zip(per_seq_lpT, per_seq_correct) if c]
    lp_bad = [v for v, c in zip(per_seq_lpT, per_seq_correct) if not c]
    disc = (sum(lp_ok) / len(lp_ok) - sum(lp_bad) / len(lp_bad)) if (len(lp_ok) >= args.min_disc_support and lp_bad) else None

    teacher_acc = teacher_gen_accuracy(
        teacher._ensure_model(), teacher._tokenizer, dataset, gen_entries, hint,
        n_samples=args.gen_samples, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, gen_batch_size=args.gen_batch_size)

    return {"hint": hint, "teacher_acc": teacher_acc, "kl_mean": kl_mean, "kl_p99": kl_p99,
            "disc": disc, "score": teacher_acc - args.beta * kl_p99, "seconds": round(time.time() - t0, 1)}


# --- reflective mutation (the GEPA-ish bit) ------------------------------------

def _parse_hint_array(text: str, cap_len: int = 280) -> list[str]:
    """Best-effort: pull a JSON array of strings out of the mutator's generation."""
    out: list[str] = []
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            out = [str(x) for x in arr if isinstance(x, str)]
        except json.JSONDecodeError:
            pass
    if not out:  # fallback: quoted lines
        out = re.findall(r'"([^"\n]{10,})"', text)
    return [h.strip()[:cap_len] for h in out if h.strip()]


@torch.no_grad()
def propose_hints(mutator_model, mutator_tok, scored: list[dict], k: int, *,
                  max_new_tokens: int = 600) -> list[str]:
    """Show the mutator the scored table (best 4 + worst 2), ask for k new candidates."""
    by_score = sorted(scored, key=lambda r: r["score"], reverse=True)
    show = by_score[:4] + by_score[-2:] if len(by_score) > 6 else by_score
    table = "\n".join(
        f'- hint: "{r["hint"] or "(no hint)"}"  ->  accuracy={r["teacher_acc"]:.3f}  kl_p99={r["kl_p99"]:.3f}'
        for r in show)
    prompt = mutator_tok.apply_chat_template(
        [{"role": "user", "content": REFLECT_TEMPLATE.format(table=table, k=k)}],
        tokenize=False, add_generation_prompt=True)
    dev = _pg.get_model_device(mutator_model)
    inputs = mutator_tok(prompt, return_tensors="pt").to(dev)
    pad_id = mutator_tok.pad_token_id if mutator_tok.pad_token_id is not None else mutator_tok.eos_token_id
    out = mutator_model.generate(**inputs, generation_config=GenerationConfig(
        temperature=0.9, top_p=0.95, do_sample=True, max_new_tokens=max_new_tokens, pad_token_id=pad_id))
    text = mutator_tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return _parse_hint_array(text)


def pareto_front(scored: list[dict]) -> list[str]:
    """Hints not dominated in (teacher_acc max, kl_p99 min)."""
    front = []
    for r in scored:
        if not any((o["teacher_acc"] >= r["teacher_acc"] and o["kl_p99"] <= r["kl_p99"] and
                    (o["teacher_acc"] > r["teacher_acc"] or o["kl_p99"] < r["kl_p99"])) for o in scored):
            front.append(r["hint"])
    return front


# --- main -----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Exp 8 stage 1: GEPA-style per-task hint search (no training).")
    ap.add_argument("--student", default="allenai/OLMo-2-0425-1B-SFT")
    ap.add_argument("--teacher", default="allenai/OLMo-2-1124-7B-SFT",
                    help="scoring teacher (= the stage-2 OPD teacher; 7B-SFT matches the §7.10 OPSD refs)")
    ap.add_argument("--mutator", default="allenai/OLMo-2-1124-7B-Instruct",
                    help="reflective-mutation LLM (proposes new hint candidates)")
    ap.add_argument("--task", default="gsm_symbolic")
    ap.add_argument("--seed", type=int, default=42, help="dev-pool seed (= the training pool; ≠ eval seed 1e6)")
    ap.add_argument("--n-prompts", type=int, default=64, help="dev-pool problems for the KL/disc estimates")
    ap.add_argument("--n-rollouts", type=int, default=4, help="student rollouts per dev problem (fixed set)")
    ap.add_argument("--gen-prompts", type=int, default=32, help="dev-pool subset for hinted-teacher generation")
    ap.add_argument("--gen-samples", type=int, default=2, help="hinted-teacher generations per gen-prompt")
    ap.add_argument("--population", type=int, default=6, help="elites kept per generation")
    ap.add_argument("--children", type=int, default=6, help="new mutator proposals per generation")
    ap.add_argument("--generations", type=int, default=3, help="mutation rounds after scoring the seeds")
    ap.add_argument("--beta", type=float, default=0.2, help="Lagrangian KL-tail weight: score = acc - beta*kl_p99")
    ap.add_argument("--min-disc-support", type=int, default=5, help="min correct rollouts to report disc")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--gen-batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny plumbing run (4 prompts, 3 seed hints, 1 generation)")
    args = ap.parse_args()

    if args.smoke:
        args.n_prompts, args.n_rollouts = 4, 2
        args.gen_prompts, args.gen_samples = 4, 1
        args.children, args.generations = 2, 1
        args.max_new_tokens = 256

    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seeds = SEED_HINTS[:3] if args.smoke else SEED_HINTS
    print(f"[hint_search] task={args.task} student={args.student} teacher={args.teacher} "
          f"mutator={args.mutator} device={device} beta={args.beta} "
          f"pool={args.n_prompts}x{args.n_rollouts} gen={args.gen_prompts}x{args.gen_samples} "
          f"seeds={len(seeds)} +{args.children}x{args.generations}", flush=True)

    _pg.seed_everything(args.seed)
    dataset = _make_dataset(args.task, max(args.n_prompts, args.gen_prompts), args.seed)
    entries = [dataset[i] for i in range(args.n_prompts)]
    gen_entries = entries[:args.gen_prompts]

    # 1) fixed student rollout set (generated once; every candidate is scored against it)
    student, stu_tok = _pg.load_model(args.student, device, gradient_checkpointing=False)
    student.eval()
    t0 = time.time()
    rollout_chunks, n_ok, n_total = collect_student_rollouts(
        student, stu_tok, dataset, entries, n_rollouts=args.n_rollouts,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        gen_batch_size=args.gen_batch_size)
    print(f"[hint_search] student rollouts: {n_total} seqs, {n_ok} verifier-correct "
          f"({n_ok / max(1, n_total):.3f}), {time.time() - t0:.0f}s", flush=True)
    del student
    torch.cuda.empty_cache()

    # 2) scoring teacher (PrivilegedInfoTeacher gives us the hint-conditioned forward + slice-back)
    teacher = PrivilegedInfoTeacher(
        TeacherSpec(kind="self", model_name=args.teacher, condition_on="fixed_hint",
                    fixed_hint="(set per-candidate)", device_id=device.index or 0, frozen_at_init=True),
        student_model_name=args.student)
    teacher._ensure_model()

    # 3) mutator (co-resident; 1B+7B+7B bf16 ≈ 30 GB)
    mutator_model, mutator_tok = _pg.load_model(args.mutator, device, gradient_checkpointing=False)
    mutator_model.eval()

    scored: list[dict] = []
    seen: set[str] = set()

    def _score_batch(cands: list[str], tag: str) -> None:
        for h in cands:
            key = h.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            r = score_hint(h, teacher, rollout_chunks, dataset, gen_entries, args)
            r["origin"] = tag
            scored.append(r)
            print(f'[hint_search] {tag}  acc={r["teacher_acc"]:.3f} kl_p99={r["kl_p99"]:.3f} '
                  f'kl_mean={r["kl_mean"]:.3f} disc={r["disc"] if r["disc"] is None else round(r["disc"], 4)} '
                  f'score={r["score"]:.3f} ({r["seconds"]}s)  "{h or "(no hint)"}"', flush=True)

    _score_batch(seeds, "gen0/seed")
    for g in range(1, args.generations + 1):
        children = propose_hints(mutator_model, mutator_tok, scored, args.children)
        print(f"[hint_search] generation {g}: mutator proposed {len(children)} candidates", flush=True)
        _score_batch(children, f"gen{g}/mutated")

    ranking = sorted(scored, key=lambda r: r["score"], reverse=True)
    best = ranking[0]
    print(f'[hint_search] BEST (beta={args.beta}): acc={best["teacher_acc"]:.3f} kl_p99={best["kl_p99"]:.3f} '
          f'score={best["score"]:.3f}  "{best["hint"]}"', flush=True)

    out = args.out or f"results/exp8_hint_search/search_{args.task}_seed{args.seed}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "args": vars(args),
            "student_rollouts": {"n_total": n_total, "n_correct": n_ok},
            "candidates": ranking,
            "pareto_front_hints": pareto_front(scored),
            "best_hint": best["hint"],
            "no_hint_reference": next((r for r in scored if r["hint"] == ""), None),
        }, f, indent=2)
    print(f"[hint_search] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
