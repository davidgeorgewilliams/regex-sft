"""Generate from a checkpoint and score with the execution verifier.

Scoring uses the same oracle that built the data, so there is no judge and no
human in this loop. Metrics reported:

  hidden_pass      the headline: the pattern satisfies strings never shown
  visible_pass     satisfies only the examples in the prompt
  overfit          visible_pass AND NOT hidden_pass -- reproduced the examples
                   without honouring the instruction. Directly measurable.
  json_valid       produced parseable output in the required shape
  compiled         produced a regex that compiles at all

MULTI-TURN
Two modes, both reported:
  gold      history contains the reference answers, so each turn is scored
            independently and a turn-3 failure is attributable to turn 3.
  free      history contains the model's own earlier answers, so errors
            compound. Realistic, but a failure cannot be localised.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verifier import verify

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

MODEL_DIR = {
    "instruct": ROOT / "models" / "original" / "Qwen3-4B-Instruct-2507",
    "thinking": ROOT / "models" / "original" / "Qwen3-4B-Thinking-2507",
}
MAX_NEW = {"instruct": 128, "thinking": 384}

# Sampling settings taken verbatim from each model card's "Best Practices".
# Used for BASELINE measurement, where the question is what the model can do
# under its intended operating conditions. Checkpoint comparison uses greedy
# instead, because the dev sweep needs determinism to select between epochs
# that differ by a point or two -- two different purposes, two configs, both
# stated rather than one default silently carried across.
RECOMMENDED = {
    "instruct": dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0),
    "thinking": dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0),
}
# Both cards: "We recommend using an output length of 32,768 tokens for most
# queries." Anything smaller risks truncating mid-reasoning, which measures the
# budget rather than the model.
RECOMMENDED_MAX_NEW = 32768


def parse_output(text: str) -> dict | None:
    """Pull the JSON answer out of a generation.

    For the thinking variant the model continues from an already-open <think>,
    so the answer follows the LAST '</think>'. Falling back to brace matching
    keeps a stray prefix from breaking an otherwise valid answer.
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = text.split("<|im_end|>")[0]
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) and "pattern" in obj else None


def build_prompt(tok, rec: dict, k: int, prior_answers: list[str]) -> str:
    msgs = [{"role": "system", "content": rec["system"]}]
    for i, prior in enumerate(rec["turns"][:k]):
        msgs.append({"role": "user", "content": prior["user"]})
        msgs.append({"role": "assistant", "content": prior_answers[i]})
    msgs.append({"role": "user", "content": rec["turns"][k]["user"]})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new: int, batch_size: int,
             gen_kwargs: dict | None = None) -> tuple[list[str], list[int]]:
    outs, lens = [], []
    kw = gen_kwargs or dict(do_sample=False)
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("mps")
        gen = model.generate(**enc, max_new_tokens=max_new,
                             pad_token_id=tok.pad_token_id, **kw)
        plen = enc["input_ids"].shape[1]
        for j in range(len(chunk)):
            new = gen[j][plen:]
            text = tok.decode(new, skip_special_tokens=False)
            outs.append(text)
            # Count real tokens, excluding right-padding added after EOS.
            lens.append(int((new != tok.pad_token_id).sum()))
        print(f"    batch {i // batch_size + 1}/"
              f"{(len(prompts) + batch_size - 1) // batch_size} "
              f"gen_len max={max(lens[-len(chunk):])}", flush=True)
    return outs, lens


def score(turn: dict, cand: dict | None) -> dict:
    if cand is None:
        return {"json_valid": False, "compiled": False,
                "visible_pass": False, "hidden_pass": False, "overfit": False}
    vis = verify(cand, {"family": turn["family"], "visible": turn["visible"]}, "visible")
    hid = verify(cand, {"family": turn["family"], "hidden": turn["hidden"]}, "hidden")
    return {"json_valid": True, "compiled": vis.compiled,
            "visible_pass": vis.passed, "hidden_pass": hid.passed,
            "overfit": vis.passed and not hid.passed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["instruct", "thinking"])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--checkpoint", default="base",
                    help="'base' for the untrained model, else a path to a LoRA adapter")
    ap.add_argument("--mode", default="gold", choices=["gold", "free"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    # The UNTRAINED thinking model reasons at length -- Qwen suggests a 32k
    # output budget for it -- so the 384 default truncated 100% of its baseline
    # generations and scored the token cap rather than the model. Fine-tuned
    # checkpoints emit short traces and are unaffected.
    ap.add_argument("--max-new", type=int, default=0, help="override MAX_NEW")
    ap.add_argument("--decode", default="greedy", choices=["greedy", "recommended"],
                    help="greedy = deterministic, for checkpoint comparison; "
                         "recommended = the model card's sampling settings, for baselines")
    ap.add_argument("--seed", type=int, default=20260819)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.mps.manual_seed(args.seed)
    if args.decode == "recommended":
        gen_kwargs = dict(RECOMMENDED[args.variant])
        max_new = args.max_new or RECOMMENDED_MAX_NEW
    else:
        gen_kwargs = dict(do_sample=False)
        max_new = args.max_new or MAX_NEW[args.variant]

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR[args.variant]))
    tok.padding_side = "left"          # required for correct batched generation
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR[args.variant]), dtype=torch.bfloat16, attn_implementation="eager")
    if args.checkpoint != "base":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint)
    model.to("mps").eval()
    model.config.use_cache = True

    records = [json.loads(p.read_text()) for p in sorted((DATA / args.split).glob("*.json"))]
    if args.limit:
        records = records[:args.limit]

    t0 = time.time()
    rows = []
    max_turns = max(len(r["turns"]) for r in records)
    prior: dict[str, list[str]] = {r["id"]: [] for r in records}

    # Turn-major so every record's turn k is generated in one batched pass.
    for k in range(max_turns):
        active = [r for r in records if len(r["turns"]) > k]
        if not active:
            continue
        prompts = [build_prompt(tok, r, k, prior[r["id"]]) for r in active]
        texts, glens = generate(model, tok, prompts, max_new, args.batch_size, gen_kwargs)

        for r, text, glen in zip(active, texts, glens):
            turn = r["turns"][k]
            cand = parse_output(text)
            s = score(turn, cand)
            rows.append({
                "id": r["id"], "turn": turn["index"], "concept": r["concept"],
                "kind": r["kind"], "family": turn["family"],
                "difficulty": r["difficulty"], "move": turn.get("move", "SINGLE"),
                "truncated": "<|im_end|>" not in text,
                "gen_tokens": glen,
                # Keep a slice of raw output. Without it, a truncation flag tells
                # you THAT generation failed but not what it was doing -- coherent
                # long reasoning and a repetition loop look identical in the metrics.
                "gen_head": text[:600],
                **s,
            })
            # gold mode feeds the reference answer forward; free mode feeds the
            # model's own answer, so errors compound across turns.
            prior[r["id"]].append(
                json.dumps(turn["target"]) if args.mode == "gold"
                else (json.dumps(cand) if cand else "{}"))
        print(f"  turn {k + 1}: {len(active)} generated "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)

    def rate(key, subset=None):
        sel = [r for r in rows if subset is None or subset(r)]
        return round(100 * sum(r[key] for r in sel) / len(sel), 1) if sel else None

    summary = {
        "variant": args.variant, "split": args.split, "checkpoint": args.checkpoint,
        "mode": args.mode, "n_turns": len(rows), "max_new": max_new,
        "decode": args.decode, "seed": args.seed,
        "gen_tokens_mean": round(sum(r["gen_tokens"] for r in rows) / max(1, len(rows)), 1),
        "gen_tokens_max": max((r["gen_tokens"] for r in rows), default=0),
        "seconds": round(time.time() - t0),
        "hidden_pass": rate("hidden_pass"),
        "visible_pass": rate("visible_pass"),
        "overfit": rate("overfit"),
        "json_valid": rate("json_valid"),
        "truncated": rate("truncated"),
        "by_difficulty": {d: rate("hidden_pass", lambda r, d=d: r["difficulty"] == d)
                          for d in ("easy", "medium", "hard")},
        "by_family": {f: rate("hidden_pass", lambda r, f=f: r["family"] == f)
                      for f in ("validate", "extract", "substitute")},
        "by_kind": {k: rate("hidden_pass", lambda r, k=k: r["kind"] == k)
                    for k in ("single", "trajectory")},
        "by_turn": {str(t): rate("hidden_pass", lambda r, t=t: r["turn"] == t)
                    for t in (1, 2, 3)},
        "by_move": {m: rate("hidden_pass", lambda r, m=m: r["move"] == m)
                    for m in sorted({r["move"] for r in rows})},
    }

    RESULTS.mkdir(exist_ok=True)
    name = args.out or f"{args.variant}_{args.split}_{Path(args.checkpoint).name}_{args.mode}.json"
    (RESULTS / name).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
