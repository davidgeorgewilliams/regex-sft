"""Render training sequences for both model variants.

PER-TURN EXPANSION
An n-turn trajectory becomes n training examples, one per assistant turn. This
is forced, not chosen: the Thinking model must emit a <think> block on every
turn, so a single sequence carrying loss on all n assistant turns would need
traces in every target. But history must not contain traces. The only
consistent construction is one example per turn, with the trace in the target
position and answer-only history behind it.

The Instruct variant is expanded IDENTICALLY even though it could train as one
sequence. If it did not, the two conditions would differ in sequence count,
token count and gradient steps, and the comparison would be confounded. The
trace must be the only difference.

CHAT TEMPLATE FACTS (verified against the downloaded tokenizers)
  Instruct generation prompt ends: '<|im_start|>assistant\\n'
  Thinking generation prompt ends: '<|im_start|>assistant\\n<think>\\n'
The Thinking template PRE-OPENS the think tag, so its target continues from
inside an open block and must NOT re-emit '<think>'. The template also strips
<think>...</think> out of history automatically -- we pass answer-only history
anyway so both variants see byte-identical context.

Neither template carries {% generation %} markers, so TRL's assistant_only_loss
cannot be used and the loss mask is built by hand here.
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "rendered"

MODELS = {
    "instruct": ROOT / "models" / "original" / "Qwen3-4B-Instruct-2507",
    "thinking": ROOT / "models" / "original" / "Qwen3-4B-Thinking-2507",
}

IM_END = "<|im_end|>"


def build_examples(rec: dict, variant: str) -> list[dict]:
    """Expand one conversation into one example per assistant turn."""
    out = []
    for k, turn in enumerate(rec["turns"]):
        history = [{"role": "system", "content": rec["system"]}]
        for prior in rec["turns"][:k]:
            history.append({"role": "user", "content": prior["user"]})
            # Answer only. The trace is deliberately absent: at inference the
            # model's own earlier reasoning is not in context either, so
            # training on trace-bearing history would create a mismatch that
            # looks like the model forgetting its own reasoning between turns.
            history.append({"role": "assistant", "content": json.dumps(prior["target"])})
        history.append({"role": "user", "content": turn["user"]})

        answer = json.dumps(turn["target"])
        if variant == "thinking":
            # Prompt already ends with an open '<think>\n', so continue inside it.
            target = f"{turn['thinking']}\n</think>\n\n{answer}{IM_END}"
        else:
            target = f"{answer}{IM_END}"

        out.append({
            "id": f"{rec['id']}#t{turn['index']}",
            "record_id": rec["id"],
            "concept": rec["concept"],
            "kind": rec["kind"],
            "family": turn["family"],
            "difficulty": rec["difficulty"],
            "move": turn.get("move", "SINGLE"),
            "turn_index": turn["index"],
            "n_turns": len(rec["turns"]),
            "history": history,
            "target_text": target,
        })
    return out


def tokenize(tok, ex: dict) -> dict:
    """Tokenize prompt+target and build the loss mask.

    Prompt and target are tokenized separately so the boundary is exact, then
    the split is verified against a single-pass tokenization of the whole
    string. If the tokenizer merged a token across the boundary the prefix
    check fails and we say so rather than silently mislabelling one token.
    """
    prompt_text = tok.apply_chat_template(ex["history"], tokenize=False,
                                          add_generation_prompt=True)
    full_text = prompt_text + ex["target_text"]

    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tok(full_text, add_special_tokens=False)["input_ids"]

    clean = full_ids[:len(prompt_ids)] == prompt_ids
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

    return {
        "input_ids": full_ids,
        "labels": labels,
        "prompt_len": len(prompt_ids),
        "target_len": len(full_ids) - len(prompt_ids),
        "boundary_clean": clean,
        "prompt_text": prompt_text,
    }


def dump_mask(tok, ex: dict, enc: dict, path: Path, limit_prompt: int = 24) -> None:
    """Write a token-by-token view of what carries loss.

    Silent mask bugs are the most common way an SFT run quietly does nothing,
    and they are invisible in the loss curve. Reading the decoded mask is the
    only way to be sure.
    """
    lines = [
        f"example : {ex['id']}",
        f"variant : {path.stem}",
        f"tokens  : {len(enc['input_ids'])}  "
        f"(prompt {enc['prompt_len']} masked / target {enc['target_len']} trained)",
        f"boundary clean (no token merged across prompt|target): {enc['boundary_clean']}",
        "",
        "  MASK = label -100, contributes no loss.  TRAIN = contributes loss.",
        "=" * 78,
    ]
    ids, labels = enc["input_ids"], enc["labels"]
    p = enc["prompt_len"]
    show = list(range(min(limit_prompt, p))) + ["..."] + list(range(max(0, p - 8), len(ids)))
    for i in show:
        if i == "...":
            lines.append(f"      ...  ({p - limit_prompt - 8} more masked prompt tokens)")
            continue
        tag = "MASK " if labels[i] == -100 else "TRAIN"
        lines.append(f"  {i:>5}  {tag}  {tok.decode([ids[i]])!r}")
    lines += ["=" * 78, "", "FULL PROMPT TEXT", "-" * 78, enc["prompt_text"],
              "", "TARGET TEXT", "-" * 78, ex["target_text"]]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    records = [json.loads(p.read_text()) for p in sorted((DATA / "train").glob("*.json"))]
    OUT.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    summary = {}

    for variant, model_dir in MODELS.items():
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        examples, lengths, tgt_lengths = [], [], []
        unclean = 0

        for rec in records:
            for ex in build_examples(rec, variant):
                enc = tokenize(tok, ex)
                if not enc["boundary_clean"]:
                    unclean += 1
                if enc["target_len"] <= 0:
                    problems.append(f"{variant}/{ex['id']}: empty target")
                examples.append({
                    **{k: ex[k] for k in ("id", "record_id", "concept", "kind", "family",
                                          "difficulty", "move", "turn_index", "n_turns")},
                    "input_ids": enc["input_ids"],
                    "labels": enc["labels"],
                    "prompt_len": enc["prompt_len"],
                    "target_len": enc["target_len"],
                })
                lengths.append(len(enc["input_ids"]))
                tgt_lengths.append(enc["target_len"])

        if unclean:
            problems.append(f"{variant}: {unclean} examples had a token merged across "
                            f"the prompt|target boundary")

        path = OUT / f"train_{variant}.jsonl"
        with open(path, "w") as f:
            for e in examples:
                f.write(json.dumps(e) + "\n")

        # Mask dumps: one single-turn example and one final-turn trajectory example.
        single = next(e for e in examples if e["kind"] == "single")
        multi = next(e for e in examples if e["kind"] == "trajectory" and e["turn_index"] == 3)
        for tag, want in (("single", single), ("turn3", multi)):
            ex = next(x for r in records for x in build_examples(r, variant) if x["id"] == want["id"])
            dump_mask(tok, ex, tokenize(tok, ex), OUT / f"mask_{variant}_{tag}.txt")

        lengths.sort()
        summary[variant] = {
            "examples": len(examples),
            "seq_len_median": lengths[len(lengths) // 2],
            "seq_len_p95": lengths[int(len(lengths) * 0.95)],
            "seq_len_max": lengths[-1],
            "target_len_mean": round(sum(tgt_lengths) / len(tgt_lengths), 1),
            "total_target_tokens": sum(tgt_lengths),
        }

    print("RENDERED")
    for v, s in summary.items():
        print(f"  {v:<9} examples={s['examples']:<5} seq_len med={s['seq_len_median']:<5} "
              f"p95={s['seq_len_p95']:<5} max={s['seq_len_max']:<5} "
              f"target_mean={s['target_len_mean']:<6} target_total={s['total_target_tokens']:,}")

    a, b = summary["instruct"], summary["thinking"]
    print(f"\n  example counts equal across variants: {a['examples'] == b['examples']} "
          f"({a['examples']} each)")
    if a["examples"] != b["examples"]:
        problems.append("variants have different example counts -- comparison confounded")
    ratio = b["total_target_tokens"] / max(1, a["total_target_tokens"])
    print(f"  thinking trains on {ratio:.1f}x the target tokens of instruct "
          f"(the traces; this is the intended difference)")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n" + "=" * 70)
    if problems:
        print(f"RENDER FAILED ({len(problems)}):")
        for p in problems[:20]:
            print(f"  - {p}")
        return 1
    print("RENDER OK -- mask dumps written to data/rendered/mask_*.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
