"""Quality gates for authored reasoning traces.

Authored traces cannot be *certified* the way rejection-sampled ones can: I
write them knowing the answer, so they are rationalisation rather than
reasoning that demonstrably produced the answer. What can be guaranteed
mechanically is checked here.

GATES
  leakage      A trace must never quote a HELD string. Held strings are not in
               the prompt, so a trace citing one teaches the model to reference
               evidence it cannot see at inference -- a silent train/test
               mismatch that would look like reasoning and behave like noise.
  conclusion   The pattern the trace commits to must be exactly the gold.
  traps        The trace must mention the reasoning the task actually requires.
  length       Capped, so traces stay trainable on MPS.
  homogeneity  Traces must not collapse into one template; if they do, the
               model learns the template rather than the skill.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MAX_TRACE_CHARS = 1200
# The floor exists to catch contentless traces, not short ones. A terse trace
# on an easy task ("One counted run, no anchors needed. Pattern: ...") is
# proportionate reasoning, and padding it to clear an arbitrary bar would make
# the writing worse. Trace length is expected to scale with difficulty, which
# is reported below rather than enforced.
MIN_TRACE_CHARS = 45
# Below this, traces are near-copies of each other and teach a template.
MIN_DISTINCT_TRIGRAM_RATIO = 0.55


def held_strings(turn: dict) -> list[str]:
    """Strings present in hidden but absent from visible -- the held set."""
    if turn["family"] == "validate":
        hidden = list(turn["hidden"]["positives"]) + list(turn["hidden"]["negatives"])
        visible = list(turn["visible"]["positives"]) + list(turn["visible"]["negatives"])
    else:
        hidden = [c[0] for c in turn["hidden"]]
        visible = [c[0] for c in turn["visible"]]
    return [h for h in hidden if h not in visible]


def trigrams(text: str) -> set[tuple]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i + 3]) for i in range(len(words) - 2)}


def check_record(rec: dict) -> list[str]:
    problems = []
    for t in rec["turns"]:
        trace = t.get("thinking")
        if not trace:
            problems.append(f"{rec['id']} turn {t['index']}: missing thinking")
            continue

        if not (MIN_TRACE_CHARS <= len(trace) <= MAX_TRACE_CHARS):
            problems.append(f"{rec['id']} turn {t['index']}: trace length {len(trace)} "
                            f"outside [{MIN_TRACE_CHARS}, {MAX_TRACE_CHARS}]")

        # -- leakage: no held string may be CITED as a quoted literal.
        #
        # Scope matters here. A bare substring test produces false positives:
        # range bounds such as 65535 appear in the instruction itself, and a
        # test input like 'nothing' collides with ordinary prose. What actually
        # constitutes leakage is a trace presenting an unseen string as
        # evidence -- "this rejects '01.2.3'" -- and traces cite strings in
        # quotes. So the gate looks for quoted citations.
        #
        # Known limit: an unquoted mention would slip through. That is accepted
        # because the authoring convention is to reason from the instruction
        # and never cite test strings at all.
        for h in held_strings(t):
            if len(h) < 2 or h in t["instruction"]:
                # A string named in the instruction is available to the model
                # at inference, so referencing it is reasoning, not leakage.
                continue
            if any(f"{q}{h}{q}" in trace for q in ("'", '"', "`")):
                problems.append(f"{rec['id']} turn {t['index']}: trace cites HELD string {h!r}")

        # -- conclusion must be the gold pattern
        if t["gold"]["pattern"] not in trace:
            problems.append(f"{rec['id']} turn {t['index']}: trace does not state the gold pattern")

    return problems


def trap_words(traps: list) -> set:
    return {w for tr in traps for w in re.findall(r"[a-z]{4,}", tr.lower())}


def check_trap_engagement(records: list[dict]) -> tuple[list[str], float]:
    """Require each BASE TURN to name its traps in at least one phrasing.

    Checking every phrasing individually is the wrong granularity. Each base
    turn has three deliberately different phrasings, and one may explain the
    construction in its own vocabulary -- "a counted quantifier, five digits,
    no anchors needed" engages with exact length without using either word.
    Failing that phrasing would push me to stuff trap keywords into prose to
    satisfy a keyword matcher, which degrades the writing and teaches nothing.

    What matters is that every trap is genuinely reasoned about somewhere in
    the training data for its concept. The per-phrasing rate is reported
    alongside as information, not as a pass/fail condition.
    """
    by_turn: dict[tuple, list[dict]] = {}
    for rec in records:
        for t in rec["turns"]:
            by_turn.setdefault((rec["concept"], t["index"]), []).append(t)

    problems, engaged, total = [], 0, 0
    for (concept, idx), turns in sorted(by_turn.items()):
        words = trap_words(turns[0]["traps"])
        if not words:
            continue
        seen_phrasings = {t.get("trace_phrasing", 0): t["thinking"] for t in turns}
        hits = [p for p, tx in seen_phrasings.items()
                if any(w in tx.lower() for w in words)]
        engaged += len(hits)
        total += len(seen_phrasings)
        if not hits:
            problems.append(f"{concept} turn {idx}: no phrasing engages its traps "
                            f"{turns[0]['traps']}")
    return problems, (engaged / total if total else 1.0)


def main() -> int:
    problems: list[str] = []
    traces: list[str] = []
    train_records: list[dict] = []
    n = 0

    for split in ("train", "dev", "test"):
        d = DATA / split
        if not d.exists():
            continue
        for path in sorted(d.glob("*.json")):
            rec = json.loads(path.read_text())
            has = [t for t in rec["turns"] if t.get("thinking")]
            if split == "train":
                n += 1
                train_records.append(rec)
                problems += check_record(rec)
                traces += [t["thinking"] for t in rec["turns"] if t.get("thinking")]
                if "<think>" not in rec["messages"][2]["content"]:
                    problems.append(f"{rec['id']}: assistant message lacks a <think> block")
            elif has:
                # Traces in dev/test would leak the answer into evaluation.
                problems.append(f"{split}/{rec['id']}: has thinking on {len(has)} turn(s); "
                                f"dev and test must not carry traces")

    print(f"train records checked : {n}")
    print(f"traces checked        : {len(traces)}")

    trap_problems, rate = check_trap_engagement(train_records)
    problems += trap_problems
    print(f"trap engagement       : every base turn covered by >=1 phrasing "
          f"({rate:.0%} of individual phrasings name a trap explicitly)")

    if traces:
        lens = [len(t) for t in traces]
        print(f"trace length          : min={min(lens)} median={sorted(lens)[len(lens) // 2]} max={max(lens)}")

        by_tier: dict[str, list[int]] = {}
        for rec in train_records:
            for t in rec["turns"]:
                if t.get("thinking"):
                    by_tier.setdefault(rec["difficulty"], []).append(len(t["thinking"]))
        print("trace length by tier  : " + "  ".join(
            f"{k}={sum(v) // len(v)}" for k, v in sorted(by_tier.items()) if v))

        # Homogeneity is measured over DISTINCT traces, not instantiated ones.
        # Each authored trace is reused across every variant of its concept, so
        # measuring over instantiations conflates "my writing is repetitive"
        # with "one template legitimately covers several variants". The reuse
        # factor is a real property worth knowing, so it is reported separately
        # rather than folded into the diversity number.
        unique = sorted(set(traces))
        reuse = len(traces) / max(1, len(unique))
        print(f"distinct traces       : {len(unique)} of {len(traces)} "
              f"(each reused ~{reuse:.1f}x across variants of its concept)")

        all_tri: Counter = Counter()
        for t in unique:
            all_tri.update(trigrams(t))
        ratio = len(all_tri) / max(1, sum(all_tri.values()))
        print(f"distinct trigram ratio: {ratio:.3f} over distinct traces "
              f"(min {MIN_DISTINCT_TRIGRAM_RATIO})")
        if ratio < MIN_DISTINCT_TRIGRAM_RATIO:
            problems.append(f"authored traces too homogeneous: ratio {ratio:.3f}; "
                            f"most repeated trigrams {all_tri.most_common(5)}")

    print("\n" + "=" * 62)
    if problems:
        print(f"TRACE GATE FAILED ({len(problems)}):")
        for p in problems[:30]:
            print(f"  - {p}")
        if len(problems) > 30:
            print(f"  ... and {len(problems) - 30} more")
        return 1
    print("TRACE GATE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
