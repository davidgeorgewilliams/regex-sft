"""Composition gate for the spec library.

The correctness gates (check_concepts / check_arcs) prove every gold answer
executes correctly. They say nothing about whether the library is *balanced* --
and an earlier draft passed correctness with 21 NARROW moves against 1 ROLLBACK,
and with only 4 of 70 concepts requiring any regex flag. Verified-correct data
can still be badly distributed, so composition gets its own gate.

Targets are asserted, not merely reported: the build fails if they are unmet.
"""

from __future__ import annotations

from collections import Counter

import concepts
from arcs import ARCS

# Uniform move coverage is deliberate: it buys equal statistical power per move
# for the question "which conversational moves does SFT improve?". It is NOT
# natural frequency -- in real use NARROW dominates. Report the trade, do not
# imply uniformity is free.
MOVE_TARGET = 6
MIN_CONCEPTS_PER_TIER = 3
FLAG_TARGETS = {"m": 8, "i": 4, "s": 3}
MIN_LOOKAROUND = 6


def all_concepts() -> list[tuple[str, dict]]:
    return ([("validate", c) for c in concepts.VALIDATE]
            + [("extract", c) for c in concepts.EXTRACT]
            + [("substitute", c) for c in concepts.SUBSTITUTE])


def tier(traps: list) -> str:
    return {1: "easy", 2: "medium"}.get(len(traps), "hard")


def main() -> int:
    problems: list[str] = []

    # --- move balance across arcs -----------------------------------------
    moves_a, moves_b = Counter(), Counter()
    for arc in ARCS:
        a, b = arc["arc"].split("->")
        moves_a[a] += 1
        moves_b[b] += 1
    totals = moves_a + moves_b

    print(f"MOVE COVERAGE (target {MOVE_TARGET} each)")
    for move in sorted(totals, key=lambda m: -totals[m]):
        t = totals[move]
        mark = "ok " if t == MOVE_TARGET else "OFF"
        print(f"  {mark} {move:<11} total={t:<3} turn2={moves_a[move]:<3} turn3={moves_b[move]}")
        if t != MOVE_TARGET:
            problems.append(f"move {move} appears {t}x, target {MOVE_TARGET}")

    seqs = Counter(a["arc"] for a in ARCS)
    dupes = {k: v for k, v in seqs.items() if v > 1}
    print(f"  distinct move sequences: {len(seqs)}/{len(ARCS)}"
          + (f"  DUPLICATES: {dupes}" if dupes else ""))
    if dupes:
        problems.append(f"duplicate move sequences: {dupes}")

    # --- difficulty spread must not be confounded with family -------------
    print(f"\nDIFFICULTY x FAMILY (min {MIN_CONCEPTS_PER_TIER} per cell)")
    grid: Counter = Counter()
    for fam, c in all_concepts():
        grid[(fam, tier(c["traps"]))] += 1
    for fam in ("validate", "extract", "substitute"):
        row = {t: grid[(fam, t)] for t in ("easy", "medium", "hard")}
        bad = [t for t, n in row.items() if n < MIN_CONCEPTS_PER_TIER]
        print(f"  {'ok ' if not bad else 'OFF'} {fam:<11} {row}")
        for t in bad:
            problems.append(f"{fam}/{t} has {row[t]} concepts, min {MIN_CONCEPTS_PER_TIER}")

    # --- flag coverage ----------------------------------------------------
    print("\nFLAG COVERAGE (concepts + arc turns)")
    flag_use: Counter = Counter()
    for _, c in all_concepts():
        for ch in c["gold"].get("flags", ""):
            flag_use[ch] += 1
    for arc in ARCS:
        for turn in arc["turns"]:
            for ch in turn["gold"].get("flags", ""):
                flag_use[ch] += 1
    for ch, target in FLAG_TARGETS.items():
        n = flag_use[ch]
        print(f"  {'ok ' if n >= target else 'OFF'} flag '{ch}': {n} (min {target})")
        if n < target:
            problems.append(f"flag '{ch}' used {n}x, min {target}")

    # --- lookaround coverage ---------------------------------------------
    look = 0
    for _, c in all_concepts():
        p = c["gold"]["pattern"]
        if "(?=" in p or "(?!" in p or "(?<=" in p or "(?<!" in p:
            look += 1
    for arc in ARCS:
        for turn in arc["turns"]:
            p = turn["gold"]["pattern"]
            if "(?=" in p or "(?!" in p or "(?<=" in p or "(?<!" in p:
                look += 1
    print(f"\nLOOKAROUND\n  {'ok ' if look >= MIN_LOOKAROUND else 'OFF'} "
          f"{look} patterns use lookahead/lookbehind (min {MIN_LOOKAROUND})")
    if look < MIN_LOOKAROUND:
        problems.append(f"only {look} lookaround patterns, min {MIN_LOOKAROUND}")

    print("\n" + "=" * 62)
    if problems:
        print(f"COVERAGE FAILURES ({len(problems)}):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("COVERAGE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
