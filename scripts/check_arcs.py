"""Verify every arc turn's gold answer against its full pool + held set."""

from __future__ import annotations

from arcs import ARCS
from verifier import verify


def as_spec(turn: dict) -> dict:
    if turn["family"] == "validate":
        cases = {
            "positives": list(turn["pool"]["positives"]) + list(turn["held"]["positives"]),
            "negatives": list(turn["pool"]["negatives"]) + list(turn["held"]["negatives"]),
        }
    else:
        cases = [tuple(x) for x in turn["pool"]] + [tuple(x) for x in turn["held"]]
    return {"family": turn["family"], "all": cases}


def main() -> int:
    bad = 0
    seen: set[str] = set()
    moves: dict[str, int] = {}

    for arc in ARCS:
        name = arc["concept"]
        if name in seen:
            print(f"DUPLICATE ARC NAME: {name}")
            bad += 1
        seen.add(name)
        moves[arc["arc"]] = moves.get(arc["arc"], 0) + 1

        failures = []
        for n, turn in enumerate(arc["turns"], start=1):
            r = verify(turn["gold"], as_spec(turn), "all")
            if not r.passed:
                failures.append((n, r))

        if failures:
            bad += 1
            print(f"FAIL  {name}  [{arc['arc']}]")
            for n, r in failures:
                print(f"        turn {n}: {r.error or ''}")
                for f in r.failures[:5]:
                    print(f"          {f}")
        else:
            print(f"ok    {name:<24} [{arc['arc']}]  {len(arc['turns'])} turns")

    print(f"\n{'=' * 64}")
    print(f"arcs: {len(ARCS)}   turns: {sum(len(a['turns']) for a in ARCS)}   failing arcs: {bad}")
    print(f"distinct move sequences: {len(moves)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
