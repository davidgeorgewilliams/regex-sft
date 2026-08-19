"""Verify every concept's gold answer against its full pool + held set.

Run this before generating variants: a concept whose own gold fails is a
poisoned well, and every variant derived from it would inherit the error.
"""

from __future__ import annotations

import concepts
from verifier import verify


def as_spec(c: dict, family: str) -> dict:
    """Merge pool + held into a single split so the gold is checked on everything."""
    if family == "validate":
        cases = {
            "positives": list(c["pool"]["positives"]) + list(c["held"]["positives"]),
            "negatives": list(c["pool"]["negatives"]) + list(c["held"]["negatives"]),
        }
    else:
        cases = [tuple(x) for x in c["pool"]] + [tuple(x) for x in c["held"]]
    return {"family": family, "all": cases}


def main() -> int:
    families = [("validate", concepts.VALIDATE),
                ("extract", concepts.EXTRACT),
                ("substitute", concepts.SUBSTITUTE)]

    total, bad = 0, 0
    seen_names: set[str] = set()

    for family, items in families:
        print(f"\n{family.upper()}  ({len(items)} concepts)")
        for c in items:
            total += 1
            name = c["concept"]
            if name in seen_names:
                print(f"  DUPLICATE CONCEPT NAME: {name}")
                bad += 1
            seen_names.add(name)

            spec = as_spec(c, family)
            r = verify(c["gold"], spec, "all")
            if r.passed:
                n = (len(spec["all"]["positives"]) + len(spec["all"]["negatives"])
                     if family == "validate" else len(spec["all"]))
                print(f"  ok    {name:<28} ({n} strings, {len(c['traps'])} traps)")
            else:
                bad += 1
                print(f"  FAIL  {name}")
                if r.error:
                    print(f"          error: {r.error}")
                for f in r.failures[:6]:
                    print(f"          {f}")

    print(f"\n{'=' * 60}")
    print(f"concepts: {total}   failing: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
