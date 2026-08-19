"""Build data/{train,dev,test}/ -- one pretty-printed conversation per file.

Pipeline
    1. VARIANTS   resample which pool strings are VISIBLE (shown in the prompt);
                  hidden = pool remainder + the entire held set.
    2. SPLIT      group by concept, stratify by family x difficulty, and hold
                  three concepts out entirely for the out-of-distribution slice.
    3. RENDER     emit a chat conversation plus the metadata the verifier and
                  eval need.
    4. RE-VERIFY  read every written file back off disk and re-check its gold
                  against its own visible and hidden splits.

Two invariants the build refuses to violate:
  * Held strings never appear in VISIBLE. They are what makes the
    visible-pass/hidden-fail overfitting signal meaningful -- a held string
    that leaks into the prompt stops testing whether the model read the
    instruction.
  * A concept lives in exactly one split. Variants of one concept are near
    duplicates, so splitting at the spec level would leak train into test.
    Trajectories are additionally atomic: all turns share one file.
"""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import concepts as concept_lib
import traces
from arcs import ARCS
from verifier import verify

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SEED = 20260819
TARGETS = {"train": 500, "dev": 100, "test": 100}

# Concepts withheld from training entirely, so the test set can separate
# "got better at this task" from "learned regex reasoning". These are chosen
# to span all three families and to be individually non-trivial.
OOD_CONCEPTS = {"roman_numeral", "srt_timecode", "mask_ssn", "repeated_word_chain"}

SYSTEM_PROMPT = (
    "You write Python regular expressions.\n"
    "Respond with a single JSON object and nothing else.\n"
    '  validation tasks:   {"pattern": "...", "flags": "..."}\n'
    '  extraction tasks:   {"pattern": "...", "flags": "..."}   (capture group 1 holds the answer)\n'
    '  substitution tasks: {"pattern": "...", "replacement": "...", "flags": "..."}\n'
    "Patterns are evaluated with re.fullmatch for validation, re.search for extraction, "
    "and re.sub for substitution.\n"
    'The flags field may only contain the characters "i", "m" and "s"; use "" for none.'
)

FAMILY_BLURB = {
    "validate": "Write a pattern that matches exactly the strings that should match.",
    "extract": "Write a pattern whose first capture group holds the requested text.",
    "substitute": "Write a pattern and replacement that produce the requested output.",
}


# --------------------------------------------------------------------------
# 1. Variants
# --------------------------------------------------------------------------

def split_validate(pool: dict, held: dict, rng: random.Random) -> tuple[dict, dict]:
    pos, neg = list(pool["positives"]), list(pool["negatives"])
    rng.shuffle(pos)
    rng.shuffle(neg)
    # Show at least one of each so the task is well-posed, but never all of
    # them -- the hidden split needs pool residue as well as held strings.
    # The count varies 1-3 rather than being fixed: it multiplies the number
    # of distinct prompts a small pool can produce, and a shorter example list
    # increases the pressure to read the instruction rather than pattern-match
    # the examples, which is exactly what the held set tests.
    n_pos = rng.randint(1, min(3, len(pos) - 1)) if len(pos) > 1 else len(pos)
    n_neg = rng.randint(1, min(3, len(neg) - 1)) if len(neg) > 1 else len(neg)
    visible = {"positives": pos[:n_pos], "negatives": neg[:n_neg]}
    hidden = {
        "positives": pos[n_pos:] + list(held["positives"]),
        "negatives": neg[n_neg:] + list(held["negatives"]),
    }
    return visible, hidden


def split_cases(pool: list, held: list, rng: random.Random) -> tuple[list, list]:
    cases = [list(c) for c in pool]
    rng.shuffle(cases)
    n = rng.randint(1, min(3, len(cases) - 1)) if len(cases) > 1 else len(cases)
    return cases[:n], cases[n:] + [list(c) for c in held]


def make_variant(base: dict, family: str, rng: random.Random) -> dict:
    if family == "validate":
        visible, hidden = split_validate(base["pool"], base["held"], rng)
    else:
        visible, hidden = split_cases(base["pool"], base["held"], rng)
    return {"visible": visible, "hidden": hidden}


# --------------------------------------------------------------------------
# 2. Rendering
# --------------------------------------------------------------------------

def render_user(family: str, instruction: str, visible) -> str:
    lines = [f"Task: {instruction}", "", FAMILY_BLURB[family], ""]
    if family == "validate":
        lines.append("Must match:")
        lines += [f"  {s!r}" for s in visible["positives"]] or ["  (none)"]
        lines.append("Must not match:")
        lines += [f"  {s!r}" for s in visible["negatives"]] or ["  (none)"]
    elif family == "extract":
        lines.append("Examples (input -> captured text, null means no match):")
        for text, exp in visible:
            lines.append(f"  {text!r} -> {'null' if exp is None else repr(exp)}")
    else:
        lines.append("Examples (input -> output):")
        for text, exp in visible:
            lines.append(f"  {text!r} -> {exp!r}")
    return "\n".join(lines)


def target_json(family: str, gold: dict) -> dict:
    out = {"pattern": gold["pattern"]}
    if family == "substitute":
        out["replacement"] = gold["replacement"]
    out["flags"] = gold.get("flags", "")
    return out


def tier(traps: list) -> str:
    return {1: "easy", 2: "medium"}.get(len(traps), "hard")


def build_single(base: dict, family: str, idx: int, rng: random.Random) -> dict:
    v = make_variant(base, family, rng)
    gold = base["gold"]
    turn = {
        "index": 1,
        "family": family,
        "instruction": base["instruction"],
        "traps": base["traps"],
        "visible": v["visible"],
        "hidden": v["hidden"],
        "gold": gold,
        "target": target_json(family, gold),
        "user": render_user(family, base["instruction"], v["visible"]),
    }
    return {
        "id": f"rx_{family}_{base['concept']}_v{idx:02d}",
        "kind": "single",
        "concept": base["concept"],
        "family": family,
        "difficulty": tier(base["traps"]),
        "traps": base["traps"],
        "variant": idx,
        "system": SYSTEM_PROMPT,
        "turns": [turn],
    }


def build_trajectory(arc: dict, idx: int, rng: random.Random) -> dict:
    turns = []
    for n, t in enumerate(arc["turns"], start=1):
        v = make_variant(t, t["family"], rng)
        turns.append({
            "index": n,
            "family": t["family"],
            "move": (["ESTABLISH"] + arc["arc"].split("->"))[n - 1],
            "instruction": t["instruction"],
            "traps": t["traps"],
            "visible": v["visible"],
            "hidden": v["hidden"],
            "gold": t["gold"],
            "target": target_json(t["family"], t["gold"]),
            "user": render_user(t["family"], t["instruction"], v["visible"]),
        })
    return {
        "id": f"rx_traj_{arc['concept']}_v{idx:02d}",
        "kind": "trajectory",
        "concept": arc["concept"],
        "arc": arc["arc"],
        "family": "mixed",
        "difficulty": tier(max(arc["turns"], key=lambda t: len(t["traps"]))["traps"]),
        "traps": sorted({tr for t in arc["turns"] for tr in t["traps"]}),
        "variant": idx,
        "system": SYSTEM_PROMPT,
        "turns": turns,
    }


def attach_traces(record: dict) -> None:
    """Attach authored reasoning traces. TRAIN ONLY.

    A trace is the training target -- it is what the model learns to emit. In
    dev or test it would either leak the answer or measure nothing, since at
    evaluation the model generates its own trace.

    Note the substitution uses str.replace, not str.format: patterns routinely
    contain braces such as \\d{4}, which format() would read as field markers.
    """
    for turn in record["turns"]:
        options = (traces.CONCEPT_TRACES[record["concept"]] if record["kind"] == "single"
                   else traces.ARC_TRACES[(record["concept"], turn["index"])])
        # Rotate phrasing by variant so consecutive variants of a concept do not
        # repeat target text, and offset by turn index so a trajectory does not
        # use the same phrasing slot on all three turns.
        tpl = options[(record["variant"] + turn["index"]) % len(options)]
        gold = turn["gold"]
        turn["thinking"] = (tpl.replace("{p}", gold["pattern"])
                                .replace("{r}", gold.get("replacement", "")))
        turn["trace_source"] = "authored"
        turn["trace_phrasing"] = (record["variant"] + turn["index"]) % len(options)


def to_messages_with_traces(record: dict) -> list[dict]:
    """Full conversation view, traces included.

    This is the authoring view. The training renderer expands a trajectory into
    one example per assistant turn and strips <think> from the history of each,
    because Qwen3-Thinking's chat template keeps only final answers in history.
    """
    msgs = [{"role": "system", "content": record["system"]}]
    for t in record["turns"]:
        msgs.append({"role": "user", "content": t["user"]})
        content = json.dumps(t["target"])
        if t.get("thinking"):
            content = f"<think>\n{t['thinking']}\n</think>\n{content}"
        msgs.append({"role": "assistant", "content": content})
    return msgs


def prompt_signature(record: dict) -> str:
    """Identity of a spec as the model sees it: the rendered user turns."""
    return json.dumps([t["user"] for t in record["turns"]], sort_keys=True)


def to_messages(record: dict) -> list[dict]:
    """Chat-format view. Assistant turns carry the final JSON only; <think>
    blocks are added later by the rejection-sampling stage, and are stripped
    from history when the Thinking variant is rendered for training."""
    msgs = [{"role": "system", "content": record["system"]}]
    for t in record["turns"]:
        msgs.append({"role": "user", "content": t["user"]})
        msgs.append({"role": "assistant", "content": json.dumps(t["target"])})
    return msgs


# --------------------------------------------------------------------------
# 3. Split
# --------------------------------------------------------------------------

ALL_MOVES = {"NARROW", "BROADEN", "REPAIR", "RETARGET", "FLAG",
             "PIVOT", "ROLLBACK", "COMPOSE", "INVERT", "GENERALIZE"}


def pick_covering_arcs(pool: list[dict], forced: list[dict],
                       rng: random.Random, tries: int = 200_000) -> list[dict] | None:
    """Choose arcs whose move slots cover all 10 moves exactly once.

    Uniform move coverage in the library does not survive a concept-level
    split on its own: splitting moves whole arcs, and each arc carries only
    two moves, so a random 5-arc dev set typically covers 6-7 of the 10.
    Selecting an exact cover instead means every move is reportable in every
    split. Five arcs give exactly 10 slots, so the cover is a perfect matching.
    """
    forced_moves = [m for a in forced for m in a["arc"].split("->")]
    if len(set(forced_moves)) != len(forced_moves):
        return None
    remaining_moves = ALL_MOVES - set(forced_moves)
    need = len(remaining_moves) // 2
    forced_names = {a["concept"] for a in forced}
    rest = [a for a in pool if a["concept"] not in forced_names
            and set(a["arc"].split("->")) <= remaining_moves]

    for _ in range(tries):
        if len(rest) < need:
            return None
        sample = rng.sample(rest, need)
        moves = forced_moves + [m for a in sample for m in a["arc"].split("->")]
        if len(set(moves)) == len(ALL_MOVES):
            return forced + sample
    return None


def assign_splits(units: list[dict], arcs: list[dict], rng: random.Random) -> dict[str, str]:
    """Assign whole CONCEPTS to splits. Returns concept -> split.

    Single-turn concepts are stratified by family x difficulty so no split is
    accidentally easier or family-skewed. Arcs are assigned separately, by
    move coverage, because for them the binding constraint is that every move
    appears in every split.
    """
    by_concept: dict[str, dict] = {}
    for u in units:
        by_concept.setdefault(u["concept"], u)

    assignment = {c: "test" for c in OOD_CONCEPTS if c in by_concept}

    # --- arcs: exact move cover for test, then for dev, remainder to train --
    ood_arcs = [a for a in arcs if a["concept"] in OOD_CONCEPTS]
    test_arcs = pick_covering_arcs(arcs, ood_arcs, rng)
    if test_arcs is None:
        raise RuntimeError("no move-covering arc set found for test")
    chosen = {a["concept"] for a in test_arcs}
    dev_arcs = pick_covering_arcs([a for a in arcs if a["concept"] not in chosen], [], rng)
    if dev_arcs is None:
        raise RuntimeError("no move-covering arc set found for dev")
    for a in test_arcs:
        assignment[a["concept"]] = "test"
    for a in dev_arcs:
        assignment[a["concept"]] = "dev"
    for a in arcs:
        assignment.setdefault(a["concept"], "train")

    # --- single-turn concepts: stratified by family x difficulty ------------
    strata: dict[tuple, list[str]] = defaultdict(list)
    arc_names = {a["concept"] for a in arcs}
    for name, u in by_concept.items():
        if name in OOD_CONCEPTS or name in arc_names:
            continue
        strata[(u["family"], u["difficulty"])].append(name)

    for _, names in sorted(strata.items()):
        names = sorted(names)
        rng.shuffle(names)
        n = len(names)
        n_dev = max(1, round(n * 0.15)) if n >= 3 else 0
        n_test = max(1, round(n * 0.20)) if n >= 3 else 0
        for i, name in enumerate(names):
            assignment[name] = "dev" if i < n_dev else "test" if i < n_dev + n_test else "train"
    return assignment


# --------------------------------------------------------------------------
# 4. Build
# --------------------------------------------------------------------------

def main() -> int:
    rng = random.Random(SEED)

    singles = ([("validate", c) for c in concept_lib.VALIDATE]
               + [("extract", c) for c in concept_lib.EXTRACT]
               + [("substitute", c) for c in concept_lib.SUBSTITUTE])

    # One probe record per concept/arc, purely to drive stratified assignment.
    probes = ([build_single(c, f, 0, random.Random(0)) for f, c in singles]
              + [build_trajectory(a, 0, random.Random(0)) for a in ARCS])
    assignment = assign_splits(probes, ARCS, rng)

    # Roughly 70/30 single vs trajectory by spec count -> near-balanced by
    # training example, since each trajectory expands into 3 examples.
    plan = {s: {"single": round(n * 0.70), "traj": round(n * 0.30)} for s, n in TARGETS.items()}

    singles_by_split = defaultdict(list)
    for f, c in singles:
        singles_by_split[assignment[c["concept"]]].append((f, c))
    arcs_by_split = defaultdict(list)
    for a in ARCS:
        arcs_by_split[assignment[a["concept"]]].append(a)

    if DATA.exists():
        for sub in ("train", "dev", "test"):
            shutil.rmtree(DATA / sub, ignore_errors=True)

    written: dict[str, list[dict]] = {}
    shortfalls: list[str] = []
    for split, want in plan.items():
        out_dir = DATA / split
        out_dir.mkdir(parents=True, exist_ok=True)
        records = []
        # Variants differ only in which pool strings are shown, so with small
        # pools two variants can render byte-identical prompts. Duplicates
        # over-weight a concept in training and silently shrink the effective
        # test n, so uniqueness is enforced by rejection rather than assumed.
        seen: set[str] = set()

        # Single-turn budget is allocated per family in proportion to that
        # family's concept count, then filled within the family. Without this,
        # dedup exhausts the smaller extract/substitute pools first and the
        # round-robin backfills with validate -- which skewed train to 54%
        # validate against a 40% test set, a train/test distribution mismatch.
        by_family: dict[str, list] = defaultdict(list)
        for fam, base in singles_by_split[split]:
            by_family[fam].append((fam, base))
        n_concepts = sum(len(v) for v in by_family.values()) or 1
        quota = {f: round(want["single"] * len(v) / n_concepts) for f, v in by_family.items()}

        jobs = [(f"single/{f}", by_family[f], quota[f],
                 lambda item, i: build_single(item[1], item[0], i, rng))
                for f in sorted(by_family)]
        jobs.append(("traj", arcs_by_split[split], want["traj"],
                     lambda item, i: build_trajectory(item, i, rng)))

        for kind, pool, want_n, builder in jobs:
            made, next_idx, exhausted = 0, Counter(), set()
            while made < want_n and len(exhausted) < len(pool):
                for j, item in enumerate(pool):
                    if made >= want_n or j in exhausted:
                        continue
                    rec = None
                    for _ in range(60):
                        cand = builder(item, next_idx[j])
                        sig = prompt_signature(cand)
                        if sig not in seen:
                            seen.add(sig)
                            rec = cand
                            break
                    if rec is None:
                        exhausted.add(j)
                    else:
                        next_idx[j] += 1
                        records.append(rec)
                        made += 1
            if made < want_n:
                shortfalls.append(
                    f"{split}/{kind}: {made} of {want_n} (pool of {len(pool)} "
                    f"concepts exhausted distinct prompts)")

        for r in records:
            r["split"] = split
            if split == "train":
                attach_traces(r)
            r["messages"] = to_messages_with_traces(r)
            (out_dir / f"{r['id']}.json").write_text(json.dumps(r, indent=2, ensure_ascii=False) + "\n")
        written[split] = records

    (DATA / "splits.json").write_text(json.dumps({
        "seed": SEED,
        "ood_concepts": sorted(OOD_CONCEPTS),
        "concept_to_split": dict(sorted(assignment.items())),
    }, indent=2) + "\n")

    if shortfalls:
        print("SHORTFALLS (reported, not padded with duplicates):")
        for s in shortfalls:
            print(f"  - {s}")
        print()
    return report_and_audit(written, assignment)


# --------------------------------------------------------------------------
# 5. Audit -- read back from disk and re-check every invariant
# --------------------------------------------------------------------------

def report_and_audit(written: dict, assignment: dict) -> int:
    problems: list[str] = []

    print("SPLIT SUMMARY")
    for split in ("train", "dev", "test"):
        recs = written[split]
        files = list((DATA / split).glob("*.json"))
        kinds = Counter(r["kind"] for r in recs)
        turns = sum(len(r["turns"]) for r in recs)
        concs = len({r["concept"] for r in recs})
        print(f"  {split:<6} files={len(files):<4} specs={len(recs):<4} "
              f"single={kinds['single']:<4} traj={kinds['trajectory']:<4} "
              f"turns={turns:<4} concepts={concs}")
        if len(files) != len(recs):
            problems.append(f"{split}: {len(files)} files vs {len(recs)} records")

    print("\n  difficulty x split")
    for split in ("train", "dev", "test"):
        c = Counter(r["difficulty"] for r in written[split])
        tot = sum(c.values())
        print(f"    {split:<6} " + "  ".join(
            f"{k}={c[k]:>3} ({100 * c[k] / tot:4.1f}%)" for k in ("easy", "medium", "hard")))

    print("\n  family x split (single-turn only)")
    for split in ("train", "dev", "test"):
        c = Counter(r["family"] for r in written[split] if r["kind"] == "single")
        tot = sum(c.values()) or 1
        print(f"    {split:<6} " + "  ".join(
            f"{k}={c[k]:>3} ({100 * c[k] / tot:4.1f}%)" for k in ("validate", "extract", "substitute")))

    # -- concept leakage across splits
    seen: dict[str, str] = {}
    for split in ("train", "dev", "test"):
        for r in written[split]:
            if seen.setdefault(r["concept"], split) != split:
                problems.append(f"concept {r['concept']} in both {seen[r['concept']]} and {split}")

    # -- OOD concepts must be test-only and absent from train
    train_concepts = {r["concept"] for r in written["train"]}
    for c in OOD_CONCEPTS:
        if c in train_concepts:
            problems.append(f"OOD concept {c} leaked into train")
    print(f"\n  OOD concepts held out of train: {sorted(OOD_CONCEPTS)}")

    # -- every move must be reportable in every split
    print("\n  move coverage per split (arc turns)")
    for split in ("train", "dev", "test"):
        mv = Counter(t["move"] for r in written[split] if r["kind"] == "trajectory" for t in r["turns"])
        mv.pop("ESTABLISH", None)
        missing = ALL_MOVES - set(mv)
        mark = "ok " if not missing else "OFF"
        print(f"    {mark} {split:<6} {len(mv)}/10 moves"
              + (f"  MISSING: {sorted(missing)}" if missing else ""))
        print(f"           {dict(sorted(mv.items()))}")
        if missing:
            problems.append(f"{split} is missing moves: {sorted(missing)}")

    # -- prompt uniqueness within each split
    print("\n  prompt uniqueness")
    for split in ("train", "dev", "test"):
        sigs = [prompt_signature(r) for r in written[split]]
        n_dup = len(sigs) - len(set(sigs))
        print(f"    {'ok ' if not n_dup else 'OFF'} {split:<6} "
              f"{len(set(sigs))}/{len(sigs)} distinct prompts")
        if n_dup:
            problems.append(f"{split} has {n_dup} duplicate prompts")

    # -- re-verify every file off disk
    print("\nAUDIT (re-reading every file from disk)")
    n_files = n_turns = 0
    for split in ("train", "dev", "test"):
        for path in sorted((DATA / split).glob("*.json")):
            rec = json.loads(path.read_text())
            n_files += 1
            for t in rec["turns"]:
                n_turns += 1
                for which in ("visible", "hidden"):
                    r = verify(t["gold"], {"family": t["family"], which: t[which]}, which)
                    if not r.passed:
                        problems.append(f"{path.name} turn {t['index']} [{which}]: {r.error or r.failures[:2]}")
                # held strings must never have leaked into the prompt
                vis = t["visible"]
                shown = (list(vis["positives"]) + list(vis["negatives"])
                         if t["family"] == "validate" else [c[0] for c in vis])
                for s in shown:
                    if s not in t["user"] and repr(s) not in t["user"]:
                        problems.append(f"{path.name} turn {t['index']}: visible string missing from prompt")
                        break
            if rec["messages"][0]["role"] != "system" or len(rec["messages"]) != 1 + 2 * len(rec["turns"]):
                problems.append(f"{path.name}: malformed messages array")

    print(f"  re-verified {n_files} files / {n_turns} turns")

    print("\n" + "=" * 66)
    if problems:
        print(f"BUILD FAILED ({len(problems)} problems):")
        for p in problems[:25]:
            print(f"  - {p}")
        return 1
    print("BUILD OK -- every gold re-verified from disk, no concept leakage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
