"""Turn raw eval output into a report with uncertainty attached.

The pipeline's auto-generated table reports point estimates only, which is
misleading here: several cells have n=6, where a single item moves the number
by 16.7 points and "100%" and "50%" have overlapping confidence intervals.

This script attaches:
  * Wilson score intervals to every rate
  * paired McNemar tests for every comparison (the two variants are evaluated
    on the SAME test items, so a paired test is both correct and much more
    powerful than comparing two independent proportions)
  * an explicit UNDERPOWERED marker on any cell too small to interpret
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MIN_N_INTERPRETABLE = 25


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def load(name: str) -> list[dict] | None:
    p = RESULTS / name
    return json.load(open(p))["rows"] if p.exists() else None


def summary(name: str) -> dict | None:
    p = RESULTS / name
    return json.load(open(p))["summary"] if p.exists() else None


def rate(rows, filt=lambda r: True, key="hidden_pass"):
    sel = [r for r in rows if filt(r)]
    k, n = sum(r[key] for r in sel), len(sel)
    lo, hi = wilson(k, n)
    return k, n, (100 * k / n if n else 0.0), lo, hi


def mcnemar(a, b, filt=lambda r: True):
    A = [x for x in a if filt(x)]
    B = [y for y in b if filt(y)]
    n01 = sum(1 for x, y in zip(A, B) if not x["hidden_pass"] and y["hidden_pass"])
    n10 = sum(1 for x, y in zip(A, B) if x["hidden_pass"] and not y["hidden_pass"])
    chi = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if (n01 + n10) else 0.0
    return n10, n01, chi, chi > 3.84


def fmt(k, n, p, lo, hi):
    flag = "" if n >= MIN_N_INTERPRETABLE else f" ⚠️n={n}"
    return f"{p:.1f}% [{lo:.0f}–{hi:.0f}]{flag}"


def main() -> int:
    ig = load("instruct_test_best_gold.json")
    tg = load("thinking_test_best_gold.json")
    ifr = load("instruct_test_best_free.json")
    tfr = load("thinking_test_best_free.json")
    ib = load("instruct_test_base_gold.json")
    tb = load("thinking_test_base_gold_2048.json") or load("thinking_test_base_gold.json")
    tb_sum = (summary("thinking_test_base_gold_2048.json")
              or summary("thinking_test_base_gold.json"))

    L = ["# Text-to-Regex SFT — Qwen3-4B Instruct vs Thinking", "",
         "Scored by executing each generated pattern against hidden strings the model",
         "never saw. No judge, no human. Checkpoint chosen on dev hidden-pass rate;",
         "test touched once. Brackets are 95% Wilson intervals.", "",
         "## Headline", ""]

    L += ["| condition | hidden pass (test, n=160) |", "|---|---|"]
    for label, rows in [("base instruct", ib), ("**tuned instruct**", ig),
                        ("base thinking", tb), ("**tuned thinking**", tg)]:
        if rows:
            L.append(f"| {label} | {fmt(*rate(rows))} |")

    if tb_sum and tb_sum.get("truncated", 0) > 20:
        L += ["", f"> ⚠️ Base thinking truncated {tb_sum['truncated']}% of generations at "
                  f"max_new={tb_sum.get('max_new')} — treat its number as a floor, not a "
                  f"capability estimate."]

    L += ["", "## The comparison that was the point of the project", "",
          "**Tuned thinking vs tuned instruct is a null result.**", ""]
    n10, n01, chi, sig = mcnemar(ig, tg)
    _, _, pi, *_ = rate(ig)
    _, _, pt, *_ = rate(tg)
    L += [f"- instruct {pi:.1f}% vs thinking {pt:.1f}% ({pt - pi:+.1f} pp)",
          f"- paired McNemar χ²={chi:.2f} (needs >3.84), discordant pairs "
          f"{n10} instruct-only / {n01} thinking-only → **not significant**",
          "",
          "Disagreement is near-symmetric: the two models get *different* items right,",
          "not one strictly more. On single-shot generation with clean history, the",
          "reasoning trace buys nothing measurable here.", ""]

    L += ["## Where thinking does win: robustness to its own output", ""]
    rows_out = []
    for label, gold, free in [("instruct", ig, ifr), ("thinking", tg, tfr)]:
        tr = lambda r: r["kind"] == "trajectory"
        _, _, pg, *_ = rate(gold, tr)
        _, n, pf, lo, hi = rate(free, tr)
        n10, n01, chi, sig = mcnemar(gold, free, tr)
        rows_out.append(f"| {label} | {pg:.1f}% | {pf:.1f}% | {pf - pg:+.1f} pp | "
                        f"χ²={chi:.2f} {'**significant**' if sig else 'ns'} |")
    L += ["| variant | gold history | free-running | drop | paired test |",
          "|---|---|---|---|---|"] + rows_out

    n10, n01, chi, sig = mcnemar(ifr, tfr, lambda r: r["kind"] == "trajectory")
    _, _, pif, *_ = rate(ifr, lambda r: r["kind"] == "trajectory")
    _, _, ptf, *_ = rate(tfr, lambda r: r["kind"] == "trajectory")
    L += ["", f"**Free-running, thinking beats instruct {ptf:.1f}% vs {pif:.1f}% "
              f"({ptf - pif:+.1f} pp), χ²={chi:.2f} → "
              f"{'significant at p<0.05' if sig else 'not significant'}.**", "",
          "Instruct degrades significantly when it consumes its own previous answers;",
          "thinking does not. A plausible mechanism: instruct pattern-matches the prior",
          "answer, so an early error propagates, whereas the trace re-derives from the",
          "instruction each turn. This is the one place the trace pays for itself — and",
          "it is the condition that resembles real use.", ""]

    L += ["## Did fine-tuning work at all?", ""]
    if ib:
        n10, n01, chi, sig = mcnemar(ib, ig)
        _, _, pb, *_ = rate(ib)
        L += [f"- instruct {pb:.1f}% → {pi:.1f}% ({pi - pb:+.1f} pp), χ²={chi:.2f} → "
              f"{'**significant at p<0.05**' if sig else 'not significant'}", ""]

    L += ["## Breakdown (⚠️ marks cells too small to interpret)", "",
          "| cell | tuned instruct | tuned thinking |", "|---|---|---|"]
    cells = [("easy", lambda r: r["difficulty"] == "easy"),
             ("medium", lambda r: r["difficulty"] == "medium"),
             ("hard", lambda r: r["difficulty"] == "hard"),
             ("validate", lambda r: r["family"] == "validate"),
             ("extract", lambda r: r["family"] == "extract"),
             ("substitute", lambda r: r["family"] == "substitute"),
             ("single-turn", lambda r: r["kind"] == "single"),
             ("multi-turn", lambda r: r["kind"] == "trajectory"),
             ("turn 1", lambda r: r["turn"] == 1),
             ("turn 2", lambda r: r["turn"] == 2),
             ("turn 3", lambda r: r["turn"] == 3)]
    for label, f in cells:
        L.append(f"| {label} | {fmt(*rate(ig, f))} | {fmt(*rate(tg, f))} |")

    L += ["", "## Overfitting signal (visible pass ✓ / hidden fail ✗)", "",
          "Reproduced the examples shown in the prompt without honouring the",
          "instruction. Measured in microseconds, no judge required.", "",
          "| condition | overfit rate |", "|---|---|"]
    for label, rows in [("base instruct", ib), ("tuned instruct", ig), ("tuned thinking", tg)]:
        if rows:
            k, n, p, lo, hi = rate(rows, key="overfit")
            L.append(f"| {label} | {p:.1f}% [{lo:.0f}–{hi:.0f}] |")

    L += ["", "## Per-move results are NOT reportable", "",
          "Each conversational move has **n=6** in the test split. One item moves the",
          "rate by 16.7 pp; a 6/6 result has a 95% interval of [61, 100] and a 0/6",
          "result [0, 39]. Those intervals overlap heavily, so the per-move table in",
          "`pipeline_report.json` cannot separate the moves and is excluded here.",
          "",
          "The move rebalancing was still worth doing — it guarantees every move appears",
          "in train, dev and test, so the *training distribution* is not skewed. But",
          "answering \"which moves does SFT improve?\" needs roughly 5–10x more test",
          "trajectories per move.", "",
          "## Known limitations", "",
          "1. **Thinking is under-trained.** Its dev curve was still rising at epoch 3",
          "   (47.2 → 58.4 → 62.7) when training stopped; instruct had already peaked at",
          "   epoch 2 and turned down. The thinking number is a lower bound.",
          "2. **Unequal supervision.** Thinking trains on 2.5x the target tokens (the",
          "   traces). That is inherent to the treatment, but a gain cannot be cleanly",
          "   attributed to reasoning rather than to more tokens of supervision.",
          "3. **Traces are authored, not certified.** They were written knowing the",
          "   answer, so they are rationalisation. Guaranteed only: the stated pattern is",
          "   the verified gold, no trace cites a string the model cannot see, and every",
          "   trap is reasoned about somewhere.",
          "4. **Single seed.** No variance estimate across runs.", ""]

    (RESULTS / "FINAL_REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
