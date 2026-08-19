"""Unattended pipeline: wait for training, sweep dev, select, test, report.

Runs after train.py finishes. Each evaluation is a separate subprocess so a
crash in one checkpoint cannot take down the whole sweep -- the run continues
and the failure is reported rather than silently dropping a data point.

Checkpoint selection is on dev HIDDEN-PASS RATE, never on validation loss. The
targets are short JSON blobs whose loss is dominated by boilerplate the model
gets right immediately, so loss can drift while task accuracy still improves.
With a free verifier there is no reason to select on a proxy.

Test is touched once, at the end, with the checkpoint dev already chose.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
CHECKPOINTS = ROOT / "models" / "checkpoints"
RESULTS = ROOT / "results"
PY = str(ROOT / ".venv" / "bin" / "python")

VARIANTS = ["instruct", "thinking"]
# Three, not six: in the first run loss reached 0.0003 by epoch 2, so later
# epochs are pure memorisation. Three still shows a peak-and-decline on dev.
EPOCHS = 3
TRAIN_TIMEOUT_S = 5 * 3600


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def training_running() -> bool:
    r = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def wait_for_training() -> bool:
    """Wait until training finishes, then proceed with whatever exists.

    Deliberately does NOT require the full epoch count. If a run dies or is
    stopped early, evaluating the checkpoints that do exist is far better than
    waiting for a file that will never appear and producing nothing. Whatever
    is missing shows up as a gap in the dev sweep table rather than as silence.
    """
    deadline = time.time() + TRAIN_TIMEOUT_S
    seen_running = False
    while time.time() < deadline:
        running = training_running()
        seen_running |= running
        done = all((CHECKPOINTS / v / f"epoch{EPOCHS}" / "adapter_model.safetensors").exists()
                   for v in VARIANTS)
        if done:
            log("both runs reached the final epoch")
            time.sleep(20)
            return True
        if seen_running and not running:
            time.sleep(30)                      # allow a final save to flush
            if not training_running():
                found = {v: sorted(p.name for p in (CHECKPOINTS / v).glob("epoch*"))
                         for v in VARIANTS if (CHECKPOINTS / v).exists()}
                log(f"training exited before the final epoch; proceeding with {found}")
                return any(found.values())
        time.sleep(45)
    log("TIMEOUT waiting for training")
    return any((CHECKPOINTS / v).exists() for v in VARIANTS)


def run_eval(variant: str, split: str, checkpoint: str, mode: str, tag: str) -> dict | None:
    out = f"{tag}.json"
    cmd = [PY, str(SCRIPTS / "evaluate.py"), variant, "--split", split,
           "--checkpoint", checkpoint, "--mode", mode, "--batch-size", "16", "--out", out]
    log(f"eval {tag}")
    t = time.time()
    proc = subprocess.run(cmd, cwd=str(SCRIPTS), capture_output=True, text=True)
    if proc.returncode != 0:
        log(f"  FAILED rc={proc.returncode}: {proc.stderr.strip().splitlines()[-3:]}")
        return None
    data = json.loads((RESULTS / out).read_text())["summary"]
    log(f"  hidden_pass={data['hidden_pass']}%  ({time.time() - t:.0f}s)")
    return data


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    if not wait_for_training():
        return 1

    report: dict = {"dev_sweep": {}, "selected": {}, "test": {}, "baseline": {}}

    # ---- 1. dev sweep over every epoch -------------------------------------
    for variant in VARIANTS:
        report["dev_sweep"][variant] = {}
        for ep in range(1, EPOCHS + 1):
            ck = CHECKPOINTS / variant / f"epoch{ep}"
            if not ck.exists():
                continue
            s = run_eval(variant, "dev", str(ck), "gold", f"{variant}_dev_epoch{ep}")
            if s:
                report["dev_sweep"][variant][ep] = s

        sweep = report["dev_sweep"][variant]
        if not sweep:
            log(f"no dev results for {variant}")
            continue
        best = max(sweep, key=lambda e: (sweep[e]["hidden_pass"], -e))
        report["selected"][variant] = {"epoch": best, "dev_hidden_pass": sweep[best]["hidden_pass"]}
        log(f"SELECTED {variant}: epoch {best} (dev hidden_pass {sweep[best]['hidden_pass']}%)")

    # ---- 2. test, once, with the dev-selected checkpoint -------------------
    for variant in VARIANTS:
        if variant not in report["selected"]:
            continue
        ep = report["selected"][variant]["epoch"]
        ck = str(CHECKPOINTS / variant / f"epoch{ep}")
        report["test"][variant] = {}
        for mode in ("gold", "free"):
            s = run_eval(variant, "test", ck, mode, f"{variant}_test_best_{mode}")
            if s:
                report["test"][variant][mode] = s

    # ---- 3. untrained baseline for the same test set ----------------------
    for variant in VARIANTS:
        s = run_eval(variant, "test", "base", "gold", f"{variant}_test_base_gold")
        if s:
            report["baseline"][variant] = s

    (RESULTS / "pipeline_report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report)
    log("pipeline complete -> results/FINAL_REPORT.md")
    return 0


def write_markdown(r: dict) -> None:
    L = ["# Text-to-Regex SFT: Instruct vs Thinking", "",
         "Scored by execution against hidden strings the model never saw.",
         "Checkpoint chosen on dev hidden-pass rate; test touched once.", ""]

    L += ["## Dev sweep (checkpoint selection)", "",
          "| variant | " + " | ".join(f"ep{e}" for e in range(1, EPOCHS + 1)) + " | selected |",
          "|---|" + "---|" * (EPOCHS + 1)]
    for v in VARIANTS:
        sweep = r["dev_sweep"].get(v, {})
        cells = [f"{sweep[e]['hidden_pass']}" if e in sweep else "-" for e in range(1, EPOCHS + 1)]
        sel = r["selected"].get(v, {}).get("epoch", "-")
        L.append(f"| {v} | " + " | ".join(cells) + f" | **epoch {sel}** |")

    L += ["", "## Test results (hidden-pass %, gold history)", "",
          "| metric | base instruct | tuned instruct | base thinking | tuned thinking |",
          "|---|---|---|---|---|"]

    def g(d, *keys):
        for k in keys:
            if d is None:
                return "-"
            d = d.get(k) if isinstance(d, dict) else None
        return "-" if d is None else d

    cols = [r["baseline"].get("instruct"), r["test"].get("instruct", {}).get("gold"),
            r["baseline"].get("thinking"), r["test"].get("thinking", {}).get("gold")]
    for label, path in [("hidden pass", ("hidden_pass",)), ("visible pass", ("visible_pass",)),
                        ("overfit (vis✓/hid✗)", ("overfit",)), ("json valid", ("json_valid",)),
                        ("easy", ("by_difficulty", "easy")), ("medium", ("by_difficulty", "medium")),
                        ("hard", ("by_difficulty", "hard")),
                        ("single-turn", ("by_kind", "single")),
                        ("multi-turn", ("by_kind", "trajectory")),
                        ("turn 1", ("by_turn", "1")), ("turn 2", ("by_turn", "2")),
                        ("turn 3", ("by_turn", "3"))]:
        L.append(f"| {label} | " + " | ".join(str(g(c, *path)) for c in cols) + " |")

    L += ["", "## Multi-turn: gold history vs free running", "",
          "| variant | gold | free |", "|---|---|---|"]
    for v in VARIANTS:
        t = r["test"].get(v, {})
        L.append(f"| {v} | {g(t.get('gold'), 'by_kind', 'trajectory')} | "
                 f"{g(t.get('free'), 'by_kind', 'trajectory')} |")

    L += ["", "## By conversational move (tuned, gold history)", "",
          "| move | instruct | thinking |", "|---|---|---|"]
    moves = set()
    for v in VARIANTS:
        moves |= set((r["test"].get(v, {}).get("gold") or {}).get("by_move", {}))
    for m in sorted(moves - {"SINGLE"}):
        L.append(f"| {m} | {g(r['test'].get('instruct', {}).get('gold'), 'by_move', m)} | "
                 f"{g(r['test'].get('thinking', {}).get('gold'), 'by_move', m)} |")

    (RESULTS / "FINAL_REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
