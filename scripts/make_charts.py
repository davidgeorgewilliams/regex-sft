"""Generate learning-curve charts as one self-contained HTML file.

No external libraries, no CDN: inline SVG plus a little CSS/JS, so the file
opens anywhere and survives being emailed. Theme-aware, with a table view of
every series underneath for accessibility and for copy-paste into notes.

Palette is categorical slots 1-2 from the reference palette, validated with
scripts/validate_palette.js in both light and dark modes (all checks pass:
worst-pair CVD dE 24.7 light / 26.8 dark against an >=8 target).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CKPT = ROOT / "models" / "checkpoints"

W, H = 640, 300
PAD = dict(l=64, r=64, t=16, b=44)   # r leaves room for end-of-line direct labels
PW, PH = W - PAD["l"] - PAD["r"], H - PAD["t"] - PAD["b"]

VARIANTS = ["instruct", "thinking"]
SERIES_CLASS = {"instruct": "s1", "thinking": "s2"}


# ---------------------------------------------------------------- primitives

def x_of(i: int, n: int) -> float:
    return PAD["l"] + (PW * i / (n - 1) if n > 1 else PW / 2)


def y_lin(v: float, lo: float, hi: float) -> float:
    return PAD["t"] + PH * (1 - (v - lo) / (hi - lo))


def y_log(v: float, lo: float, hi: float) -> float:
    v = max(v, lo)
    f = (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    return PAD["t"] + PH * (1 - f)


def axes(xticks, yticks, xlabel, ylabel) -> str:
    """Recessive grid and axes; gridlines behind the marks."""
    out = []
    for yv, ypix, lab in yticks:
        out.append(f'<line class="grid" x1="{PAD["l"]}" y1="{ypix:.1f}" '
                   f'x2="{PAD["l"] + PW}" y2="{ypix:.1f}"/>')
        out.append(f'<text class="tick" x="{PAD["l"] - 10}" y="{ypix + 4:.1f}" '
                   f'text-anchor="end">{lab}</text>')
    for xpix, lab in xticks:
        out.append(f'<text class="tick" x="{xpix:.1f}" y="{PAD["t"] + PH + 20}" '
                   f'text-anchor="middle">{lab}</text>')
    out.append(f'<line class="axis" x1="{PAD["l"]}" y1="{PAD["t"] + PH}" '
               f'x2="{PAD["l"] + PW}" y2="{PAD["t"] + PH}"/>')
    out.append(f'<text class="axlabel" x="{PAD["l"] + PW / 2}" y="{H - 6}" '
               f'text-anchor="middle">{xlabel}</text>')
    out.append(f'<text class="axlabel" transform="translate(14,{PAD["t"] + PH / 2}) '
               f'rotate(-90)" text-anchor="middle">{ylabel}</text>')
    return "\n".join(out)


def line_chart(cid, title, subtitle, series, xticks, yticks, xlabel, ylabel,
               label_last=True, markers=True) -> str:
    """series: {name: [(xpix, ypix, raw_x, raw_y_label), ...]}"""
    body = [axes(xticks, yticks, xlabel, ylabel)]
    for name, pts in series.items():
        cls = SERIES_CLASS[name]
        d = " ".join(("M" if i == 0 else "L") + f"{p[0]:.1f},{p[1]:.1f}"
                     for i, p in enumerate(pts))
        body.append(f'<path class="ln {cls}" d="{d}"/>')
        if markers:
            for px, py, rx, ry in pts:
                # 2px surface ring so overlapping markers stay separable
                body.append(f'<circle class="mk {cls}" cx="{px:.1f}" cy="{py:.1f}" r="5">'
                            f'<title>{name} — {rx}: {ry}</title></circle>')
        if label_last:
            px, py, _, ry = pts[-1]
            body.append(f'<text class="dlabel {cls}" x="{px + 10:.1f}" y="{py + 4:.1f}">'
                        f'{ry}</text>')
    return f'''<figure class="chart">
  <figcaption><h3>{title}</h3><p>{subtitle}</p></figcaption>
  <div class="legend">{"".join(f'<span class="lg"><i class="{SERIES_CLASS[n]}"></i>{n}</span>' for n in series)}</div>
  <svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}">{"".join(body)}</svg>
</figure>'''


def grouped_bars(cid, title, subtitle, cats, series, ylabel, ymax) -> str:
    """cats: [label,...]; series: {name: [v,...]}"""
    body = []
    # Round tick values, not ymax/4 -- awkward ticks (0,22,45,68,90) make a
    # reader do arithmetic to place a bar.
    step = 20
    yticks = [(v, y_lin(v, 0, ymax), f"{v}%") for v in range(0, ymax + 1, step)]
    body.append(axes([], yticks, "", ylabel))
    gw = PW / len(cats)
    bw = min(46, (gw - 26) / len(series))
    for ci, cat in enumerate(cats):
        cx = PAD["l"] + gw * (ci + 0.5)
        for si, (name, vals) in enumerate(series.items()):
            v = vals[ci]
            x = cx - (len(series) * bw + 2) / 2 + si * (bw + 2)   # 2px surface gap
            yy = y_lin(v, 0, ymax)
            h = PAD["t"] + PH - yy
            body.append(f'<rect class="br {SERIES_CLASS[name]}" x="{x:.1f}" y="{yy:.1f}" '
                        f'width="{bw:.1f}" height="{h:.1f}" rx="4">'
                        f'<title>{name} — {cat}: {v}%</title></rect>')
            body.append(f'<text class="blabel" x="{x + bw / 2:.1f}" y="{yy - 7:.1f}" '
                        f'text-anchor="middle">{v}</text>')
        body.append(f'<text class="tick" x="{cx:.1f}" y="{PAD["t"] + PH + 20}" '
                    f'text-anchor="middle">{cat}</text>')
    return f'''<figure class="chart">
  <figcaption><h3>{title}</h3><p>{subtitle}</p></figcaption>
  <div class="legend">{"".join(f'<span class="lg"><i class="{SERIES_CLASS[n]}"></i>{n}</span>' for n in series)}</div>
  <svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}">{"".join(body)}</svg>
</figure>'''


def table(caption, headers, rows) -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (f'<details class="tbl"><summary>{caption} — data table</summary>'
            f'<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></details>')


# ---------------------------------------------------------------- build

def main() -> int:
    charts, tables = [], []

    # --- 1. training loss -------------------------------------------------
    logs = {v: json.loads((CKPT / v / "trainlog.json").read_text())["log"] for v in VARIANTS}
    lo, hi = 1e-4, 2.0
    yt = [(v, y_log(v, lo, hi), f"{v:g}") for v in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)]
    maxstep = max(p["step"] for v in VARIANTS for p in logs[v])
    xt = [(PAD["l"] + PW * s / maxstep, str(s)) for s in (0, 50, 100, 150) if s <= maxstep]
    ser = {}
    for v in VARIANTS:
        ser[v] = [(PAD["l"] + PW * p["step"] / maxstep, y_log(p["loss"], lo, hi),
                   f"step {p['step']}", f"{p['loss']:.4f}") for p in logs[v]]
    charts.append(line_chart("loss", "Training loss",
        "Log scale. Instruct drives loss to ~0 by epoch 2 — its target is a short JSON blob it can "
        "memorise. Thinking plateaus near 0.27: a reasoning trace still carries real uncertainty.",
        ser, xt, yt, "optimizer step", "loss (log)", label_last=False, markers=False))
    tables.append(table("Training loss", ["step", "instruct", "thinking"],
        [[logs["instruct"][i]["step"], f'{logs["instruct"][i]["loss"]:.4f}',
          f'{logs["thinking"][i]["loss"]:.4f}'] for i in range(0, len(logs["instruct"]), 3)]))

    # --- 2. dev sweep -----------------------------------------------------
    rep = json.loads((RESULTS / "pipeline_report.json").read_text())
    sweep = rep["dev_sweep"]
    epochs = sorted(int(e) for e in sweep["instruct"])
    yt2 = [(v, y_lin(v, 40, 70), f"{v}%") for v in (40, 50, 60, 70)]
    xt2 = [(x_of(i, len(epochs)), f"epoch {e}") for i, e in enumerate(epochs)]
    ser2 = {v: [(x_of(i, len(epochs)), y_lin(sweep[v][str(e)]["hidden_pass"], 40, 70),
                 f"epoch {e}", f'{sweep[v][str(e)]["hidden_pass"]}%')
                for i, e in enumerate(epochs)] for v in VARIANTS}
    charts.append(line_chart("dev", "Dev accuracy by epoch — checkpoint selection",
        "Instruct peaks at epoch 2 then declines; Thinking is still climbing at 3, so its result is a "
        "lower bound. Selecting on validation loss would have picked Instruct epoch 3 — the worse checkpoint.",
        ser2, xt2, yt2, "training epoch", "dev hidden-pass"))
    tables.append(table("Dev sweep", ["epoch", "instruct", "thinking"],
        [[e, f'{sweep["instruct"][str(e)]["hidden_pass"]}%',
          f'{sweep["thinking"][str(e)]["hidden_pass"]}%'] for e in epochs]))

    # --- 3. free-running degradation by turn ------------------------------
    def turn_rates(fname):
        rows = json.loads((RESULTS / fname).read_text())["rows"]
        out = []
        for t in (1, 2, 3):
            sel = [r for r in rows if r["kind"] == "trajectory" and r["turn"] == t]
            out.append(round(100 * sum(r["hidden_pass"] for r in sel) / len(sel), 1))
        return out
    free = {v: turn_rates(f"{v}_test_best_free.json") for v in VARIANTS}
    yt3 = [(v, y_lin(v, 20, 90), f"{v}%") for v in (20, 40, 60, 80)]
    xt3 = [(x_of(i, 3), f"turn {i + 1}") for i in range(3)]
    ser3 = {v: [(x_of(i, 3), y_lin(free[v][i], 20, 90), f"turn {i + 1}", f"{free[v][i]}%")
                for i in range(3)] for v in VARIANTS}
    charts.append(line_chart("free", "Free-running: accuracy collapse across turns",
        "Each model reads its OWN previous answers. Instruct compounds its mistakes; Thinking "
        "re-derives from the instruction each turn and degrades far less. Each point is 30 turns; "
        "averaging the three gives the free-running bars in the next chart (51.1% and 72.2%).",
        ser3, xt3, yt3, "conversation turn", "hidden-pass, per turn (n=30)"))
    tables.append(table("Free-running by turn", ["turn", "instruct", "thinking"],
        [[i + 1, f"{free['instruct'][i]}%", f"{free['thinking'][i]}%"] for i in range(3)]))

    # --- 4. gold vs free --------------------------------------------------
    def traj_rate(fname):
        rows = json.loads((RESULTS / fname).read_text())["rows"]
        sel = [r for r in rows if r["kind"] == "trajectory"]
        return round(100 * sum(r["hidden_pass"] for r in sel) / len(sel), 1)
    bars = {v: [traj_rate(f"{v}_test_best_gold.json"), traj_rate(f"{v}_test_best_free.json")]
            for v in VARIANTS}
    charts.append(grouped_bars("gf", "Multi-turn: given correct history vs its own",
        "All three turns pooled (n=90), so these are the averages of the per-turn curves at left — "
        "not turn-3 values. Same turns, same models; only the conversation history differs. Instruct "
        "drops 15.6 pp when it must read its own output (p&lt;0.05); Thinking drops 4.5 pp (ns).",
        ["gold history", "free-running"], bars, "hidden-pass, all 3 turns (n=90)", 80))
    tables.append(table("Gold vs free", ["condition", "instruct", "thinking"],
        [["gold history", f"{bars['instruct'][0]}%", f"{bars['thinking'][0]}%"],
         ["free-running", f"{bars['instruct'][1]}%", f"{bars['thinking'][1]}%"]]))

    html = f"""<!doctype html>
<meta charset="utf-8">
<title>regex-sft — learning curves</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ margin:0; padding:32px 20px 64px; font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:var(--surface-1); color:var(--text-primary); }}
.viz-root {{
  --surface-1:#fcfcfb; --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#75736c;
  --grid:#e6e5e0; --axis:#c9c8c1; --s1:#2a78d6; --s2:#eb6834;
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) .viz-root {{
    --surface-1:#1a1a19; --text-primary:#fff; --text-secondary:#c3c2b7; --text-muted:#8f8e85;
    --grid:#2e2e2c; --axis:#46453f; --s1:#3987e5; --s2:#d95926;
  }}
}}
:root[data-theme="dark"] .viz-root {{
  --surface-1:#1a1a19; --text-primary:#fff; --text-secondary:#c3c2b7; --text-muted:#8f8e85;
  --grid:#2e2e2c; --axis:#46453f; --s1:#3987e5; --s2:#d95926;
}}
.wrap {{ max-width:1360px; margin:0 auto; }}
h1 {{ font-size:24px; margin:0 0 4px; letter-spacing:-.01em; }}
.sub {{ color:var(--text-secondary); margin:0 0 28px; max-width:70ch; }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(560px,1fr)); gap:28px; }}
.cell {{ min-width:0; }}
.chart {{ margin:0; background:var(--surface-1); }}
figcaption h3 {{ font-size:16px; margin:0 0 4px; }}
figcaption p {{ font-size:13px; color:var(--text-secondary); margin:0 0 6px; max-width:62ch; }}
svg {{ width:100%; height:auto; display:block; overflow:visible; }}
.grid {{ stroke:var(--grid); stroke-width:1; }}
.axis {{ stroke:var(--axis); stroke-width:1; }}
.tick {{ fill:var(--text-muted); font-size:11px; }}
.axlabel {{ fill:var(--text-secondary); font-size:12px; }}
.ln {{ fill:none; stroke-width:2; stroke-linejoin:round; stroke-linecap:round; }}
.mk {{ stroke:var(--surface-1); stroke-width:2; }}
.br {{ stroke:var(--surface-1); stroke-width:2; }}
.dlabel, .blabel {{ font-size:12px; font-weight:600; fill:var(--text-primary); }}
.s1 {{ stroke:var(--s1); }} .s1.mk, .s1.br {{ fill:var(--s1); }}
.s2 {{ stroke:var(--s2); }} .s2.mk, .s2.br {{ fill:var(--s2); }}
.legend {{ display:flex; gap:16px; margin:2px 0 8px; font-size:12.5px; color:var(--text-secondary); }}
.lg {{ display:inline-flex; align-items:center; gap:6px; }}
.lg i {{ width:14px; height:3px; border-radius:2px; display:inline-block; }}
.lg i.s1 {{ background:var(--s1); }} .lg i.s2 {{ background:var(--s2); }}
.tbl {{ margin-top:8px; font-size:12.5px; color:var(--text-secondary); }}
.tbl summary {{ cursor:pointer; }}
table {{ border-collapse:collapse; margin-top:8px; }}
th,td {{ text-align:right; padding:3px 12px 3px 0; }}
th:first-child, td:first-child {{ text-align:left; }}
th {{ color:var(--text-muted); font-weight:600; }}
circle:hover {{ r:7; }} rect:hover {{ opacity:.85; }}
</style>
<body class="viz-root"><div class="wrap">
<h1>regex-sft — learning curves</h1>
<p class="sub">Qwen3-4B Instruct vs Thinking, LoRA SFT on execution-verified text-to-regex.
Accuracy is <em>hidden-pass</em>: the generated pattern is executed against strings the model was
never shown. Hover any point for its value; each chart has a data table beneath it.</p>
<div class="grid2">
{"".join(f'<div class="cell">{c}{t}</div>' for c, t in zip(charts, tables))}
</div>
</div></body>"""

    out = RESULTS / "curves.html"
    out.write_text(html)
    print(f"wrote {out}  ({len(html) / 1024:.1f} KB, {len(charts)} charts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
