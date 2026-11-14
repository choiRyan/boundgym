"""Static SVG figures for the README (standard library only). Writes docs/figures/*.svg.

  .venv/bin/python scripts/make_readme_figures.py

1. eval_vs_base.svg: solve-rate difference vs base (pts, paired bootstrap 95% CI), per checkpoint, d3 and d5.
2. training_budget.svg: share cut off and "solved, of those that finished" over training steps, g2 (8k) vs g4
   (16k), same levels at every step, 10-step ratio-of-sums.
Light and dark styles switch with the viewer's colour scheme (CSS media query inside the SVG).
"""
from __future__ import annotations

import glob
import json
import os
import random
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "docs", "figures")

# Reference palette (dataviz skill), validated light + dark for these two slots.
STYLE = """<style>
  .bg{fill:#fcfcfb} .t1{fill:#0b0b0b} .t2{fill:#52514e} .mu{fill:#898781}
  .grid{stroke:#e1e0d9;stroke-width:1} .base{stroke:#c3c2b7;stroke-width:1}
  .s1{stroke:#2a78d6} .f1{fill:#2a78d6} .s2{stroke:#eb6834} .f2{fill:#eb6834}
  .sg{stroke:#898781} .fg{fill:#898781}
  text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
  @media (prefers-color-scheme: dark){
    .bg{fill:#1a1a19} .t1{fill:#f0efec} .t2{fill:#c3c2b7}
    .grid{stroke:#2c2c2a} .base{stroke:#383835}
    .s1{stroke:#3987e5} .f1{fill:#3987e5} .s2{stroke:#d95926} .f2{fill:#d95926}
  }
</style>"""


def svg(w, h, body, label):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='{w}' height='{h}' "
            f"role='img' aria-label='{label}'>{STYLE}<rect class='bg' width='{w}' height='{h}' rx='8'/>{body}</svg>\n")


# ---------------------------------------------------------------- figure 1: eval vs base
def per_level(path):
    d = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        d[r["idx"]].append(r["success"])
    return {k: sum(v) / len(v) for k, v in d.items()}


def paired(base, other, rng):
    idx = sorted(set(base) & set(other))
    diffs = [other[i] - base[i] for i in idx]
    boots = sorted(sum(rng.choice(diffs) for _ in idx) / len(idx) for _ in range(4000))
    return 100 * sum(diffs) / len(idx), 100 * boots[100], 100 * boots[3899]


def eval_figure():
    rng = random.Random(0)
    rows = [  # label, results root, dir, highlighted
        ("g1 · 8k budget", "results", "g1-4b-d234-8k_checkpoint-100", False),
        ("g2 @100 · 8k", "results", "g2-4b-d234-8k-nomask_checkpoint-100", False),
        ("g2 @200 · 8k", "results", "g2-4b-d234-8k-nomask_checkpoint-200", False),
        ("g4 · 16k budget", "results", "g4-4b-d234-16k-nomask_checkpoint-100", True),
        ("g4 · 16k, fresh levels", "results_rep", "g4-4b-d234-16k-nomask_checkpoint-100", True),
    ]
    data = {}
    for s in ("d3", "d5"):
        for label, root, d, hi in rows:
            b = per_level(glob.glob(os.path.join(ROOT, "chapter3", root, "base", f"*_{s}.jsonl"))[0])
            o = per_level(glob.glob(os.path.join(ROOT, "chapter3", root, d, f"*_{s}.jsonl"))[0])
            data[(s, label)] = paired(b, o, rng)

    W, H = 860, 330
    lab_w, pad_t, row_h = 170, 74, 44
    pw, gap = 300, 40
    lo, hi_ = -10, 30
    body = ("<text class='t1' x='20' y='30' font-size='17' font-weight='600'>Solve rate vs the untrained model "
            "(Qwen3-4B), percentage points</text>"
            "<text class='t2' x='20' y='52' font-size='13'>Dot: difference on the same levels, 4 tries each, 32k "
            "budget. Bar: 95% range. A bar clear of 0 is a real gain.</text>")
    for i, (label, *_r, hi) in enumerate(rows):
        y = pad_t + 38 + i * row_h
        body += f"<text class='{'t1' if hi else 't2'}' x='20' y='{y + 4}' font-size='13'" \
                f"{' font-weight=\"600\"' if hi else ''}>{label}</text>"
    for p, s in enumerate(("d3", "d5")):
        x0 = lab_w + 20 + p * (pw + gap)
        X = lambda v: x0 + pw * (v - lo) / (hi_ - lo)
        sub = "in training range" if s == "d3" else "harder than training"
        body += f"<text class='t1' x='{x0}' y='{pad_t + 6}' font-size='14' font-weight='600'>{s}</text>" \
                f"<text class='mu' x='{x0 + 26}' y='{pad_t + 6}' font-size='12'>{sub}</text>"
        for v in range(lo, hi_ + 1, 10):
            cls = "base" if v == 0 else "grid"
            body += f"<line class='{cls}' x1='{X(v):.1f}' x2='{X(v):.1f}' y1='{pad_t + 18}' y2='{pad_t + 18 + len(rows) * row_h}'/>" \
                    f"<text class='mu' x='{X(v):.1f}' y='{pad_t + 34 + len(rows) * row_h}' font-size='11' " \
                    f"text-anchor='middle'>{v:+d}</text>"
        for i, (label, *_r, hi) in enumerate(rows):
            y = pad_t + 38 + i * row_h
            m, a, b = data[(s, label)]
            sc, fc = ("s1", "f1") if hi else ("sg", "fg")
            body += f"<line class='{sc}' stroke-width='2' stroke-linecap='round' x1='{X(a):.1f}' x2='{X(b):.1f}' y1='{y}' y2='{y}'/>" \
                    f"<circle class='{fc}' cx='{X(m):.1f}' cy='{y}' r='5'/>" \
                    f"<text class='{'t1' if hi else 't2'}' x='{X(b) + 8:.1f}' y='{y + 4}' font-size='12'>" \
                    f"{int(round(m)):+d} [{int(round(a)):+d}, {int(round(b)):+d}]</text>"
    return svg(W, H, body, "Solve-rate difference versus base per checkpoint with 95% intervals, d3 and d5")


# ---------------------------------------------------------------- figure 2: budget decomposition
def metrics(run):
    last = {}
    for line in open(os.path.join(ROOT, "chapter3", "runs", run, "metrics.jsonl")):
        d = json.loads(line)
        if "reward" in d:
            last[d["step"]] = (float(d["reward"]), float(d["completions/clipped_ratio"]))
    return [last[k] for k in sorted(last)][:100]


def rolling(rows, fn):
    out = []
    for i in range(len(rows)):
        w = rows[max(0, i - 9):i + 1]
        out.append(fn(w))
    return out


def budget_figure():
    runs = {"g2 · 8k budget": ("g2-4b-d234-8k-nomask", "2"), "g4 · 16k budget": ("g4-4b-d234-16k-nomask", "1")}
    series = {}
    for name, (run, slot) in runs.items():
        r = metrics(run)
        cut = rolling(r, lambda w: sum(x[1] for x in w) / len(w))
        sof = rolling(r, lambda w: sum(x[0] for x in w) / sum(1 - x[1] for x in w))
        series[name] = (slot, cut, sof)

    W, H = 860, 360
    pad_t, ph, pw, gap, x_l = 96, 200, 330, 70, 60
    body = ("<text class='t1' x='20' y='30' font-size='17' font-weight='600'>Why the token budget mattered: "
            "same levels, same steps, two budgets</text>"
            "<text class='t2' x='20' y='52' font-size='13'>A cut-off attempt scores 0, so reward = (1 − share cut "
            "off) × solved-of-finished. At 8k the cheap win was finishing; at 16k it was solving.</text>")
    # legend (always present for 2 series)
    lx = 20
    for name, (slot, *_s) in series.items():
        body += f"<line class='s{slot}' stroke-width='3' x1='{lx}' x2='{lx + 22}' y1='74' y2='74'/>" \
                f"<text class='t2' x='{lx + 28}' y='78' font-size='12'>{name}</text>"
        lx += 150
    panels = [("Share of attempts cut off at the budget", 1), ("Solved, of the attempts that finished", 2)]
    for p, (title, k) in enumerate(panels):
        x0 = x_l + p * (pw + gap)
        X = lambda step: x0 + pw * (step - 1) / 99
        Y = lambda v: pad_t + 20 + ph * (1 - v)
        body += f"<text class='t1' x='{x0}' y='{pad_t + 4}' font-size='14' font-weight='600'>{title}</text>"
        for v in (0, 0.25, 0.5, 0.75, 1):
            body += f"<line class='{'base' if v == 0 else 'grid'}' x1='{x0}' x2='{x0 + pw}' y1='{Y(v):.1f}' y2='{Y(v):.1f}'/>" \
                    f"<text class='mu' x='{x0 - 6}' y='{Y(v) + 4:.1f}' font-size='11' text-anchor='end'>{v:.0%}</text>"
        for s in (1, 50, 100):
            body += f"<text class='mu' x='{X(s):.1f}' y='{Y(0) + 18:.1f}' font-size='11' text-anchor='middle'>{s}</text>"
        body += f"<text class='mu' x='{x0 + pw / 2:.1f}' y='{Y(0) + 34:.1f}' font-size='11' text-anchor='middle'>training step</text>"
        ends = []
        for name, (slot, *vals) in series.items():
            ys = vals[k - 1]
            path = " ".join(f"{'M' if i == 0 else 'L'}{X(i + 1):.1f},{Y(v):.1f}" for i, v in enumerate(ys))
            body += f"<path class='s{slot}' d='{path}' fill='none' stroke-width='2' stroke-linejoin='round'/>"
            ends.append((Y(ys[-1]), ys[-1], name.split(" ·")[0]))
        ends.sort()
        if len(ends) == 2 and ends[1][0] - ends[0][0] < 14:  # keep end labels apart
            ends[0] = (ends[1][0] - 14, *ends[0][1:])
        for y, v, short in ends:
            body += f"<text class='t2' x='{x0 + pw + 6}' y='{y + 4:.1f}' font-size='12'>{short} {v:.0%}</text>"
    return svg(W, H, body, "Share cut off and solved-of-finished over 100 training steps for g2 at 8k and g4 at 16k")


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("eval_vs_base.svg", eval_figure), ("training_budget.svg", budget_figure)):
        with open(os.path.join(OUT, name), "w") as f:
            f.write(fn())
        print("wrote", os.path.join("docs", "figures", name))


if __name__ == "__main__":
    main()
