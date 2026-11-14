"""Build demo GIFs (and MP4s if ffmpeg is present) of the environment.

  python scripts/make_demo.py            # → docs/media/

1. timing.gif        same level: naive run-straight (dies) vs exact solver (waits, wins)
2. modes/*.gif       real Qwen3-4B attempts (think, 32k) for each outcome, plus a cut-off 8k attempt
3. failure_modes.gif the outcome GIFs in one 3x2 grid
All examples come from the frozen eval sets and saved results; every trace is
checked against the verifier.
"""
import argparse, json, os, shutil, subprocess, sys
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from PIL import Image
from boundgym import Level, naive_moves, parse_moves, solve
from boundgym.viz import STEP_MS, SURFACE, evenly_timed, render_frames, save_gif, side_by_side, trace

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(ROOT, "docs", "media"))
ap.add_argument("--difficulty", type=int, default=3)
ap.add_argument("--results", default=os.path.join(ROOT, "chapter1", "results"))
a = ap.parse_args()
os.makedirs(os.path.join(a.out, "modes"), exist_ok=True)

levels = [Level.from_json(l) for l in open(os.path.join(ROOT, "levels", f"d{a.difficulty}.jsonl"))]
D = a.difficulty


def mp4(gif):
    """MP4 (H.264) and WebM (VP9) next to the GIF, same frame timing (ffmpeg reads the GIF's delays)."""
    if not shutil.which("ffmpeg"):
        return
    even = ["-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", gif, "-movflags", "faststart", *even,
                    gif[:-4] + ".mp4"], check=False)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", gif, "-c:v", "libvpx-vp9", "-crf", "30",
                    "-b:v", "0", "-row-mt", "1", *even, gif[:-4] + ".webm"], check=False)


def stack(rows, gap=12):
    n = max(len(r) for r in rows)
    rows = [r + [r[-1]] * (n - len(r)) for r in rows]
    out = []
    for i in range(n):
        imgs = [r[i][0] for r in rows]
        w, h = max(im.width for im in imgs), sum(im.height for im in imgs) + gap * (len(imgs) - 1)
        canvas, y = Image.new("RGB", (w, h), SURFACE), 0
        for im in imgs:
            canvas.paste(im, (0, y))
            y += im.height + gap
        out.append((canvas, STEP_MS))
    return evenly_timed(out)


# 1. timing: a level where naive dies to a hazard and the optimum includes waits
for idx, lvl in enumerate(levels):
    nm, sol = naive_moves(lvl), solve(lvl)
    tn = trace(lvl, nm)
    if tn.reason == "hazard" and "W" in sol and len(tn.positions) >= 4:
        break
to = trace(lvl, sol)
sub = f"d{D} level #{idx} (seed {lvl.seed}), limit {lvl.max_moves} moves"
fr = side_by_side(render_frames(lvl, tn, title="Naive: run the shortest path", subtitle=sub),
                  render_frames(lvl, to, title="Solver: wait for the gap", subtitle=sub))
p = os.path.join(a.out, "timing.gif"); save_gif(fr, p); mp4(p); print("wrote", p)

# 2. real model outcomes
def rows(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []

m32 = rows(os.path.join(a.results, "think32k", f"Qwen_Qwen3-4B_d{D}.jsonl"))
m8 = rows(os.path.join(a.results, f"Qwen_Qwen3-4B_d{D}.jsonl"))
labels = {"goal": "Success", "hazard": "Hazard death", "illegal_move": "Illegal move",
          "did_not_reach_goal": "Stopped short", "out_of_moves": "Out of moves"}
picked = []
for reason, title in labels.items():
    # prefer short-ish plans on small maps so the GIF stays readable
    cands = sorted((r for r in m32 if r["reason"] == reason), key=lambda r: len(r["moves"]))
    if not cands:
        continue
    r = cands[len(cands) // 2]
    lvl = levels[r["idx"]]
    tr = trace(lvl, parse_moves(r["completion"]))
    assert tr.reason == reason
    picked.append((reason, title, lvl, tr, r))
cut = next((r for r in m8 if r["reason"] == "truncated"), None)
if cut:
    lvl = levels[cut["idx"]]
    tr = trace(lvl, None, no_answer_note=f"No answer: thinking cut off at {cut['completion_tokens']} tokens")
    picked.append(("truncated", "Ran out of tokens (8k budget)", lvl, tr, cut))

grid = []
for reason, title, lvl, tr, r in picked:
    tok = r.get("completion_tokens") or 0
    sub = f"Qwen3-4B, d{D} level #{r['idx']}, {tok/1000:.1f}k tokens of thinking"
    fr = render_frames(lvl, tr, title=title, subtitle=sub)
    p = os.path.join(a.out, "modes", f"{reason}.gif"); save_gif(fr, p); mp4(p); print("wrote", p)
    grid.append(fr)

# 3. grid, 3 columns
if grid:
    rws = [side_by_side(*grid[i:i + 3]) for i in range(0, len(grid), 3)]
    p = os.path.join(a.out, "failure_modes.gif"); save_gif(stack(rws), p); mp4(p); print("wrote", p)
