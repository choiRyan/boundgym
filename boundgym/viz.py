"""Animated GIF rendering of a level and a move plan (needs `pip install -e ".[viz]"`).

    from boundgym.viz import trace, render_frames, save_gif
    tr = trace(level, list("RRWDR"))
    save_gif(render_frames(level, tr, title="oracle"), "out.gif")

Each frame is the state after t moves. Below the map, a hazard schedule strip shows
which hazard is active at every t, with a cursor at the current t: the timing the
model has to plan around. Outcome banners always pair an icon and a label with the
status color, so color never carries meaning alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont

from .level import MOVES, Cell, Level
from .verify import simulate

# Palette: dataviz-skill reference values (surface/ink/grid, series-1, status).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
FLOOR = "#f4f3ee"
WALL = "#3a3a38"
AGENT = "#2a78d6"
TRAIL = "#9cc1ee"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
WARNING = "#fab219"
HAZ_IDLE = "#f3dcd6"
HAZ_IDLE_INK = "#a4574a"

_FONT_DIR = "/usr/share/fonts/truetype/dejavu/"


def _font(size: int, bold: bool = False, mono: bool = False):
    name = "DejaVuSansMono" if mono else "DejaVuSans"
    name += "-Bold.ttf" if bold else ".ttf"
    try:
        return ImageFont.truetype(_FONT_DIR + name, size)
    except OSError:  # pragma: no cover
        return ImageFont.load_default(size)


@dataclass
class Trace:
    moves: list[str]
    positions: list[Cell]          # positions[t] = cell after t moves (positions[0] = start)
    reason: str                    # same vocabulary as verify.Result.reason (+ "no_answer")
    attempted: Cell | None = None  # target of the illegal move, if any
    killer: str | None = None      # hazard name that killed the unit
    note: str = ""                 # extra caption (e.g. why there was no answer)
    extras: dict = field(default_factory=dict)


def trace(level: Level, moves: list[str] | None, no_answer_note: str = "") -> Trace:
    """Replay moves like verify.simulate, keeping every position for animation."""
    if moves is None:
        return Trace([], [level.start], "no_answer", note=no_answer_note)
    pos, t, path = level.start, 0, [level.start]
    reason, attempted, killer = "did_not_reach_goal", None, None
    for m in moves:
        if t >= level.max_moves:
            reason = "out_of_moves"
            break
        dr, dc = MOVES[m]
        nxt = (pos[0] + dr, pos[1] + dc)
        if not level.passable(nxt):
            reason, attempted = "illegal_move", nxt
            break
        t += 1
        pos = nxt
        path.append(pos)
        if level.hazard_active(pos, t):
            reason = "hazard"
            killer = next(h.name for h in level.hazards if pos in h.cells and h.active(t))
            break
        if pos == level.goal:
            reason = "goal"
            break
    assert reason == simulate(level, moves).reason, "trace disagrees with verifier"
    return Trace(list(moves), path, reason, attempted, killer)


# ---------------------------------------------------------------- drawing

def _layout(level: Level, cell: int):
    pad, top = 16, 64
    label = 22
    map_w, map_h = level.width * cell, level.height * cell
    strip_cell = max(10, min(18, (map_w + label - 40) // max(level.max_moves, 1)))
    strip_h = 18 + len(level.hazards) * (strip_cell + 3) + 16
    width = max(pad * 2 + label + map_w, pad * 2 + 30 + strip_cell * level.max_moves, 360)
    height = top + label + map_h + 14 + strip_h + 58
    return dict(pad=pad, top=top, label=label, map_w=map_w, map_h=map_h, strip_cell=strip_cell,
                strip_h=strip_h, width=width, height=height)


def _banner(reason: str, tr: Trace, level: Level, t: int):
    if reason == "goal":
        return GOOD, "✓", f"Reached the goal in {t} moves (optimal {level.optimal_moves})"
    if reason == "hazard":
        return CRITICAL, "✕", f"Killed by hazard {tr.killer}, active at t={t}"
    if reason == "illegal_move":
        what = "off the map" if not level.in_bounds(tr.attempted) else "into a wall"
        return CRITICAL, "✕", f"Illegal move at t={t}: walked {what}"
    if reason == "out_of_moves":
        return WARNING, "!", f"Out of moves (limit {level.max_moves})"
    if reason == "did_not_reach_goal":
        return WARNING, "!", f"Plan ended at t={t}, short of the goal"
    if reason == "no_answer":
        return WARNING, "!", tr.note or "No answer given"
    return INK_3, "", ""


def draw_frame(level: Level, tr: Trace, t: int, *, cell: int = 44, title: str = "",
               subtitle: str = "", final: bool = False) -> Image.Image:
    L = _layout(level, cell)
    img = Image.new("RGB", (L["width"], L["height"]), SURFACE)
    d = ImageDraw.Draw(img)
    f_title, f_sub = _font(17, bold=True), _font(12)
    f_cell, f_lab = _font(int(cell * 0.42), bold=True), _font(11, mono=True)
    pos = tr.positions[min(t, len(tr.positions) - 1)]

    # header
    d.text((L["pad"], 10), title, font=f_title, fill=INK)
    d.text((L["pad"], 33), subtitle, font=f_sub, fill=INK_2)
    # move tape: past moves dark, current move highlighted, future moves muted
    x = L["pad"]
    y_tape = 48
    for i, m in enumerate(tr.moves):
        col = INK if i < t else INK_3
        if i == t - 1:
            d.rounded_rectangle((x - 2, y_tape - 1, x + 11, y_tape + 14), 3, fill=AGENT)
            col = SURFACE
        d.text((x, y_tape), m, font=f_lab, fill=col)
        x += 14
        if x > L["width"] - 30:
            d.text((x, y_tape), "…", font=f_lab, fill=INK_3)
            break

    # map
    ox, oy = L["pad"] + L["label"], L["top"] + L["label"]
    for c in range(level.width):
        d.text((ox + c * cell + cell // 2 - 3, oy - 16), str(c % 10), font=f_lab, fill=INK_3)
    for r in range(level.height):
        d.text((ox - 16, oy + r * cell + cell // 2 - 7), str(r % 10), font=f_lab, fill=INK_3)
    for r in range(level.height):
        for c in range(level.width):
            x0, y0 = ox + c * cell, oy + r * cell
            box = (x0 + 1, y0 + 1, x0 + cell - 2, y0 + cell - 2)
            cc = (r, c)
            h = level.hazard_at(cc)
            if cc in level.walls:
                d.rounded_rectangle(box, 4, fill=WALL)
                continue
            if h is not None:
                active = h.active(t)
                d.rounded_rectangle(box, 4, fill=CRITICAL if active else HAZ_IDLE)
                ink = SURFACE if active else HAZ_IDLE_INK
                label = h.name + ("!" if active else "")
                tw = d.textlength(label, font=f_cell)
                d.text((x0 + (cell - tw) / 2, y0 + cell * 0.25), label, font=f_cell, fill=ink)
                continue
            fill = FLOOR
            d.rounded_rectangle(box, 4, fill=fill, outline=GRID)
            if cc == level.start:
                d.text((x0 + cell * 0.33, y0 + cell * 0.25), "S", font=f_cell, fill=INK_2)
            if cc == level.goal:
                d.rounded_rectangle(box, 4, fill=SURFACE, outline=GOOD, width=3)
                d.text((x0 + cell * 0.32, y0 + cell * 0.25), "G", font=f_cell, fill=GOOD)

    # trail
    centers = [(ox + c * cell + cell / 2, oy + r * cell + cell / 2) for r, c in tr.positions[: t + 1]]
    if len(centers) > 1:
        d.line(centers, fill=TRAIL, width=4, joint="curve")
    # wait badges: how many times the unit waited on each cell so far
    waits: dict = {}
    for i in range(1, min(t, len(tr.positions) - 1) + 1):
        if tr.positions[i] == tr.positions[i - 1]:
            waits[tr.positions[i]] = waits.get(tr.positions[i], 0) + 1
    f_badge = _font(11, bold=True)
    for (wr, wc), n in waits.items():
        bx, by = ox + wc * cell + cell - 4, oy + wr * cell + 4
        txt = "W" if n == 1 else f"W\u00d7{n}"
        tw = d.textlength(txt, font=f_badge)
        d.rounded_rectangle((bx - tw - 8, by, bx, by + 15), 7, fill=INK)
        d.text((bx - tw - 4, by + 1), txt, font=f_badge, fill=SURFACE)
    # agent
    ax, ay = centers[-1]
    rad = cell * 0.3
    dead = final and tr.reason == "hazard"
    d.ellipse((ax - rad - 2, ay - rad - 2, ax + rad + 2, ay + rad + 2), fill=SURFACE)
    d.ellipse((ax - rad, ay - rad, ax + rad, ay + rad), fill=INK if dead else AGENT)
    if dead:
        k = rad * 0.55
        d.line((ax - k, ay - k, ax + k, ay + k), fill=SURFACE, width=3)
        d.line((ax - k, ay + k, ax + k, ay - k), fill=SURFACE, width=3)
    if final and tr.reason == "illegal_move" and tr.attempted is not None:
        tr_, tc = tr.attempted
        tx, ty = ox + tc * cell + cell / 2, oy + tr_ * cell + cell / 2
        mx, my = (ax + tx) / 2, (ay + ty) / 2
        d.line((ax, ay, mx, my), fill=CRITICAL, width=4)
        d.ellipse((mx - 9, my - 9, mx + 9, my + 9), fill=CRITICAL)
        d.text((mx - 4, my - 8), "✕", font=_font(12, bold=True), fill=SURFACE)

    # hazard schedule strip
    sy = oy + L["map_h"] + 14
    sc = L["strip_cell"]
    sx = L["pad"] + 30
    d.text((L["pad"], sy), "hazard schedule  (t = 1 … max moves, ▼ = now)", font=f_lab, fill=INK_2)
    sy += 18
    if t >= 1:
        cx = sx + (t - 1) * sc + sc / 2
        d.polygon([(cx - 5, sy - 5), (cx + 5, sy - 5), (cx, sy + 1)], fill=INK)
        d.rectangle((sx + (t - 1) * sc, sy + 1, sx + t * sc - 1,
                     sy + 3 + len(level.hazards) * (sc + 3)), outline=INK)
    for i, h in enumerate(level.hazards):
        y = sy + 3 + i * (sc + 3)
        d.text((L["pad"] + 8, y + sc / 2 - 7), h.name, font=f_lab, fill=INK_2)
        for k in range(1, level.max_moves + 1):
            x0 = sx + (k - 1) * sc
            on = h.active(k)
            d.rectangle((x0 + 1, y, x0 + sc - 2, y + sc - 1), fill=CRITICAL if on else GRID)

    # footer: outcome banner, or running status
    fy = L["height"] - 44
    if final:
        col, icon, text = _banner(tr.reason, tr, level, t)
        d.rounded_rectangle((L["pad"], fy, L["width"] - L["pad"], fy + 32), 6, fill=col)
        ink = INK if col == WARNING else SURFACE
        msg, size = f"{icon}  {text}", 14
        while size > 9 and d.textlength(msg, font=_font(size, bold=True)) > L["width"] - 2 * L["pad"] - 20:
            size -= 1
        d.text((L["pad"] + 10, fy + 16 - size * 0.65), msg, font=_font(size, bold=True), fill=ink)
    else:
        d.text((L["pad"], fy + 8), f"t = {t}   moves left = {level.max_moves - t}", font=_font(13), fill=INK_2)
    return img


STEP_MS = 450   # one move
HOLD_MS = 2600  # pause on the very last frame of an animation


def evenly_timed(frames):
    """Combined animations: every frame advances at STEP_MS and only the final frame holds. (Taking the max
    of per-panel durations made a finished panel's hold slow every later step of the others.)"""
    return [(im, HOLD_MS if i == len(frames) - 1 else STEP_MS) for i, (im, _) in enumerate(frames)]


def render_frames(level: Level, tr: Trace, *, cell: int = 44, title: str = "", subtitle: str = ""):
    """List of (image, duration_ms). One frame per t, the last held longer."""
    end = len(tr.positions) - 1
    frames = []
    for t in range(0, end + 1):
        last = t == end
        if last:
            frames.append((draw_frame(level, tr, t, cell=cell, title=title, subtitle=subtitle), STEP_MS))
            frames.append((draw_frame(level, tr, t, cell=cell, title=title, subtitle=subtitle, final=True), HOLD_MS))
        else:
            frames.append((draw_frame(level, tr, t, cell=cell, title=title, subtitle=subtitle), STEP_MS))
    return frames


def side_by_side(*frame_lists, gap: int = 12):
    """Combine several frame lists into one, padding shorter ones with their last frame."""
    n = max(len(f) for f in frame_lists)
    padded = [f + [f[-1]] * (n - len(f)) for f in frame_lists]
    out = []
    for i in range(n):
        imgs = [p[i][0] for p in padded]
        w = sum(im.width for im in imgs) + gap * (len(imgs) - 1)
        h = max(im.height for im in imgs)
        canvas = Image.new("RGB", (w, h), SURFACE)
        x = 0
        for im in imgs:
            canvas.paste(im, (x, 0))
            x += im.width + gap
        out.append((canvas, STEP_MS))
    return evenly_timed(out)


def save_gif(frames, path: str):
    imgs = [f[0] for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=[f[1] for f in frames],
                 loop=0, optimize=True, disposal=2)
