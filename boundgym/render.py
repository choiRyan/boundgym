"""Text rendering for LLM prompts."""
from __future__ import annotations

from .level import Level

SYSTEM_PROMPT = (
    "You solve timed obstacle courses. Think step by step if you like, then give "
    "your final answer as moves inside <moves></moves>."
)


def render_map(level: Level) -> str:
    header = "   " + "".join(str(c % 10) for c in range(level.width))
    rows = [header]
    for r in range(level.height):
        line = []
        for c in range(level.width):
            cell = (r, c)
            if cell == level.start:
                ch = "S"
            elif cell == level.goal:
                ch = "G"
            elif cell in level.walls:
                ch = "#"
            else:
                h = level.hazard_at(cell)
                ch = h.name if h else "."
            line.append(ch)
        rows.append(f"{r:>2} " + "".join(line))
    return "\n".join(rows)


def render_prompt(level: Level) -> str:
    haz = []
    for h in level.hazards:
        phases = ", ".join(str(p) for p in h.on)
        haz.append(f"  {h.name}: active when t % {h.period} is in {{{phases}}}   "
                   f"(pattern from t%{h.period}=0: {h.timeline()})")
    haz_text = "\n".join(haz) if haz else "  (none)"
    return f"""Guide the unit from S to G through this timed obstacle course.

Map (row numbers on the left, column numbers on top):
{render_map(level)}

Legend: S start, G goal, # wall, . floor, capital letters are hazard cells.

Hazards (t is the number of moves made so far; t = 0 at the start):
{haz_text}

Rules:
- Each move is one of U (up), D (down), L (left), R (right), W (wait in place).
- Every move, including W, increases t by 1.
- After a move, if the unit stands on a hazard cell that is active at the new t, it dies.
- Moving into a wall or off the map fails the course.
- You must reach G within {level.max_moves} moves.

Give your final answer as space-separated moves, for example <moves>R R D W R</moves>."""


def hazard_timeline(level: Level, horizon: int | None = None) -> str:
    """One line per step t = 1..horizon listing the hazards active at that t."""
    horizon = level.max_moves if horizon is None else horizon
    lines = []
    for t in range(1, horizon + 1):
        on = [h.name for h in level.hazards if h.active(t)]
        lines.append(f"  t={t:>2}: {' '.join(on) if on else '-'}")
    return "\n".join(lines)


def render_prompt_timeline(level: Level) -> str:
    """Same as render_prompt, but the hazard rule is spelled out per step
    (no modulo arithmetic needed). Tests reading vs. reasoning."""
    return f"""Guide the unit from S to G through this timed obstacle course.

Map (row numbers on the left, column numbers on top):
{render_map(level)}

Legend: S start, G goal, # wall, . floor, capital letters are hazard cells.

Hazard schedule (t is the number of moves made so far; t = 0 at the start).
Each line lists the hazards that are active at that t ("-" means none):
{hazard_timeline(level)}

Rules:
- Each move is one of U (up), D (down), L (left), R (right), W (wait in place).
- Every move, including W, increases t by 1.
- After a move, if the unit stands on a hazard cell that is active at the new t, it dies.
- Moving into a wall or off the map fails the course.
- You must reach G within {level.max_moves} moves.

Give your final answer as space-separated moves, for example <moves>R R D W R</moves>."""


# Prompt variants by name. "modulo" is the original and the reference for all ch 1 results.
PROMPTS = {"modulo": render_prompt, "timeline": render_prompt_timeline}
