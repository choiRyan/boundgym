"""Verifier: replay a move list and score it. This is the reward function.

It is hard to reward-hack by construction: the only way to get reward 1.0 is
a move sequence that actually reaches the goal alive within max_moves.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .level import MOVES, Cell, Level


@dataclass
class Result:
    success: bool
    reason: str          # "goal" | "hazard" | "illegal_move" | "out_of_moves" | "did_not_reach_goal" | "bad_format"
    moves_used: int
    final_pos: Cell | None
    progress: float      # 0..1, fraction of static distance to goal closed at the furthest point


def simulate(level: Level, moves: list[str]) -> Result:
    from .solve import static_distances  # local import avoids a cycle
    dist = static_distances(level)
    d0 = max(dist.get(level.start, 1), 1)
    best = d0
    pos, t = level.start, 0

    def prog():
        return max(0.0, 1.0 - best / d0)

    for m in moves:
        if t >= level.max_moves:
            return Result(False, "out_of_moves", t, pos, prog())
        dr, dc = MOVES[m]
        nxt = (pos[0] + dr, pos[1] + dc)
        if not level.passable(nxt):
            return Result(False, "illegal_move", t, pos, prog())
        t += 1
        pos = nxt
        if level.hazard_active(pos, t):
            return Result(False, "hazard", t, pos, prog())
        best = min(best, dist.get(pos, best))
        if pos == level.goal:
            return Result(True, "goal", t, pos, 1.0)
    return Result(False, "did_not_reach_goal", t, pos, prog())


# Body may not contain another "<moves>": a stray opening tag earlier in the reasoning
# must not swallow everything up to the final answer's closing tag.
_TAG = re.compile(r"<moves>((?:(?!<moves>).)*?)</moves>", re.IGNORECASE | re.DOTALL)
_TOKEN = re.compile(r"[UDLRW]")
_ALLOWED = re.compile(r"^[\sUDLRW,;|\-\[\]'\"]*$", re.IGNORECASE)


def parse_moves(text: str) -> list[str] | None:
    """Take the LAST <moves>...</moves> block. Returns None if missing or if it
    contains anything other than move letters and separators."""
    blocks = _TAG.findall(text or "")
    if not blocks:
        return None
    body = blocks[-1]
    if not _ALLOWED.match(body):
        return None
    return _TOKEN.findall(body.upper())


def reward(level: Level, completion: str, shaped: bool = False) -> float:
    """1.0 on success. With shaped=True, failures get up to 0.2 for progress
    (useful early in GRPO when almost nothing succeeds). Bad format gets 0."""
    moves = parse_moves(completion)
    if moves is None:
        return 0.0
    r = simulate(level, moves)
    if r.success:
        return 1.0
    return 0.2 * r.progress if shaped else 0.0
