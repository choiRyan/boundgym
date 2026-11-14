"""Exact solver. Hazards depend only on t % cycle, so BFS over
(row, col, t % cycle) finds the shortest surviving move sequence."""
from __future__ import annotations

from collections import deque

from .level import MOVES, Cell, Level

_ORDER = "RDLUW"  # deterministic tie-break


def static_distances(level: Level) -> dict[Cell, int]:
    """BFS distance to goal ignoring hazards (walls only)."""
    dist = {level.goal: 0}
    q = deque([level.goal])
    while q:
        c = q.popleft()
        for m in "RDLU":
            dr, dc = MOVES[m]
            n = (c[0] + dr, c[1] + dc)
            if level.passable(n) and n not in dist:
                dist[n] = dist[c] + 1
                q.append(n)
    return dist


def naive_moves(level: Level) -> list[str] | None:
    """Shortest path that ignores hazards entirely (the 'run straight' baseline)."""
    dist = static_distances(level)
    if level.start not in dist:
        return None
    pos, out = level.start, []
    while pos != level.goal:
        for m in "RDLU":
            dr, dc = MOVES[m]
            n = (pos[0] + dr, pos[1] + dc)
            if dist.get(n, 1 << 30) == dist[pos] - 1:
                out.append(m)
                pos = n
                break
    return out


def solve(level: Level, max_moves: int | None = None) -> list[str] | None:
    limit = level.max_moves if max_moves is None else max_moves
    L = level.cycle
    s0 = (level.start, 0)
    parent = {s0: None}
    q = deque([(level.start, 0)])  # (cell, t)
    while q:
        pos, t = q.popleft()
        if t >= limit:
            continue
        for m in _ORDER:
            dr, dc = MOVES[m]
            n = (pos[0] + dr, pos[1] + dc)
            if not level.passable(n) or level.hazard_active(n, t + 1):
                continue
            key = (n, (t + 1) % L)
            if key in parent:
                continue
            parent[key] = ((pos, t % L), m)
            if n == level.goal:
                moves, k = [], key
                while parent[k] is not None:
                    k, mv = parent[k]
                    moves.append(mv)
                return moves[::-1]
            q.append((n, t + 1))
    return None
