"""Procedural level generator with a single difficulty knob (1..10).

Layout: start on the left edge, goal on the right edge, scattered walls, and
hazard "bars" (horizontal/vertical line segments, like bound-map explosion
rows) that fire on periodic schedules. Every level is proven solvable by the
exact solver, and from difficulty 2 the naive run-straight path must die, from
difficulty 3 the optimal route must be longer than the hazard-free shortest
path (the player has to wait or detour)."""
from __future__ import annotations

import random
from dataclasses import dataclass

from .level import Hazard, Level
from .solve import naive_moves, solve, static_distances
from .verify import simulate


# Letters that never clash with moves (U D L R W) or map symbols (S G).
HAZARD_NAMES = "ABCEFHIJKMNOPQTVXYZ"


@dataclass(frozen=True)
class Params:
    width: int
    height: int
    n_hazards: int
    min_period: int
    max_period: int
    max_bar: int          # longest hazard bar
    wall_density: float
    naive_must_die: bool
    must_wait: bool
    min_waits: int        # optimal - static shortest >= this


def params_for(d: int) -> Params:
    if not 1 <= d <= 10:
        raise ValueError("difficulty must be in 1..10")
    size = 4 + d
    return Params(
        width=size + 1, height=size,
        n_hazards=1 + d,
        min_period=2, max_period=min(2 + (d + 1) // 2, 7),
        max_bar=min(2 + d // 2, size),
        wall_density=0.12 if d < 6 else 0.16,
        naive_must_die=d >= 2,
        must_wait=d >= 3,
        min_waits=0 if d < 3 else (1 if d < 6 else 2),
    )


def _rng(difficulty: int, seed: int) -> random.Random:
    return random.Random(seed * 1_000_003 + difficulty * 7919)


def _try_one(rng: random.Random, p: Params, difficulty: int, seed: int) -> Level | None:
    W, H = p.width, p.height
    start = (rng.randrange(H), 0)
    goal = (rng.randrange(H), W - 1)
    walls = set()
    for r in range(H):
        for c in range(1, W - 1):
            if rng.random() < p.wall_density:
                walls.add((r, c))
    walls -= {start, goal}

    hazards, used = [], set()
    names = iter(HAZARD_NAMES)
    for _ in range(p.n_hazards):
        horiz = rng.random() < 0.4  # vertical bars cross the left-to-right route more often
        length = rng.randint(2, p.max_bar)
        if horiz:
            r0, c0 = rng.randrange(H), rng.randrange(1, max(2, W - length))
            cells = [(r0, c0 + k) for k in range(length)]
        else:
            r0, c0 = rng.randrange(max(1, H - length + 1)), rng.randrange(1, W - 1)
            cells = [(r0 + k, c0) for k in range(length)]
        cells = [c for c in cells if 0 <= c[0] < H and 0 < c[1] < W - 1
                 and c not in walls and c not in used and c not in (start, goal)]
        if not cells:
            continue
        period = rng.randint(p.min_period, p.max_period)
        duty = rng.randint(1, period - 1)
        offset = rng.randrange(period)
        on = tuple(sorted((offset + k) % period for k in range(duty)))
        hazards.append(Hazard(next(names), tuple(cells), period, on))
        used.update(cells)

    lvl = Level(W, H, frozenset(walls), start, goal, tuple(hazards), max_moves=10_000,
                seed=seed, difficulty=difficulty)
    dist = static_distances(lvl)
    if start not in dist:
        return None
    sol = solve(lvl, max_moves=4 * W * H)
    if sol is None:
        return None
    opt, static = len(sol), dist[start]
    if p.must_wait and opt - static < max(1, p.min_waits):
        return None
    slack = max(3, opt // 3)
    lvl = Level(W, H, frozenset(walls), start, goal, tuple(hazards),
                max_moves=opt + slack, seed=seed, difficulty=difficulty, optimal_moves=opt)
    if p.naive_must_die:
        nm = naive_moves(lvl)
        if nm is not None and simulate(lvl, nm).success:
            return None
    return lvl


def generate(difficulty: int, seed: int, max_tries: int = 5000) -> Level:
    """Deterministic: same (difficulty, seed) always gives the same level."""
    p = params_for(difficulty)
    rng = _rng(difficulty, seed)
    for _ in range(max_tries):
        lvl = _try_one(rng, p, difficulty, seed)
        if lvl is not None:
            return lvl
    raise RuntimeError(f"could not generate level d={difficulty} seed={seed}")
