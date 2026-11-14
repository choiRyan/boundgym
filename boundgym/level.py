"""Level data model. Coordinates are (row, col), row 0 at the top.

Time model: t counts moves made so far (t=0 at the start). After each move
t increases by 1 and the unit dies if its new cell has an active hazard at
that new t. Hazard state depends only on t, so it repeats every lcm(periods).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1), "W": (0, 0)}

Cell = tuple[int, int]


@dataclass(frozen=True)
class Hazard:
    name: str
    cells: tuple[Cell, ...]
    period: int
    on: tuple[int, ...]  # phases (t % period) when the hazard is active

    def active(self, t: int) -> bool:
        return (t % self.period) in self.on

    def timeline(self) -> str:
        """One period as a string, '#' = active, '.' = safe. Index = t % period."""
        return "".join("#" if p in self.on else "." for p in range(self.period))


@dataclass
class Level:
    width: int
    height: int
    walls: frozenset[Cell]
    start: Cell
    goal: Cell
    hazards: tuple[Hazard, ...]
    max_moves: int
    seed: int = 0
    difficulty: int = 0
    optimal_moves: int | None = None
    _cell_hazards: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        self._cell_hazards = {}
        for h in self.hazards:
            for c in h.cells:
                self._cell_hazards.setdefault(c, []).append(h)

    def in_bounds(self, c: Cell) -> bool:
        return 0 <= c[0] < self.height and 0 <= c[1] < self.width

    def passable(self, c: Cell) -> bool:
        return self.in_bounds(c) and c not in self.walls

    def hazard_active(self, c: Cell, t: int) -> bool:
        return any(h.active(t) for h in self._cell_hazards.get(c, ()))

    def hazard_at(self, c: Cell) -> Hazard | None:
        hs = self._cell_hazards.get(c)
        return hs[0] if hs else None

    @property
    def cycle(self) -> int:
        out = 1
        for h in self.hazards:
            out = out * h.period // math.gcd(out, h.period)
        return out

    # ---- serialization -------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "width": self.width, "height": self.height,
            "walls": sorted(list(c) for c in self.walls),
            "start": list(self.start), "goal": list(self.goal),
            "hazards": [{"name": h.name, "cells": [list(c) for c in h.cells],
                         "period": h.period, "on": list(h.on)} for h in self.hazards],
            "max_moves": self.max_moves, "seed": self.seed,
            "difficulty": self.difficulty, "optimal_moves": self.optimal_moves,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Level":
        return cls(
            width=d["width"], height=d["height"],
            walls=frozenset(tuple(c) for c in d["walls"]),
            start=tuple(d["start"]), goal=tuple(d["goal"]),
            hazards=tuple(Hazard(h["name"], tuple(tuple(c) for c in h["cells"]),
                                 h["period"], tuple(h["on"])) for h in d["hazards"]),
            max_moves=d["max_moves"], seed=d.get("seed", 0),
            difficulty=d.get("difficulty", 0), optimal_moves=d.get("optimal_moves"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> "Level":
        return cls.from_dict(json.loads(s))
