"""Gymnasium env over the same levels, for the tiny from-scratch PPO baseline.

Observation (C, H_max, W_max) float32 grid, zero-padded to the largest size:
  0 walls, 1 goal, 2 agent, 3 in-bounds mask,
  4..4+K-1  hazard active at t+1 .. t+K (lookahead makes the state Markov)
plus a scalar "moves left" fraction broadcast in the last channel.
Actions: 0 U, 1 D, 2 L, 3 R, 4 W. Episode ends on goal (+1), death (-1),
illegal move (-1) or running out of moves (0).
"""
from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install gymnasium numpy") from e

from .generate import generate, params_for
from .level import MOVES, Level

ACTIONS = "UDLRW"


class BoundGridEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, difficulty: int | tuple[int, int] = 3, lookahead: int = 6,
                 levels: list[Level] | None = None, step_penalty: float = 0.0,
                 render_mode: str | None = None):
        self.dmin, self.dmax = (difficulty, difficulty) if isinstance(difficulty, int) else difficulty
        self.levels = levels
        self.K = lookahead
        self.step_penalty = step_penalty
        self.render_mode = render_mode
        if levels:
            self.Hm = max(l.height for l in levels)
            self.Wm = max(l.width for l in levels)
        else:
            p = params_for(self.dmax)
            self.Hm, self.Wm = p.height, p.width
        self.C = 4 + self.K + 1
        self.observation_space = spaces.Box(0.0, 1.0, (self.C, self.Hm, self.Wm), np.float32)
        self.action_space = spaces.Discrete(5)
        self.level: Level | None = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "level" in options:
            self.level = options["level"]
        elif self.levels:
            self.level = self.levels[self.np_random.integers(len(self.levels))]
        else:
            d = int(self.np_random.integers(self.dmin, self.dmax + 1))
            self.level = generate(d, int(self.np_random.integers(2**31)))
        self.pos, self.t = self.level.start, 0
        self._static = self._static_planes()
        return self._obs(), {"difficulty": self.level.difficulty}

    def _static_planes(self):
        l = self.level
        o = np.zeros((4, self.Hm, self.Wm), np.float32)
        for (r, c) in l.walls:
            o[0, r, c] = 1
        o[1, l.goal[0], l.goal[1]] = 1
        o[3, :l.height, :l.width] = 1
        return o

    def _obs(self):
        l = self.level
        o = np.zeros(self.observation_space.shape, np.float32)
        o[:4] = self._static
        o[2, self.pos[0], self.pos[1]] = 1
        for k in range(self.K):
            tt = self.t + 1 + k
            for h in l.hazards:
                if h.active(tt):
                    for (r, c) in h.cells:
                        o[4 + k, r, c] = 1
        o[-1, :l.height, :l.width] = max(0, l.max_moves - self.t) / l.max_moves
        return o

    def step(self, action):
        l = self.level
        dr, dc = MOVES[ACTIONS[int(action)]]
        nxt = (self.pos[0] + dr, self.pos[1] + dc)
        info = {}
        if not l.passable(nxt):
            return self._obs(), -1.0, True, False, {"reason": "illegal_move"}
        self.t += 1
        self.pos = nxt
        if l.hazard_active(nxt, self.t):
            return self._obs(), -1.0, True, False, {"reason": "hazard"}
        if nxt == l.goal:
            return self._obs(), 1.0, True, False, {"reason": "goal"}
        if self.t >= l.max_moves:
            return self._obs(), 0.0, False, True, {"reason": "out_of_moves"}
        return self._obs(), -self.step_penalty, False, False, info

    def render(self):
        from .render import render_map
        lines = render_map(self.level).splitlines()
        r, c = self.pos
        row = list(lines[r + 1])
        row[3 + c] = "@"
        lines[r + 1] = "".join(row)
        return f"t={self.t}\n" + "\n".join(lines)
