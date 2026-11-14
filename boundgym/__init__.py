"""boundgym: procedurally generated, turn-based timed obstacle courses
(inspired by Brood War "bound" maps) for LLM RL, evals and tiny-policy RL."""
from .level import Hazard, Level, MOVES
from .verify import Result, simulate, parse_moves, reward
from .solve import solve, static_distances, naive_moves
from .generate import generate, params_for
from .render import render_prompt, SYSTEM_PROMPT

__all__ = ["Hazard", "Level", "MOVES", "Result", "simulate", "parse_moves", "reward",
           "solve", "static_distances", "naive_moves", "generate", "params_for",
           "render_prompt", "SYSTEM_PROMPT"]
