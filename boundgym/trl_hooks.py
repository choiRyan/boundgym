"""Hooks for chapter 3 (GRPO with Hugging Face TRL). Pure Python, no TRL import.

    from datasets import Dataset
    from boundgym.trl_hooks import make_records, boundgym_reward
    ds = Dataset.from_list(make_records(difficulties=[1, 2, 3], n_per=2000))
    trainer = GRPOTrainer(model=..., reward_funcs=[boundgym_reward], train_dataset=ds, ...)

TRL passes extra dataset columns (here "level") to reward functions as kwargs.
Training seeds start at 0; frozen eval sets use seeds >= 1_000_000.
"""
from __future__ import annotations

from .generate import generate
from .level import Level
from .render import SYSTEM_PROMPT, render_prompt
from .verify import reward


def make_records(difficulties, n_per: int, seed_start: int = 0) -> list[dict]:
    out = []
    for d in difficulties:
        for i in range(n_per):
            lvl = generate(d, seed_start + i)
            out.append({
                "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                           {"role": "user", "content": render_prompt(lvl)}],
                "level": lvl.to_json(),
                "difficulty": d,
            })
    return out


def _text(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[-1], dict):  # conversational format
        return c[-1].get("content", "")
    return str(c)


def boundgym_reward(completions, level, shaped: bool = False, **kwargs) -> list[float]:
    return [reward(Level.from_json(lv), _text(c), shaped=shaped) for c, lv in zip(completions, level)]


def boundgym_reward_shaped(completions, level, **kwargs) -> list[float]:
    return boundgym_reward(completions, level, shaped=True)
