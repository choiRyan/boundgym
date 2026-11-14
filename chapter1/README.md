> **Layout note (2026-11-08):** the package, `levels/`, `scripts/`, `tests/` and `pyproject.toml` now live at the
> project root (`boundgym/`), not in this folder. Run the commands below from the root. This folder keeps
> chapter 1's README and `results/`. See `../progress.md`.

# boundgym, chapter 1: the course, the verifier, the baseline

Turn-based, procedurally generated versions of Brood War "bound" maps. A unit
walks from `S` to `G` on a grid while hazard bars fire on periodic schedules.
The model has to plan *when* to move, not just *where*.

This is chapter 1 of the storyline: build the env, then evaluate (ch 2), GRPO
train a small LLM on it (ch 3), speed up rollouts (ch 4), look inside (ch 5),
publish (ch 6).

## Setup (Ubuntu, Python 3.10+)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                       # 16 tests, ~2 s
python scripts/baselines.py     # sanity table, see below
```

The core package has no dependencies. `gymnasium` and `numpy` are only needed
for `boundgym.env`.

## What's in here

| File | What it does |
|---|---|
| `boundgym/level.py` | Level data model, JSON round-trip |
| `boundgym/generate.py` | `generate(difficulty 1..10, seed)`, deterministic |
| `boundgym/solve.py` | Exact BFS solver over (row, col, t mod cycle); proves solvability, gives optimal length |
| `boundgym/verify.py` | `parse_moves`, `simulate`, `reward` (the RL reward / eval scorer) |
| `boundgym/render.py` | LLM prompt text |
| `boundgym/env.py` | Gymnasium env (grid obs + hazard lookahead planes) for a tiny from-scratch PPO comparison |
| `boundgym/trl_hooks.py` | Dataset records and reward function shaped for TRL's `GRPOTrainer` (chapter 3) |
| `scripts/make_levels.py` | Freezes eval sets into `levels/d*.jsonl` (seeds >= 1,000,000, never used for training) |
| `scripts/baselines.py` | Oracle / naive / wait-then-run / random success rates |
| `scripts/eval_openai.py` | Evaluates any OpenAI-compatible server (vLLM, SGLang, llama.cpp) on the frozen sets |
| `levels/` | Frozen eval sets, 100 levels each for d1..d8 |

## Rules (exactly what the verifier enforces)

- Moves: `U D L R W` (W = wait). Every move, including W, advances t by 1.
- After a move, dying = standing on a hazard cell active at the new t.
  Hazard `X` is active when `t % period` is in its `on` set.
- Walking into a wall or off the map fails. Exceeding `max_moves` fails.
- Answers must be inside the last `<moves>...</moves>` block; anything other
  than move letters and separators in that block counts as bad format.
- `max_moves` = optimal + max(3, optimal/3), so there is slack but not unlimited.

## Difficulty knob

Grid grows from 6x5 (d1) to 15x14 (d10); more hazard bars, longer periods,
more walls. Guarantees checked at generation time:

- every level is solvable (exact solver),
- d>=2: running the hazard-free shortest path dies,
- d>=3: the optimal route is longer than the hazard-free shortest path
  (you must wait or detour), d>=6: by at least 2 moves.

Baselines on 200 fresh levels per difficulty (measured):

```
 set  oracle   naive  wait+run  random  opt_len
  d1    100%     61%       49%      0%      7.2
  d2    100%      0%       12%      0%      9.2
  d3    100%      0%       16%      0%     10.8
  d4    100%      0%        9%      0%     12.0
  d5    100%      0%        5%      0%     13.5
  d6    100%      0%        0%      0%     16.1
  d8    100%      0%        0%      0%     19.1
 d10    100%      0%        0%      0%     21.7
```

Generation costs 1 to 40 ms per level, so on-the-fly levels for training are free.

## Chapter 1 exit task: get your first LLM baseline curve

```bash
pip install vllm
vllm serve Qwen/Qwen3-4B --max-model-len 16384 --gpu-memory-utilization 0.85
# in another shell
python scripts/eval_openai.py --model Qwen/Qwen3-4B --sets d1,d2,d3,d5,d8 --limit 50
```

Then repeat with Qwen3-1.7B and a quantized 14B-ish model. The table you get
(solve rate, format errors, hazard deaths vs illegal moves, tokens used) is
the "before" picture that chapter 3's GRPO run tries to move. Look at a few
failed completions in `results/*.jsonl`. Whether models fail on timing
arithmetic or on map reading tells you what to change in the prompt or the
difficulty curve.

Things worth trying once that works:
- Let the model think (`/think` vs `/no_think` on Qwen3) and compare
  solve rate against tokens. That trade-off becomes the chapter 4 story.
- A second prompt format (per-step hazard timeline instead of the modulo
  rule) to see how much is reasoning versus reading.
- `BoundGridEnv` + any PPO (CleanRL's `ppo.py` works with small edits) for
  the from-scratch comparison.

## Status

Tested here: unit tests, 100% oracle solve rate on d1..d10, gym env oracle
rollouts, and `eval_openai.py` end to end against a mock server. **Not yet
run against a real model** (this machine has no GPU); that's your first step.
