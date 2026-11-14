# Architecture

Layout, module dependencies, data flow and key contracts of the boundgym repo.

## Directory layout (as it stands)

Restructured 2026-11-08: the shared package, levels, scripts and tests now sit at the project root.
Each `chapterN/` holds only that chapter's README and results.

```
boundgym/                    project root (run everything from here)
├── progress.md              hub: chapter table, current status, index of notes/
├── todo.md                  task list (Now / Next / Later / Done), read by the dashboard
├── notes/                   results, findings, issues, lessons (project-independent), changes, architecture, environment
├── pyproject.toml           package "boundgym", no core deps; [gym]/[dev] extras add gymnasium+numpy
├── .venv/                   kit env (Python 3.14): pip install -e ".[dev]"
├── .vllm-venv/              GPU env (Python 3.12): vLLM 0.33, torch 2.14, TRL 1.17, PEFT 0.21, boundgym (editable)
├── boundgym/                the core package (shared by all chapters)
│   ├── level.py             Level/Hazard data model, time model, JSON round-trip
│   ├── solve.py             static BFS, naive path, exact BFS solver over (cell, t mod cycle)
│   ├── verify.py            simulate(), parse_moves(), reward(): the scorer and the RL reward
│   ├── generate.py          generate(difficulty, seed): procedural levels with guarantees
│   ├── render.py            SYSTEM_PROMPT, render_prompt() ("modulo"), render_prompt_timeline(), PROMPTS
│   ├── viz.py               animated GIF renderer: trace(), render_frames(), side_by_side(), save_gif()
│   ├── env.py               Gymnasium BoundGridEnv (grid + hazard look-ahead planes)
│   └── trl_hooks.py         make_records() + boundgym_reward() for TRL GRPOTrainer
├── scripts/
│   ├── serve.sh             swap the vLLM server to a model, block until it serves that model id
│   ├── eval_openai.py       eval any OpenAI-compatible server (--prompt, --no-think, --timeout, --out)
│   ├── rescore.py           re-score saved results with the current verifier
│   ├── make_demo.py         demo GIFs/MP4s → docs/media/ (real levels + real model attempts)
│   ├── dashboard.py         local status page (stdlib) → http://127.0.0.1:8765; --background/--status/--stop/--restart
│   ├── baselines.py         oracle / naive / wait+run / random
│   └── make_levels.py       freeze eval sets → levels/
├── levels/d1..d8.jsonl      frozen eval sets, 100 levels each (paths default relative to scripts/)
├── tests/test_core.py       18 tests
├── docs/dashboard.md        how to run the dashboard (start/stop/restart, sources)
├── docs/media/              timing.gif, failure_modes.gif (+ .mp4), modes/<outcome>.gif
├── docs/figures/            README figures (make_readme_figures.py) + dashboard.png
├── chapter1/
│   ├── README.md            original kit README (commands assume the old layout, see note at its top)
│   └── results/             root = think 8k, nothink/, think32k/ (1.7B, 4B, 14B-AWQ)
├── chapter2/
│   └── results/             think32k_timeline/ (4B, timeline prompt)
├── chapter3/
│   ├── WRITEUP.md           one-page chapter 3 story (question, 8k vs 16k, results, lessons)
│   ├── train_grpo.py        GRPO (TRL, LoRA, colocated vLLM); --resume, --profile; writes runs/<run>/metrics.jsonl
│   ├── eval_ckpt.sh         serve base + LoRA checkpoint, paired eval → results/<run>_<ckpt>/
│   ├── compare.py           paired bootstrap CI between two result dirs
│   ├── runs/<run>/          args.json, checkpoint-N/, profile.jsonl, metrics.jsonl, completions/, resumes.jsonl
│   └── results/             base/, <run>_checkpoint-N/ (d3, d5; 50 levels x 4 samples, 32k)
└── chapter5/
    └── failure_taxonomy.py  outcome mix + hazard-timing / illegal-move diagnostics over chapter3/results
```

## Module dependencies

```
level  ◄── solve  ◄── verify          (verify imports solve.static_distances for progress)
  ▲          ▲          ▲
  │          └── generate ──┘          (generator calls the solver and verifier to enforce guarantees)
  └── render
generate, render, verify  ──►  trl_hooks   (ch 3)
generate, level, render   ──►  env         (PPO comparison)
```

The core is pure standard-library Python. Only `env.py` needs gymnasium/numpy.

## Data flow: one level, end to end

```
(difficulty, seed) ─► generate ─► Level ─────────────────────────────┐
                        │  uses solve(): solvable? optimal length?   │
                        │  sets max_moves = opt + max(3, opt/3)      │
                        ▼                                            │
                  render_prompt(Level) + SYSTEM_PROMPT               │
                        ▼                                            │
              LLM (vLLM / TRL policy)                                │
                        ▼                                            │
              completion text ─► parse_moves (last <moves> block)    │
                        ▼                                            ▼
                     simulate(Level, moves) ─► Result(success, reason, progress)
                        ├─► eval_openai.py: per-set table + results/*.jsonl
                        └─► reward(): 1.0 / 0.0 (or 0.2·progress if shaped)  → GRPO
```

## Three consumers of the same core

| Consumer | Entry | Input to policy | Score |
|---|---|---|---|
| LLM eval (ch 1–2, 4) | `scripts/eval_openai.py` | `render_prompt` text over HTTP | `simulate` → table |
| LLM RL (ch 3) | `trl_hooks.make_records` + `boundgym_reward` | same prompt text | `reward()` |
| Tiny PPO baseline | `env.BoundGridEnv` | (C,H,W) tensor, step by step | ±1 per episode |

Eval and GRPO use **the same prompt and the same scorer**, so ch 3 gains show up directly on the ch 1 curve.
PPO sees a different observation (step-by-step, with look-ahead), so compare it on solve rate only.

## Key contracts (changing any of these changes every downstream number)

- **Time model:** a move increases t by 1. You die if the new cell's hazard is active at the new t. Hazard state depends only on t % period.
- **Answer format:** last `<moves>…</moves>` block. Only `UDLRW` and separators are allowed inside it.
- **Seed ranges:** training `0..`, frozen eval `1,000,000+`, `baselines.py` on-the-fly `2,000,000+`. These must never overlap.
- **Level JSON schema** (`Level.to_dict`): frozen eval sets depend on it and on the generator being deterministic.
  Any change to `generate.py` changes the levels you get for a given seed, so the frozen `levels/` files are the reference, not the seeds.
- **Prompt text** (`render.py`): eval results are only comparable under the same prompt. A prompt variant (ch 2)
  should be a new function or flag, not an edit to `render_prompt`.

## Planned additions per chapter (proposal, not built yet)

| Ch | New pieces | Builds on |
|---|---|---|
| 2 | prompt variants in `render.py` (e.g. per-step hazard timeline); aggregate/plot script over `results/`; multi-sample pass@k runs | `eval_openai.py` |
| 3 | GRPO training script (TRL + vLLM rollouts), curriculum over d3–d5, checkpoint → `eval_openai.py` | `trl_hooks.py`, `generate` on training seeds |
| 4 | rollout throughput experiments: budget vs solve rate, think length, vLLM settings | `--max-tokens`, `--no-think`, ch 1 token data |
| 5 | activation capture + probes (e.g. does the model track t or t % period?) with an HF model, not vLLM | `Level`, `simulate` for labels |
| 6 | write-up, figures, published eval sets; repo README.md embeds `docs/media/` GIFs (timing + failure modes) | everything |
| — | PPO run (CleanRL `ppo.py` + `BoundGridEnv`) | `env.py` |

~~Open structural question: package under `chapter1/`.~~ Resolved 2026-11-08: moved to the root (see layout).
New chapter-specific scripts go in `chapterN/`. Anything reused across chapters goes in `scripts/` or the package.
