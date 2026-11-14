# boundgym progress

Hub for where the project stands. Details live in `notes/`; the task list in [todo.md](todo.md).
Live view: `.venv/bin/python scripts/dashboard.py --background`, then http://127.0.0.1:8765. Start / stop / restart: [docs/dashboard.md](docs/dashboard.md).

## The larger project

Storyline from `chapter1/README.md`:

| Ch | Goal | Status |
|---|---|---|
| 1 | Build the env (generator, exact solver, verifier = RL reward), get a first LLM baseline curve | **Exit task done**: env verified, "before" curves for 1.7B / 4B / 14B-AWQ |
| 2 | Evaluate properly (models, prompts, think vs no-think) | **Done**: think/no-think, budget, model size, prompt format; demo media |
| 3 | GRPO-train a small LLM on it (TRL, `boundgym/trl_hooks.py`) | **Done (initial validation)**: g4 (16k budget) beats base at 32k. Fresh levels: d3 +14 [+3, +24], d5 +6 [−2, +14] n.s. Clear gain at training difficulty, smaller transfer to d5. [Write-up](chapter3/WRITEUP.md). 8k runs only shortened output; g3 (LR 5e-5) collapsed. |
| 4 | Speed up rollouts (solve rate vs tokens trade-off) | Not started; data from ch 1 already relevant |
| 5 | Look inside the model | **In progress** (paused): failure taxonomy done (`chapter5/failure_taxonomy.py`); probes not started |
| 6 | Publish | Not started |

## Current status (2026-11-14): paused after initial validation

- **The environment can teach a measurable planning gain.** g4 (Qwen3-4B, LoRA GRPO, 100 steps at a 16k budget)
  beats base at 32k on held-out levels. On fresh levels generated afterwards (the only unbiased test, since g4 was
  selected on the originals): **d3 +14 [+3, +24]** holds; d5 +6 [−2, +14] is n.s. alone. Pooled with the originals
  (100 levels per difficulty): d3 +12.8 [+6.2, +19.2], d5 +9.0 [+3.2, +14.5]. Framing: a clear gain at training-level difficulty, a likely but smaller transfer to harder levels.
  Write-up: [chapter3/WRITEUP.md](chapter3/WRITEUP.md).
- What it took: a training budget at which most attempts finish (16k: ~12% cut off). At 8k (g1, g2) the reward
  paid for finishing and the model only got shorter; a higher LR (g3) collapsed.
- **Stopping point** (by decision 2026-11-14): experiment rounded out. Ch 3 written up. Open next steps in `todo.md`: the per-tick
  scratchpad screen, ch 4–6. GPU is free; the dashboard is stopped.

- History of the ch 3 runs (g1–g4, the step-50 rule and its correction): [notes/results.md](notes/results.md) and
  the write-up.

## Repository

Git repo since 2026-11-14 (branch `main`, remote `choiRyan/boundgym` on GitHub, **private while the repo is under review**). MIT licence. History was restarted from a single commit on 2026-11-14 to drop local-machine details (see `notes/changes.md`). Not tracked: virtualenvs, LoRA checkpoints and raw
training samples (`chapter3/runs/*/{checkpoint-*,final,completions}`, ~13 GB, kept on disk only), and `*.log`.
Training curves live in `runs/<run>/metrics.jsonl`. **Git LFS** holds the eval results
(`chapter*/results*/**/*.jsonl`) and `docs/media` GIFs/MP4s: 56 files, ~127 MB (patterns in `.gitattributes`); the
plain git pack is < 1 MB. Needs `git-lfs` installed to clone. GitHub Free includes 10 GiB LFS storage and 10 GiB/month
bandwidth; keep the LFS budget at $0 so overage is blocked, not billed. A push still uploads the GIFs: hold until ch 6.

## Where things are written down

| file | contents | update when |
|---|---|---|
| [todo.md](todo.md) | task list by chapter (Now / Next / Later / Done); the dashboard reads it | a task starts, finishes or is added |
| [notes/results.md](notes/results.md) | every measured number, with its protocol | a run or eval finishes |
| [notes/findings.md](notes/findings.md) | what the results mean | a result changes the story |
| [notes/issues.md](notes/issues.md) | dated log of things that went wrong here | something breaks |
| [notes/lessons.md](notes/lessons.md) | **project-independent lessons** distilled from the issues; reusable in other projects | an issue teaches something general |
| [notes/changes.md](notes/changes.md) | changes to the kit's code and why | code changes |
| [notes/architecture.md](notes/architecture.md) | layout, modules, data flow, contracts | structure changes |
| [notes/environment.md](notes/environment.md) | hardware, venvs, capacity | setup changes |
| [docs/dashboard.md](docs/dashboard.md) | dashboard: start / stop / restart, data sources, troubleshooting | dashboard changes |
| [notes/demo-media.md](notes/demo-media.md) | the GIFs in `docs/media/` | media regenerated |
