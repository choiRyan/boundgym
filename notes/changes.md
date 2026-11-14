# Changes to the kit

Every change made to the code as delivered in the chapter 1 kit, and why.

- `boundgym/verify.py`: `parse_moves` no longer lets a stray `<moves>` in the reasoning swallow the final
  answer (block body may not contain `<moves>`). Regression test added to `tests/test_core.py`. **This changes
  the reward function.**
- **Restructure (2026-11-08):** `boundgym/`, `levels/`, `scripts/`, `tests/`, `pyproject.toml` moved from
  `chapter1/` to the root; venv rebuilt at root. Default `levels/` paths in scripts are now script-relative.
- `boundgym/render.py`: `render_prompt_timeline()` + `hazard_timeline()` (per-step list of active hazards
  instead of the modulo rule) and `PROMPTS = {"modulo", "timeline"}`. `render_prompt` output verified
  byte-identical to before. Test added.
- `scripts/eval_openai.py --prompt {modulo,timeline}`; rows record `prompt`.
- `boundgym/viz.py` + `scripts/make_demo.py` (new): GIF renderer and demo builder. `pillow` added as the
  `viz` extra (and to `dev`). Test added.
- `chapter3/train_grpo.py`: `--profile` (TRL phase timings → `runs/<run>/profile.jsonl`) and
  `--no-mask-truncated`. `--resume CKPT|latest` (refuses if training args differ from `args.json`; logs to
  `resumes.jsonl`). Checkpoints now save optimizer/scheduler/RNG by default (~0.5 GB extra);
  `--model-only-ckpt` restores the old behaviour. (The prof-* run dirs were deleted after their numbers were recorded here.)
- **Pre-publication cleanup (2026-11-14):** MIT `LICENSE` added (`license` in `pyproject.toml`). Local-machine details
  (other services on the machine, home-directory paths) removed from `notes/issues.md`, `notes/lessons.md` and
  `docs/dashboard.md`. README: name origin, 1- vs 4-attempt baseline note, "How this was built" section, ops tips
  moved out. Git history restarted as one commit so the removed details don't survive in GitHub history.
- `scripts/serve.sh` (new): safe server swap (kills `vllm serve`, waits for port, checks served model id).
- `scripts/rescore.py` (new): re-scores saved `results/*.jsonl` with the current verifier, in place. Changed rows
  keep `reason_before_rescore`.
- `scripts/eval_openai.py`:
  - `truncated` column (from `finish_reason == "length"`), split out of `bad_format`. Server errors count under `bad_fmt`.
    Replaces the old `format_ok` column. Each row now records `finish_reason`.
  - `--no-think` (sends `chat_template_kwargs: {enable_thinking: false}`).
  - `--timeout` (per-request seconds, default 600).
  - `--out` subdirectories used per condition: `results/` (think 8k, original), `results/nothink/`, `results/think32k/`.
- `scripts/dashboard.py` (new, 2026-11-11): stdlib-only local status page. Reads progress.md, todo.md, runs, results
  and nvidia-smi on each request; writes nothing.
- `chapter3/train_grpo.py`: `MetricsLog` callback writes one JSON line per logged step to `runs/<run>/metrics.jsonl`
  (from g4 on; for g1–g3 the dashboard parses the dict lines in `chapter3/<prefix>.log`).
- Docs: `progress.md` split into a hub + `notes/*.md`; `todo.md` and `notes/lessons.md` added (2026-11-11).
- `scripts/dashboard.py`: lifecycle flags `--background`, `--status`, `--stop`, `--restart` (pidfile `.dashboard.pid`,
  log `.dashboard.log`), plus a "What you're looking at" intro with a difficulty table. Documented in `docs/dashboard.md`.
- `scripts/dashboard.py`: third chart per run, "solved, of those that finished" = reward ÷ (1 − cut off), with a
  10-step ratio-of-sums line, a whole-run value in the summary, and an explainer entry. It separates finishing more
  (brevity) from solving better; g2 fell 0.72 → 0.56 while reward rose (2026-11-11).
- Runs: g3 stopped at step 50 (2026-11-12 03:10) by the pre-set rule; g4 = g2 setup at a 16k budget started 03:10
  (`--max-completion 16384`, so `vllm_max_model_length` = 17408).
- `chapter5/failure_taxonomy.py`: failure breakdown over saved evals (ch 5 step 1, 2026-11-12).
- dashboard explainer: shared level order, compare runs at matched steps (2026-11-12).
- dashboard: metrics.jsonl keeps the latest entry per step (a resume re-logs the redone steps) (2026-11-13).
- `chapter3/eval_rep.sh` + `levels_rep/` (d3, d5; 50 levels each, seeds 2,000,000+): replication eval on never-used levels; results in `chapter3/results_rep/` (2026-11-14).
- dashboard: replication table from `chapter3/results_rep/` (2026-11-14).
- Stopping point 2026-11-14: ch 3 initial validation done (g4 replicated); docs updated.
- `chapter3/WRITEUP.md`: one-page ch 3 story (2026-11-14).
- Git repo initialized (2026-11-14): .gitignore covers venvs, checkpoints, completions, logs; g1–g3 metrics.jsonl backfilled from logs.
- Git LFS (2026-11-14): eval results + docs/media moved via `git lfs migrate import --everything` (local history rewritten before any push); all 136 files verified byte-identical.
- README.md skeleton (stubs + points to make, for handwriting) with GIFs, a mermaid workflow diagram and figures from
  `scripts/make_readme_figures.py` (`docs/figures/eval_vs_base.svg`, `training_budget.svg`) plus a dashboard
  snapshot (`docs/figures/dashboard.png`, LFS). progress.md status trimmed to current; dashboard explainer no longer
  says the budget is 8k (2026-11-14).
- Demo media: even frame timing (viz.evenly_timed; side_by_side/stack no longer take the max duration), WebM + MP4 for every animation, *.webm in LFS (2026-11-14).
