# Lessons (project-independent)

Distilled from [issues.md](issues.md) and the chapter 3 results, written to be reused in other projects. Each
lesson names the incident it came from (boundgym, date) so the detail can be looked up.
Add one when an issue teaches something that would bite again somewhere else.

## Processes, servers and the GPU

- **`pkill -f PATTERN` also kills any shell whose command line contains PATTERN**, including the wrapper you
  launched it from. Symptom: exit 144 (SIGTERM) with the real work orphaned or killed. Use a pattern that matches
  the target but not itself, e.g. `pkill -f "bin/vllm serv[e]"` (the regex `[e]` matches `e`; the literal text
  `serv[e]` doesn't), or kill by PID. *(boundgym 2026-11-08 and again 2026-11-11.)*
  The bracket trick fails if the same command line also contains the plain name elsewhere (e.g. kill and restart
  in one command: `pgrep -f "x.p[y]"; ...; python x.py`). **Kill and restart in separate commands.** *(3rd time, 2026-11-11.)*
  It happened a 4th time minutes after writing that down. Rules don't hold; **design it out**: long-lived services write a
  pidfile and offer `--restart` that kills by PID (as `scripts/dashboard.py` now does).
- **Kill patterns must match how the process was actually started.** A path-based pattern
  (`boundgym/.vllm-venv/bin/vllm`) missed a server started with a relative path, so the old server kept the port and
  every request went to the wrong model (all HTTP 404). *(2026-11-08.)*
- **"Server is up" is not "the right server is up".** The readiness check must confirm the identity of what is
  served (the model id in `/v1/models`), not just that the port answers. Before starting, wait for the port to be free.
- **Check who else is on the GPU before a long job** (`nvidia-smi --query-compute-apps=...`). Other services
  on the machine can hold or grab memory mid-run. *(2026-11-09.)*
- **Background samplers must stop on their own.** `(loop) & PID; ...; kill PID` can leave the loop's children
  running. Prefer `while kill -0 $TARGET; do ...; done`. *(2026-11-09.)*
- **Don't edit a shell script while it runs**: bash reads scripts from disk as it goes. Edit after it exits.
- Use `python -u` (or `PYTHONUNBUFFERED=1`) for anything whose log you watch.
- Client timeouts must scale with the work: a 600 s HTTP timeout was fine at 8k tokens and failed at 32k when
  requests queued behind each other. *(2026-11-08.)*

## Long training runs

- **Make checkpoints resumable from day one.** Save optimizer, scheduler and RNG state, not just the model, and
  have a `--resume` that refuses mismatched args. Model-only checkpoints forced a wait for the next checkpoint and
  a cold optimizer when the GPU had to be lent out. The optimizer state for LoRA is small (~0.5 GB at r=32 on 4B).
  *(2026-11-09.)*
- **Verify a resume by the first logged step number** (26/100, not 1/100) before walking away. The progress
  bar's ETA right after a resume is wrong (it averages over skipped steps).
- **Load the policy in bf16 explicitly** when colocating a generation engine: libraries can default to fp32 and
  silently take twice the memory. *(2026-11-08.)*
- **Profile before tuning.** One profiled step showed 59% of time in generation and most of it wasted on
  completions the loss then masked out. Cheaper than guessing. *(2026-11-09.)*
- Record GPU memory peaks per run; at ~97% of VRAM, don't raise memory knobs for a 10 h run.

- **After `git lfs migrate import`, check the working tree, not just the history.** It rewrote the commits
  correctly but left 132-byte pointer files on disk in place of the real files; `git lfs checkout` restored them.
  Checksum every tracked file before a history rewrite and compare after. *(boundgym 2026-11-14.)*

- **Combining timed frame sequences: don't take the max duration per frame.** Padding a finished
  panel with its long final hold made every later frame of the combined GIF slow (0.45 s steps became 2.6 s).
  Give combined frames one step duration and hold only the true last frame. *(boundgym demo media, 2026-11-14.)*

## Evaluation and measurement

- **The scorer is part of the experiment: test its parser on adversarial text.** A lazy regex let a stray
  opening tag in the model's reasoning swallow the real answer, rejecting correct answers (up to 8 pts per set).
  In RL this would punish correct behaviour. *(2026-11-08.)*
- **Report why something failed, not only that it failed.** "Bad format" hid that most failures were
  truncation. Separate `truncated` / `bad_format` / domain-specific failure reasons. *(2026-11-08.)*
- **Single-sample evals at n=50 are ±~10 pts.** Use several samples per item and a *paired* test (bootstrap CI
  over items, or a sign test) for before/after claims. *(ch 2.)*
- **Keep the same sampling settings in training and eval** (here: Qwen3's generation_config temp/top_k/top_p).
- **Replicate a win on fresh data before calling it.** g4's d5 gain was +12 [+4, +20] on the original set and
  +6 [−2, +14] on 50 new levels; d3 held (+12 → +14). The first significant result is the one most likely to be
  inflated by luck. Generate the replication set *after* choosing the checkpoint. *(ch 3, 2026-11-14.)*
- **Re-score saved outputs when the scorer changes**, and keep the old label next to the new one.

## RL on reasoning models

- **The token budget defines what the reward pays for.** With an 8k cap and truncation scored 0, GRPO taught the
  model to be ~20% shorter, not more accurate at 32k. Rising training reward came from fewer cut-offs. Check that
  training reward rises for the reason you intend: watch the clipped ratio next to it. *(ch 3, 2026-11-11.)*
  Confirmed by the fix: the same setup at 16k (cut-offs ~12% instead of ~55%) gave +12 pts at eval on both sets.
  Rule of thumb: pick a budget at which most attempts finish, so the reward measures the skill. *(2026-11-14.)*
- **Check where learning signal is lost before fixing it.** "Sparse reward" was the suspected problem, but 74% of
  GRPO groups were mixed (signal present); only 23% were all-fail. Count all-0 / all-1 / mixed groups from the
  saved completions before reaching for filtering or shaping.
- **Re-check hyperparameters inherited from another regime.** LR 1e-5 is a full-finetune value; LoRA usually needs
  several times more. But "usually" is not "here": at 5e-5 (g3) the reward collapsed within 30 steps (longer,
  higher-entropy output). Scale the LR up in one step and watch the first ~20 steps. *(ch 3, 2026-11-12.)*
- **A fixed data order makes within-run trends lie.** With a seeded shuffle, every run sees the same levels at
  the same step, and the difficulty mix drifts by chance. Early-vs-late in one run then mixes learning with data
  order; it made three runs look like they "declined". Compare runs at matched steps (the shared order is good
  for that), or against a base-model reference on the same data. *(ch 3, 2026-11-12.)*
- **Fix a midpoint stop rule before the data arrives.** Write the metric, windows, threshold and the action for each
  outcome while the run is still early. g3 was stopped at step 50 of 100 by such a rule, saving ~6 h of GPU with
  no argument about it afterwards. But the rule's comparison must itself be valid: g3's compared steps 31–50 with
  1–20 of the same run, which the data order confounds (above). *(ch 3, 2026-11-12.)*

## Working practice

- Keep a progress hub + todo list + issues log from the start, and distil lessons as they happen.
- When a number goes in a summary, write down its protocol (samples, budget, levels) next to it.
- **Read times from the machine; don't infer them.** Twice a run's start time was guessed from conversation flow,
  and both guesses were ~12 h off (09:05 vs 21:47; morning vs 20:57). Stamp logs with `date` at launch and quote that.
