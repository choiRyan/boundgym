# Demo media (`docs/media/`, regenerate with `python scripts/make_demo.py`)

- `timing.gif`: one d3 level side by side. The naive shortest path dies to hazard C at t=3. The exact solver
  waits once and walks through the A and B bars between their pulses (11 moves, optimal).
  Shows that timing, not pathfinding, is the skill.
- `failure_modes.gif`: real Qwen3-4B attempts on d3 (think 32k), one per outcome: success (12 moves vs
  optimal 11), hazard death (B at t=2), illegal move (into a wall at t=12), stopped short (plan ends one
  cell from G), plus an 8k-budget attempt cut off mid-thought. Individual GIFs in `modes/`.
- Frame anatomy: map (active hazards = red + "!", idle = pale), blue unit + trail, `W×n` badges where it
  waited, move tape on top (current move highlighted), hazard schedule strip below (▼ = current t), and an
  outcome banner (icon + label + status color). Colors from the dataviz skill's reference palette.
- Ch 6 plan: embed `timing.gif` and `failure_modes.gif` in the published repo's README.md. Not uploaded anywhere yet (decided 2026-11-08).
- **Timing (fixed 2026-11-14):** every frame advances at 0.45 s; only the last frame of each animation holds
  (2.6 s). Before, combined animations took the max per-panel duration, so a finished panel's hold slowed every
  later step of the others (timing.gif ran 25 s with 2.6 s steps after t=3). Now 8.0 s / 8.4 s.
- **Formats:** GIF (for the README: GitHub renders GIFs from the repo inline, but doesn't play repo-hosted
  video in a README; video needs an uploaded attachment), plus MP4 (H.264) and WebM (VP9) of every animation,
  same timing, for docs/slides or a ch 6 upload.
- These come straight from the verifier (`trace()` asserts agreement with `simulate()`), so
  they are faithful illustrations, not mock-ups.
