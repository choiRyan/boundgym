# boundgym todo

Format (the dashboard parses it): sections `## Now`, `## Next`, `## Later`, `## Done`; one task per line as
`- [ ] (chN) text` or `- [x] (chN) text`. Move items between sections as they progress; keep Done as the record.

## Now

## Next
- [ ] (ch4) Screen a per-tick scratchpad (model writes t, position, hazard states each move) by prompting base 4B at 32k, no training (~2 h GPU); targets both the timing off-by-one and position-tracking failures

## Later
- [ ] (ch4) If the scratchpad helps: GRPO with it on 4B, at whatever budget keeps cut-offs low
- [ ] (ch4) Step-time levers: FP8 KV cache (TRL patch), Liger fused loss for micro-batch > 1; rent an H100 / 2-GPU server mode only for long confirm runs
- [ ] (ch3) Shaped reward run (`--shaped`) if the strict reward stays flat
- [ ] (ch4) Rollout speed: solve rate vs tokens trade-off (profile data from g1/g2 is a start)
- [ ] (ch5) Look inside the model: linear probes for position and tick t during the model's reasoning (where does tracking break?)
- [ ] (ch5) Moonshot: weight edit (direction removal / steering) for the one-tick-early hazard timing error, then the 32k paired eval
- [ ] (ch6) Publish: README with `docs/media` GIFs (don't upload media anywhere before this)

## Done
- [x] (ch3) Write-up: `chapter3/WRITEUP.md`
- [x] (ch3) Replication on fresh levels: d3 +14 [+3, +24], d5 +6 [−2, +14]; pooled d3 +12.8 [+6.2, +19.2], d5 +9.0 [+3.2, +14.5]
- [x] (ch3) g4 eval at 32k: d3 54% (+12 [+4, +20]), d5 30% (+12 [+4, +20]) vs base, first significant gain
- [x] (ch3) g4 training: 100 steps at 16k, 2 pauses, gap to g2 −0.14 → +0.08
- [x] (ch3) g4 step-50 check (level-matched vs g2): gap closed −0.14 → −0.04, continue
- [x] (ch5) Failure taxonomy (`chapter5/failure_taxonomy.py`): no single dominant error; timing off-by-one is partial
- [x] (ch3) g3 (LR 5e-5): stopped at step 50 by the pre-set rule; reward collapsed 0.25 → 0.06 (LR too high)
- [x] (ch1) Environment verified: generator, exact solver, verifier, 18 tests
- [x] (ch1) "Before" curves: Qwen3 1.7B / 4B / 14B-AWQ, think, 32k
- [x] (ch1) Non-LLM baselines: oracle, naive, wait+run
- [x] (ch2) Think vs no-think, 8k vs 32k budget (4B)
- [x] (ch2) Prompt format: modulo vs timeline (n.s.; modulo stays the reference)
- [x] (ch2) Fix `parse_moves` stray-tag bug; rescore all results
- [x] (ch2) Demo media: timing.gif, failure_modes.gif, per-outcome GIFs
- [x] (ch3) GRPO pipeline: train_grpo.py, eval_ckpt.sh, compare.py; smoke test
- [x] (ch3) g1: d2–d4, 8k, masked truncation, 100 steps; eval n.s.
- [x] (ch4) Profile a training step (59% generation); vLLM memory share 0.55
- [x] (ch3) g2: unmasked truncation, 4 prompts/step, 100 steps; eval n.s., ~18% shorter
- [x] (ch3) Resume support (`--resume`, full checkpoints); g2 paused and resumed twice
- [x] (ch3) g2 extended to 200 steps; eval n.s.; group check (74% mixed)
- [x] (docs) Split progress log into `notes/`, add lessons.md, todo.md, dashboard
