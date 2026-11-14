# Findings

What the results mean, and the implications drawn from them.

- **The 8k budget was the main thing holding back the first run.** At think-8k, truncation was 60% at d3 and
  100% at d8. Median *successful* completion for 4B: ~9k tok (d3), ~10k (d5), ~17k (d8).
- **With budget removed, failures are real timing errors.** Hazard deaths dominate at d5+ for both 4B and
  14B. Scale helps (14B: 68% at d3 vs 44%) but doesn't fix timing at d8 (10%).
- **Bigger is also cheaper per answer.** 14B uses fewer tokens than 4B at every difficulty and solves more.
  1.7B uses the most and still runs out of budget. Relevant to ch 4: tokens spent mostly reflect struggling, not care.
- **Thinking is essential.** No-think fails at map reading (walls / off-map ~50%) and can't time hazards.
- **1.7B has a format problem on top of the skill gap.** It often gives the right kind of answer outside the
  tags. If 1.7B is a ch 3 candidate, expect early GRPO signal to be mostly format learning.
- **Spelling out the schedule barely helps (ch 2).** The timeline prompt gives +2 / +6 / +10 pts at d1–d3, but
  that's within noise (p ≈ 0.37), and nothing changes at d5/d8. Reading the modulo rule isn't the bottleneck.
  Planning a route against the schedule is. Keep the modulo prompt as the reference.
- **Single-sample evals are noisy.** At d3, 23 of 50 levels flipped between the two runs, in both directions.
  Per-set differences under ~10 pts at n=50 and 1 sample shouldn't be trusted. For ch 3 before/after
  comparisons, use `--samples 4` or more, or more levels, and a paired test.
- Even the 4B clearly beats wait-then-run (78% vs 22% at d2), so it's really reasoning about timing.

- **GRPO at a 16k budget improves solving (g4, 2026-11-14), and it replicates.** Checkpoint-100 vs base at 32k,
  pooled over the original and a fresh set (100 levels per difficulty): **d3 +12.8 [+6.2, +19.2], d5 +9.0
  [+3.2, +14.5]**. On the fresh levels alone (the only unbiased test: the originals are where g4 was selected) d3
  replicated (+14 [+3, +24]) and d5 shrank (+6 [−2, +14], n.s.). Framing: a clear gain at training-level
  difficulty, a likely but smaller transfer to harder levels. Write-up: [chapter3/WRITEUP.md](../chapter3/WRITEUP.md). Same setup as g2 except the budget, so
  the 8k budget was the blocker (hypothesis b): at 8k ~55% of attempts were cut off and the reward paid for
  finishing; at 16k ~12% were, and the reward paid for solving. Gains came mainly from fewer wall bumps, then fewer
  hazard deaths; d5 (never trained on) improved as much as d3.
- **GRPO at an 8k budget teaches brevity, not planning (g1, g2@100, g2@200).** No checkpoint is significantly
  better than base at 32k. Unmasking truncation cuts tokens ~20% at equal accuracy. Training reward is a
  misleading progress signal while the clipped ratio is falling.
  Possible cost: g2@200 is 10 pts below g1 on d3 (CI [−19, −2]), borderline given the number of comparisons.
  The trend g1 48% → g2@100 42% → g2@200 38% on d3 is consistent with brevity pressure hurting accuracy.
  Training-side view of the same thing: reward = (1 − cut off) × solved-of-finished. In g2 the second factor fell
  0.72 → 0.56 (20-step windows) while reward rose 0.32 → 0.38, so all of the gain was finishing more often.
  For a run that learns planning, solved-of-finished should rise *relative to a reference on the same levels*
  (all runs share the level order; a run's own early steps are not a valid baseline). That is the success criterion for g3 onward,
  with the 32k eval as the selection-free check.
- **A higher LoRA LR (5e-5) destabilizes rather than teaches (g3).** On the same levels as g2, reward 0.30 lower
  over steps 31–50 as output got longer and higher-entropy (cut off 90% vs 46%). So the 8k runs weren't simply under-powered by LR 1e-5; the
  budget (hypothesis b) is now the lead, tested by g4 at 16k.

- **The model's failures have no single dominant cause (ch 5 taxonomy, 2026-11-12).** Base d5: 30% hazard deaths,
  26% walking into walls mid-plan (lost track of position), 20% plans that stop short. Hazard deaths lean
  one-tick-early (off at t−1 in 82% vs 70% by chance), a partial off-by-one. Implication for weight editing:
  there's one narrow target (timing), not a removable obstacle like the looping in the weight-space post.

## Implications for ch 3 (GRPO)
- **4B looks like the right target:** clean format, d3–d5 in the useful band (44% / 22%), and 14B shows the
  headroom (68% / 36%) is reachable.
- 10k+ token completions make rollouts expensive; an 8k cap would mostly reward thinking faster, not timing better.
- Cheaper lever to try first: prompt format (per-step hazard timeline instead of the modulo rule).
- The reward depends on `parse_moves`. The stray-tag bug (see [issues.md](issues.md)) would have punished correct answers during
  training, which is a reason to keep the parser covered by tests.
