# Chapter 3: Can RL on boundgym teach a small model to plan?

*Written 2026-11-14. Detail and raw tables: [notes/results.md](../notes/results.md), [notes/findings.md](../notes/findings.md).*

## The question

boundgym levels are grid puzzles with hazards that switch on and off on fixed schedules. A plan only works if
every move is right in both space and time. Chapter 1 showed Qwen3-4B (thinking mode, 32k-token budget)
solves 42% of d3 and 18% of d5 held-out levels. **Does GRPO on generated training levels (d2–d4) improve that,
or does it only change surface behaviour?**

Setup for every run: TRL GRPO with LoRA r=32, colocated vLLM, a strict 0/1 reward from the exact verifier, and
4 levels × 8 attempts per step. Evaluation uses the same held-out sets every time (50 levels × 4 attempts, 32k
budget), paired by level with a bootstrap 95% CI.

## What happened

| run | change | outcome |
|---|---|---|
| g1, g2 | 8k training budget, LR 1e-5 | Output ~20% shorter; **no accuracy gain** at 32k |
| g3 | LR 5e-5 | Collapsed: longer, higher-entropy output; stopped at step 50 |
| g4 | **16k training budget**, LR 1e-5 | **Accuracy gain**, replicated (below) |

**The finding: the token budget decides what the reward teaches.** A cut-off attempt scores 0, so
`reward = (1 − share cut off) × solve rate among finished attempts`.

- **At 8k:** ~55% of attempts were cut off, so the cheapest way to raise reward was to finish sooner. In g2,
  all of the reward gain came from finishing more often. Accuracy among finished attempts went *down*
  (0.72 → 0.56).
- **At 16k:** ~12% were cut off, so the reward paid for solving. g4's training reward rose with the cut-off
  share flat. Against g2 on the same levels at the same steps, its solve rate among finished attempts went
  from −0.14 to +0.08.

## Result

Qwen3-4B base vs g4 checkpoint-100, 32k budget, 4 attempts per level, paired by level:

| set | original 50 levels | **fresh 50 levels** | pooled (100) |
|---|---|---|---|
| d3 (in training range) | +12 [+4, +20] | **+14 [+3, +24]** | +12.8 [+6.2, +19.2] |
| d5 (harder than training) | +12 [+4, +20] | **+6 [−2, +14]** (n.s.) | +9.0 [+3.2, +14.5] |

**The fresh levels are the only unbiased test.** g4 emerged as the winner among four checkpoints evaluated on the
original sets (g1, g2@100, g2@200, g4; g3 was not evaluated). That selection inflates the original numbers. The fresh
set was generated afterwards (seeds 2,000,000+) and never used for any decision.

- **d3:** a clear gain that holds on fresh levels (+14, CI above zero).
- **d5:** points the same way but is not significant on fresh levels alone (+6, CI includes zero). The pooled +9
  excludes zero, but it includes the selected set.

**Summary: a clear gain at training-level difficulty (≈ +13 points on d3, 42% → ~54%), with a likely but smaller
transfer to harder levels (d5, ≈ +6 to +9).** Answers got slightly shorter, not longer, so this isn't a
budget effect.

Most of the improvement is fewer moves into walls (the model keeps better track of its position), then fewer
hazard deaths. One systematic error remains: hazard deaths lean toward being one tick early.

## Limits

- One training run per config (no seed replicates); one model size; 100 training steps.
- The yardstick in training was g2, itself training and at a different budget. The eval is what counts.
- A 16k run takes ~14 h of training on one RTX 5090 (~16.5 min/step).

## Lessons

1. **Pick a budget at which most attempts finish**, or the reward measures speed instead of skill. Watch the
   cut-off share next to training reward.
2. **Track "solved among finished" during training**, not reward alone.
3. **With a fixed data order, compare runs at the same step.** All runs saw the same levels in the same order, and
   the difficulty drift made every run look like it declined.
4. **Fix a midpoint stop rule in advance, and check that its comparison is valid.** Ours stopped g3 for the right
   result but the wrong reason.
5. **Raise the LR in one step and watch the first ~20 steps.** 5e-5, normal for LoRA elsewhere, collapsed here.
6. **Replicate the first significant result on data generated after the choice.**

## Next (when resumed)

Screen a per-tick scratchpad by prompting only (the model writes t, position and hazard states each move; ~2 h of
GPU), aimed at the remaining timing and position-tracking errors. Then chapter 4 (rollout speed; the 16k runs are
its motivation).
