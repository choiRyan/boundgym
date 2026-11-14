# Results

All measured numbers, newest chapter last. Each table says its protocol (samples, budget, levels).

All: Qwen3 family, 50 frozen levels per set, 1 sample, temperature 0.6. Rescored with fixed parser.

## Model size (think, 32k budget): the ch 1 "before" curve

| set | 1.7B | 4B | 14B-AWQ |
|---|---|---|---|
| d1 | 68% (6.8k tok) | 90% (6.1k) | 94% (4.7k) |
| d2 | 34% (9.4k) | 78% (7.5k) | 80% (6.4k) |
| d3 | 14% (13.6k) | 44% (10.6k) | 68% (8.5k) |
| d5 | 2% (16.7k) | 22% (12.5k) | 36% (10.2k) |
| d8 | 0% (23.7k) | 4%* (15.1k) | 10% (14.2k) |

Failure mix (think, 32k):
- 1.7B: truncation grows with difficulty (2% → 58% at d8) plus real format misses (8–24%, mostly answer in a code block without `<moves>`).
- 4B / 14B: ~0 truncation or format issues below d8. Failures are hazard deaths (4B 42%, 14B 32% at d5; ~50% at d8 for both) and illegal moves.

\* 4/50 at d8 were 600 s client timeouts (fixed since: `--timeout`). True d8 may be 1–2 pts higher.

## Thinking mode and budget (Qwen3-4B)

| set | no-think, 8k | think, 8k | think, 32k |
|---|---|---|---|
| d1 | 14% (1.5k tok) | 88% (5.8k) | 90% (6.1k) |
| d2 | 6% (1.5k) | 62% (6.8k) | 78% (7.5k) |
| d3 | 2% (1.7k) | 18% (7.7k) | 44% (10.6k) |
| d5 | 0% (2.0k) | 4% (8.1k) | 22% (12.5k) |
| d8 | 0% (1.5k) | 0% (8.2k) | 4% (15.1k) |

Think-8k truncation: 6% d1, 30% d2, 60% d3, 84% d5, 100% d8. No-think also 0% at d4, d6, d7.

## Prompt format: modulo rule vs per-step timeline (Qwen3-4B, think 32k), ch 2

| set | modulo | timeline | paired flips (timeline-only / modulo-only) |
|---|---|---|---|
| d1 | 90% (6.1k tok) | 92% (5.8k) | 4 / 3 |
| d2 | 78% (7.5k) | 84% (7.4k) | 7 / 4 |
| d3 | 44% (10.6k) | 54% (9.7k) | 14 / 9 |
| d5 | 22% (12.5k) | 20% (15.0k) | 7 / 8 |
| d8 | 4% (15.1k) | 4% (19.4k) | 2 / 2 |

Pooled over all sets: 34 vs 26 flips, sign test p ≈ 0.37, so **not significant**.
At d5/d8 timeline costs more tokens and truncates more (10% / 18%). At d8, hazard deaths drop (25 → 14),
but illegal moves (13 → 18) and truncation take their place.

## Ch 3 GRPO smoke test (2026-11-08)

Qwen3-4B + LoRA r=32 (all linear), lr 1e-5, DAPO loss, beta 0, truncated completions masked, 2 prompts x 8
generations per step, d2–d4 training seeds, 8k budget, sampling temp 0.6 / top_p 0.95 / top_k 20 (same as eval).

| step | reward | groups with no signal | cut off at 8k | mean length | step time |
|---|---|---|---|---|---|
| 1 | 0.31 | 0% | 62% | 7.6k | 4.5 min |
| 2 | 0.13 | 50% | 38% | 7.4k | 4.6 min |
| 3 | 0.31 | 50% | 50% | 7.4k | 4.4 min |

Peak GPU 25.6 / 32 GB (vLLM at 0.45, sleep mode). About 435 tokens/s end to end, so 100 steps ≈ 8 h.
About half the rollout compute goes to cut-off completions that get masked: the main ch 4 target.

## Ch 3 run g1: training curve (2026-11-08/09)

Same config as the smoke test, 100 steps, 7.4 h (266 s/step). 10-step windows:

| steps | reward | groups with no signal | cut off at 8k | entropy |
|---|---|---|---|---|
| 1–10 | 0.32 | 40% | 59% | 0.074 |
| 11–20 | 0.23 | 50% | 71% | 0.051 |
| 21–30 | 0.20 | 65% | 74% | 0.051 |
| 31–40 | 0.24 | 40% | 58% | 0.079 |
| 41–50 | 0.23 | 40% | 67% | 0.060 |
| 51–60 | 0.23 | 40% | 58% | 0.075 |
| 61–70 | 0.19 | 35% | 61% | 0.070 |
| 71–80 | 0.42 | 25% | 41% | 0.108 |
| 81–90 | 0.37 | 20% | 51% | 0.086 |
| 91–100 | 0.24 | 25% | 66% | 0.062 |

No clear trend. With 2 prompts/step, the windowed reward mostly reflects which levels were drawn.
Truncation stays ~60%, so most rollout compute is masked out. Only the paired eval decides whether g1 helped.

## Ch 3 run g1: paired eval (d3 + d5, 50 levels x 4 samples, 32k)

| set | base | g1 ckpt-100 | diff | 95% CI (paired bootstrap) | pass@4 base → g1 | tokens base → g1 |
|---|---|---|---|---|---|---|
| d3 | 42% | 48% | +6% | [−2%, +15%] | 80% → 86% | 10.9k → 10.4k |
| d5 | 18% | 20% | +2% | [−5%, +8%] | 44% → 62% | 14.3k → 13.8k |

**Not significant.** The direction is positive and successful completions are ~8% shorter, but both CIs include 0.
The base model's 4-sample numbers (42% / 18%) agree with the earlier single-sample run (44% / 22%), so the protocol is consistent.
The d5 pass@4 jump (44 → 62%) is suggestive but has no paired test, so don't lean on it.

## Ch 3 run g2: unmasked truncation, 4 prompts/step (2026-11-09 to 11)

Config: like g1 (d2–d4, 8k, strict 0/1 reward, LR 1e-5, LoRA r=32), but vLLM 0.55, 4 prompts × 8 per step, and
**cut-off completions kept in the loss as reward 0** (`--no-mask-truncated`). Peak GPU 31.0 GB at vLLM 0.55, near
the limit, so do not raise it. ~6.8–7 min/step. Trained to step 100, then extended to 200 by resuming.

**Interruptions (resume support added for this):** paused at step 25 (2026-11-09 18:32, GPU lent out). That
checkpoint was model-only (the old default), so resuming restarted Adam's moments and the 5-step warmup; data order
and step count carried over. One aborted resume (19:00–19:07, before step 26 finished, nothing lost). Resumed
~23:00, confirmed at 26/100. The step-100 checkpoint was a full save, so extending to 200
(`--max-steps 200 --resume checkpoint-100`, from 2026-11-10 21:47) was exact; confirmed at 101/200.
tqdm's ETA right after a resume is wrong because it averages over the skipped steps.

Training curve, 20-step windows (reward / clipped ratio / mean length / zero-variance groups):

| steps | reward | clipped | mean len | zero-std |
|---|---|---|---|---|
| 1–20 | 0.32 | 0.56 | 7438 | 0.29 |
| 21–40 | 0.29 | 0.55 | 7492 | 0.30 |
| 41–60 | 0.36 | 0.45 | 7240 | 0.29 |
| 61–80 | 0.30 | 0.47 | 7278 | 0.24 |
| 81–100 | 0.32 | 0.45 | 7084 | 0.26 |
| 101–120 | 0.37 | 0.40 | 7044 | 0.29 |
| 121–140 | 0.33 | 0.42 | 7118 | 0.28 |
| 141–160 | 0.37 | 0.42 | 7027 | 0.24 |
| 161–180 | 0.38 | 0.38 | 6855 | 0.21 |
| 181–200 | 0.38 | 0.32 | 6683 | 0.25 |

Training reward rises ~0.32 → ~0.37 over steps 101–200 while clipping falls ~0.56 → 0.32. g1 had 38% zero-std
groups and 0.27 mean reward; g2 (1–100) had 28% and 0.32.

## Ch 3 run g2: paired evals (d3 + d5, 50 levels × 4 samples, 32k)

| set | base | g1@100 | g2@100 | g2@200 | g2@100 − base | g2@200 − base | g2@200 − g2@100 |
|---|---|---|---|---|---|---|---|
| d3 | 42% | 48% | 42% | 38% | +0 [−10, +10] | −4 [−13, +4] | −4 [−14, +6] |
| d5 | 18% | 20% | 21% | 18% | +2 [−4, +10] | −1 [−9, +8] | −3 [−10, +4] |

**Checkpoint vs checkpoint (computed 2026-11-11 by the dashboard's pairwise table):** g2@200 − g1@100 on d3 =
**−10 [−19, −2]**, the only comparison so far whose CI excludes 0. g2@100 − g1@100: d3 −6 [−16, +3], d5 +1 [−6, +9].
g2@200 − g1@100 d5: −2 [−10, +6]. With 12 comparisons in play, one borderline hit is expected by chance, so this is
a lead (g2's brevity pressure may cost some d3 accuracy) rather than an established result. Re-test with more levels if it matters.

Mean tokens (d3 / d5): base 10933 / 14270, g1 10412 / 13844, g2@100 8839 / 11902, g2@200 8744 / 11322.
pass@4 base → g2@100 → g2@200: d3 78 → 78 → 84%, d5 56 → 56 → 46% (mixed, noise-level). g2@100 vs g1:
d3 −6 [−16, +3], d5 +1 [−6, +8]. Evals took ~1.3 h instead of ~3 h because completions are shorter.

**Reading:** the training-reward rise came from fewer 8k cut-offs (a truncated completion always scores 0), not
from better solving. At 32k the base model rarely truncates anyway, so the gain shows up only as ~20% fewer tokens.

**Group check** (`runs/g2-…/completions/*.parquet`, 800 groups of 8): 74% mixed (these give signal), 23% all-fail,
3% all-pass; about the same in steps 1–100 and 101–200. A lack of learning signal is not the bottleneck, and filtering
to "sometimes solved" levels could recover at most ~25% of groups.

## Ch 3 run g3: LR 5e-5, stopped at step 50 (2026-11-11/12)

Config: g2 with LR 5e-5 (was 1e-5). 8k budget, 4 levels × 8 tries per step, cut-offs scored 0, from base.
Started 2026-11-11 20:57, ~7 min/step, stopped 2026-11-12 03:10 after checkpoint-50 (full, resumable) by the
step-50 rule fixed beforehand (stop if solved-of-finished over steps 31–50 is > 0.05 below steps 1–20).

| steps | reward | cut off | solved of finished | mean length | entropy |
|---|---|---|---|---|---|
| 1–10 | 0.31 | 64% | 0.86 | 7656 | 0.171 |
| 11–20 | 0.19 | 72% | 0.68 | 7863 | 0.196 |
| 21–30 | 0.09 | 87% | 0.69 | 8088 | 0.211 |
| 31–40 | 0.06 | 92% | 0.69 | 8112 | 0.215 |
| 41–50 | 0.06 | 89% | 0.57 | 8131 | 0.222 |

**Correction (2026-11-12 11:45): the rule was confounded by level order.** All runs train on the *same levels
in the same order* (`ds.shuffle(seed=0)` and trainer seed 0; verified by hashing prompts: g2 = g3 = g4 at every
step), and the mix varies (steps 21–27: 18% d2 vs 38% in 1–10). Level-matched against g2 (LR 1e-5, near base
early on), g3's solved-of-finished barely moved: +0.06 (steps 1–20) → −0.04 (31–50). What *is* real is the
collapse: on the same levels, g3's reward was 0.30 below g2's over steps 31–50 (cut off 90% vs 46%). So stopping
g3 was right, but for the collapse, not for the rule's metric.

| steps | g2 reward / cut / solved-of-finished | g3 | g4 (16k) |
|---|---|---|---|
| 1–10 | 0.34 / 58% / 0.81 | 0.31 / 64% / 0.86 | 0.55 / 12% / 0.63 |
| 11–20 | 0.29 / 54% / 0.64 | 0.19 / 72% / 0.68 | 0.45 / 16% / 0.54 |
| 21–27 | 0.20 / 63% / 0.54 | 0.08 / 87% / 0.62 | 0.37 / 17% / 0.44 |
| 31–40 | 0.36 / 48% / 0.69 | 0.06 / 92% / 0.69 | |
| 41–50 | 0.35 / 45% / 0.64 | 0.06 / 89% / 0.57 | |

Original rule reading: solved-of-finished 0.78 (1–20) vs 0.62 (31–50) → stop. Reward collapsed rather than drifting, so
by the rule g4 uses LR 1e-5. Reading: at 5e-5 the policy moved toward longer, higher-entropy output, which at an
8k cap means more cut-offs and less reward. Not evaluated at 32k (a collapsed policy; checkpoint-50 is kept if
needed).

## Ch 3 run g4: 16k budget (started 2026-11-12 03:10)

Config: g2 (LR 1e-5, 4 levels × 8 tries, cut-offs scored 0, from base) with `--max-completion 16384`.
Steps 1–2: reward 0.59 / 0.72, cut off 12.5% / 9.4%, mean length 9669 / 8641, step time 1022 / 971 s.
Peak GPU memory 28.0 GB (g3 at 8k: 29.6 GB), so a 16k training pass fits.
Paused 2026-11-12 11:34 at step 27 (GPU lent out); resume from full checkpoint-25. Its fall over steps 1–27
(solved-of-finished 0.63 → 0.54 → 0.44) tracks g2's on the same levels (0.81 → 0.64 → 0.54): level order, not
evidence against g4. Resumed 2026-11-13 00:45 from checkpoint-25 (first logged step 26, verified); paused again
10:56 at step 59 (GPU lent out). checkpoint-50 is the latest; steps 51–59 will be redone.

Step-50 check, level-matched against g2 (same levels at every step; g2 is 8k, so only the *change* in the gap
means anything):

| steps | g2 reward / cut / solved-of-finished | g4 | g4 − g2 (solved-of-finished) |
|---|---|---|---|
| 1–10 | 0.34 / 58% / 0.81 | 0.55 / 12% / 0.63 | −0.18 |
| 11–20 | 0.29 / 54% / 0.64 | 0.45 / 16% / 0.54 | −0.10 |
| 21–30 | 0.22 / 62% / 0.59 | 0.44 / 11% / 0.50 | −0.09 |
| 31–40 | 0.36 / 48% / 0.69 | 0.56 / 13% / 0.64 | −0.05 |
| 41–50 | 0.35 / 45% / 0.64 | 0.53 / 13% / 0.61 | −0.03 |
| 51–59 | 0.34 / 47% / 0.65 | 0.55 / 12% / 0.62 | −0.03 |

Gap −0.14 (steps 1–20) → −0.04 (31–50): g4 gains ~0.10 on g2 on the same levels, about twice the noise of a
20-step difference. Its cut-off share is flat at 11–16%, so g4's finished pool isn't shifting much. First run with
a training-side sign of better solving rather than finishing more. Caveats: g2 is a moving, different-budget
reference; the 32k eval of checkpoint-50 is the real check. Decision: continue g4.

Resumed 2026-11-13 19:37 from checkpoint-50 (first step 51, verified; peak GPU 29.2 GB). **Finished 2026-11-14
~09:45**, 100/100 steps, ~14.3 h of training in total. A `CUDA Error: invalid argument at cumem_allocator.cpp`
appears at process exit (vLLM sleep-mode allocator teardown), after `final/` was saved: harmless.

Whole run, 20-step windows, level-matched vs g2 (reward / cut off / solved-of-finished / mean tokens):

| steps | g2 (8k) | g4 (16k) | gap (solved-of-finished) |
|---|---|---|---|
| 1–20 | 0.32 / 56% / 0.72 / 7438 | 0.50 / 14% / 0.58 / 9998 | −0.14 |
| 21–40 | 0.29 / 55% / 0.65 / 7492 | 0.50 / 12% / 0.57 / 9801 | −0.08 |
| 41–60 | 0.36 / 45% / 0.65 / 7240 | 0.52 / 15% / 0.61 / 9766 | −0.04 |
| 61–80 | 0.30 / 47% / 0.57 / 7278 | 0.57 / 12% / 0.65 / 9597 | +0.08 |
| 81–100 | 0.32 / 45% / 0.58 / 7084 | 0.58 / 11% / 0.65 / 9530 | +0.08 |

The gap swung +0.22 over the run. g4's reward rose 0.50 → 0.58 with the cut-off share flat (14% → 11%) and
length down only ~5%: unlike g2, the gain is in solving, not finishing.

## Ch 3 run g4: paired eval (d3 + d5, 50 levels × 4 samples, 32k, 2026-11-14 09:56–11:33)

| vs | d3 | d5 |
|---|---|---|
| base (42% / 18%) | **54%, +12 [+4, +20]** | **30%, +12 [+4, +20]** |
| g1@100 | +6 [−2, +13] | +10 [+2, +19] |
| g2@100 | +12 [+4, +20] | +10 [+1, +18] |
| g2@200 | +16 [+7, +25] | +12 [+3, +22] |

Mean tokens 10,506 (d3) / 13,166 (d5) vs base 10,933 / 14,270: 4–8% shorter, so not a brevity effect.
pass@4: 88% / 66%. **First checkpoint significantly better than base, on both sets**, including d5, which is
harder than anything it trained on (d2–d4). Both lower bounds are +4 and the two sets agree, so this survives the
multiple-comparisons caveat that applied to the earlier lone d3 hit; still, a replication on fresh held-out
levels would make it solid.

Where the gain came from (`chapter5/failure_taxonomy.py`, counts out of 200):

| set | outcome | base | g4@100 |
|---|---|---|---|
| d3 | goal | 83 | 107 |
| d3 | illegal move (wall) | 31 | 15 |
| d3 | hazard death | 49 | 41 |
| d3 | stopped short | 35 | 35 |
| d5 | goal | 37 | 61 |
| d5 | illegal move (wall) | 53 | 43 |
| d5 | hazard death | 61 | 57 |
| d5 | stopped short | 39 | 32 |

Mostly fewer wall bumps (position tracking), then fewer hazard deaths. The one-tick-early timing lean remains
(d3: off at t−1 in 95% of hazard deaths vs 80% chance), so it's still an open target (scratchpad / ch 5).
(Also: at 16k, solved-of-finished is lower than at 8k on the same levels, the selection
effect: harder attempts now finish.)

## Ch 5 (early): what kind of mistakes? (failure taxonomy, 2026-11-12)

Script: `chapter5/failure_taxonomy.py` over the saved 32k evals (50 levels × 4 samples per set). Purpose: decide
whether a weight edit (direction removal / steering, cf. "weight-space programming") has a single systematic
error to target. Base Qwen3-4B:

| set | goal | hazard death | illegal move | stopped short | cut off |
|---|---|---|---|---|---|
| d3 | 42% | 24% | 16% | 18% | 1% |
| d5 | 18% | 30% | 26% | 20% | 5% |

- **Illegal moves are ~95% walking into a wall, mid-plan** (only 3–6% within the first 3 moves for base). So the
  model loses track of where it is while executing its own plan, rather than misreading the start. (A check
  "would the row/col-swapped move have been legal" gave 80–93%, but walls are sparse, so that has no useful
  chance baseline; not evidence of axis confusion.)
- **Hazard deaths lean toward one-tick-early timing.** The hazard was off one tick before the death in 77–93% of
  deaths vs 63–81% expected by chance for those hazards, higher than chance in all 8 sets (base d5: 82% vs 70%;
  base d3: 82% vs 78%). Read as a partial off-by-one (checking the hazard at the pre-move time, or miscounting
  ticks), explaining very roughly 20–40% of hazard deaths. 40–70 deaths per set, so each figure is ±~6 pts, and
  the sets share levels, so the 8 are not independent.
- Training (g1, g2) didn't change the outcome mix much. One shift: among hazard deaths, base had never waited
  in 72–76%, g2@200 in only 41–49%. So training taught it to use W more, but not when to.

Verdict: **no single dominant error.** Failures split across hazard timing, position tracking and plans that stop
short. The timing off-by-one is the only candidate with a specific mechanism; a weight edit aimed at it could at
best fix part of ~25–35% of attempts. Position tracking is the "missing ability" case where the weight-space post
itself reports no gain.

## Ch 3 run g4: replication on fresh levels (2026-11-14 11:59–15:35)

`chapter3/eval_rep.sh` on `levels_rep/` (50 new d3 and d5 levels, seeds 2,000,000+, generated after g4 was
chosen; never used for training, selection or any earlier eval). Same protocol: 4 samples, 32k, paired.

| set | base | g4@100 | diff [95% CI] | pass@4 base → g4 | mean tokens base → g4 |
|---|---|---|---|---|---|
| d3 | 40% | 53% | **+14 [+3, +24]** | 80% → 88% | 11,969 → 10,783 |
| d5 | 23% | 29% | +6 [−2, +14] n.s. | 58% → 70% | 13,228 → 13,323 |

Pooled with the original sets (100 levels per difficulty, all paired by level):
**d3 +12.8 [+6.2, +19.2], d5 +9.0 [+3.2, +14.5].**

Reading: d3 replicates almost exactly (+12 → +14). The d5 gain is smaller on fresh levels (+12 → +6), so the
original +12 was partly luck; the pooled +9 is the better estimate, and it still excludes 0. Base scored about the
same on the fresh sets (40% / 23% vs 42% / 18%), so the new levels are comparable.

## Ch 4 (early): where a g1 training step goes (profile, 2026-11-09)

g1 config (vLLM 0.45, 2 prompts x 8, 8k budget), 3 steps, ~255 s/step:

| phase | time | share |
|---|---|---|
| vLLM generation (16 x ~7.4k tokens) | ~150 s | ~59% |
| backward + optimizer + vLLM sleep/wake (not timed by TRL; remainder) | ~66 s | ~26% |
| log-prob pass for importance sampling | ~19 s | ~8% |
| loss forward passes | ~19 s | ~7% |
| weight sync, reward | <1 s | 0% |

vLLM at 0.45 = 7.56 GB weights + 6.06 GB KV = **44k tokens of KV, max ~4.8 concurrent sequences** at 9.2k.
So generation runs in waves of ~5 at ~790 tok/s. GPU util reads 97% but the work is memory-bound decoding.

Second profile, vLLM 0.60 + 4 prompts/step (32 completions): KV 75k tokens (~8 at once), generation
~1,065 tok/s (+35%), ~446 s/step = **~14 s/completion vs ~16 (−12%)**, but **peak GPU 31.4 / 31.4 GB**, no headroom.
Conclusion: more vLLM memory is a modest gain capped by the card. The bigger waste is that 50–65% of rollouts
hit 8k and g1 masked them, so most generation taught nothing.

## Ch 4 (early): where a g4 step goes (16k budget, profile over 59 steps, 2026-11-13)

~990 s/step. vLLM generation 449 s (~45%); per-token logprob passes 216 s; loss/backward 109 s (+ remaining
`_prepare_inputs` overhead). So at 16k the training side (32 micro-batches of up to 17k tokens, batch size 1,
gradient checkpointing) is ~40% of the step, not a footnote: the "59% generation" figure was g1 at 8k.
Levers on that side: fused/chunked loss (Liger) to allow micro-batches > 1. On the generation side: FP8 KV cache
(needs a TRL patch; sleep mode is already on, peak 28 of 31.8 GB leaves little room to raise the vLLM share).

## Non-LLM baselines (frozen sets)

oracle 100% everywhere; naive 63% d1, 0% after; wait+run 57% d1, 22% d2, 18% d3, 5% d5, 2% d8; random 0%.
