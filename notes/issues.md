# Issues encountered (boundgym)

Dated log of what went wrong in this project and how it was fixed. Generalizable lessons are distilled into [lessons.md](lessons.md).

- **2026-11-09: profiling run blocked by another GPU user.** An unrelated process held ~25 GB of GPU memory, so
  colocated vLLM got 0.43 GB and failed at startup. Training needs the GPU to itself (~26 GB peak). Check
  `nvidia-smi` before launching.
  Also: the `(while …) & MON; …; kill $MON` GPU sampler pattern didn't kill the loop's child, so the profile-run
  sampler kept running until killed by hand. Use a sampler that exits on its own (e.g. stop when the trainer PID is gone).
  Resolved: the other process had exited by the time the run was retried.

- **2026-11-08: GRPO smoke test, two config failures before it ran.** (1) TRL 1.17 rejects `top_p < 1` without
  `top_k` when vLLM importance-sampling correction is on. Set `top_k=20`, which matches the eval: the vLLM log shows
  the server used Qwen3's generation_config (temp 0.6, top_k 20, top_p 0.95). (2) The policy loaded in fp32 by
  default (~16 GB for 4B), so colocated vLLM couldn't get its 0.45 share. Fixed with `model_init_kwargs={"dtype": "bfloat16"}`.
  Harmless noise: `CUDA Error: invalid argument at cumem_allocator.cpp:249` at process exit (vLLM sleep allocator
  teardown, after the save), and `SM 120 requires CUDA >= 12.8` startup probe messages.

- **2026-11-08: `parse_moves` bug rejected correct answers.** The regex `<moves>(.*?)</moves>` matched
  from a stray `<moves>` mentioned in the reasoning ("I'll put it in <moves> format") through to the final
  `</moves>`, so the "block" was the whole thought and was rejected as bad format. 25 attempts were re-labelled
  across all runs, 15 of them flipping to success (0–8 pts per set, e.g. 14B d1 86% → 94%). Found while classifying 1.7B format
  errors. Fixed, regression test added, all results rescored (`scripts/rescore.py`). The original 8k files
  also had their cut-off rows relabelled `truncated` (inferred from 8192 completion tokens).

- **2026-11-08: 1.7B / 14B first attempt was invalid (all 500 requests → HTTP 404).** The driver's
  `pkill -f "boundgym/.vllm-venv/bin/vllm"` didn't match the 4B server (started with relative path
  `.vllm-venv/bin/vllm`), so it kept port 8000. The readiness check only tested that `/v1/models` answered.
  Fix: kill on `vllm serve`, wait for port free, only call ready when `/v1/models` reports the expected model id.
  Bad result files deleted; rerun completed fine.
- **2026-11-08: d8 client timeouts at 32k.** Long completions queued behind each other exceeded the hard-coded
  600 s urllib timeout (4/50). Added `--timeout`; model runs now use 3600.
- **2026-11-08: original 8k eval misread truncation as format errors.** The eval script reported `format_ok`
  without telling truncation apart. Fixed via the `truncated` column (see [changes.md](changes.md)).
- **2026-11-08: moving from the Claude project session to local development.** The chapter 1 kit was written in
  that session; its README came over first and the code later as `boundgym-chapter1.zip`, extracted to `boundgym/chapter1`.
- Minor: `pkill -f "vllm serve"` run from an interactive shell kills that shell too (its own command line matches).
  Harmless, but the exit code 144 looks like a failure.
- Minor: eval table rows don't show in redirected logs until the process exits (Python stdout buffering).
  Use `python -u` if live progress matters.
- **2026-11-09: g2 had to stop mid-run (GPU needed elsewhere) but wasn't resumable.** Checkpoints were
  `save_only_model=True`, with no optimizer/RNG state. Transformers 5.20 still resumes from these: it loads the
  adapter and `trainer_state.json`, skips trained batches, and silently skips the missing optimizer/RNG files.
  So the stop waited for checkpoint-25 (losing nothing), and `--resume` plus full checkpoints were added for
  later runs.
- **2026-11-11: eval wrapper killed by its own pkill (exit 144), again.** `eval_ckpt.sh` starts with
  `pkill -f "vllm serve"`, which also matches any parent shell whose command line contains that text (my wrapper
  had a trailing cleanup `pkill -f "vllm serve"`). The script itself survived as an orphan and kept running; only
  the completion notification was lost. **Fixed** after the eval: `eval_ckpt.sh` and `scripts/serve.sh` now use `"bin/vllm serv[e]"`. That regex still matches the server,
  but no longer matches a command line that contains the pattern itself.
