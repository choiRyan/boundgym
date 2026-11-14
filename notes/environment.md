# Environment

Hardware, venvs and capacity numbers.

- RTX 5090 (32 GB, sm_120). Kit venv: `chapter1/.venv` (Python 3.14). vLLM 0.33 + torch 2.14 cu132 in
  `boundgym/.vllm-venv` (Python 3.12; kept separate because vLLM wheels may not support 3.14).
- 4B @ 40k context, 0.85 util → ~17.7 GiB KV (~120k tokens), so 32k-token completions run only a few at a time.
  The 4B 32k pass on d3/d5/d8 took 73 min.
