"""Evaluate any OpenAI-compatible chat endpoint (vLLM, SGLang, llama.cpp server)
on frozen level sets. Standard library only.

  vllm serve Qwen/Qwen3-4B --max-model-len 16384
  python scripts/eval_openai.py --model Qwen/Qwen3-4B --levels levels --sets d1,d3,d5

Writes one JSONL line per attempt to results/<model>_<set>.jsonl and prints a table.
"""
import argparse, concurrent.futures as cf, glob, json, os, sys, time, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boundgym import SYSTEM_PROMPT, Level, parse_moves, simulate
from boundgym.render import PROMPTS

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://localhost:8000/v1")
ap.add_argument("--model", required=True)
ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
ap.add_argument("--levels", default=os.path.join(os.path.dirname(__file__), "..", "levels"))
ap.add_argument("--sets", default=None, help="comma list like d1,d3 (default: all)")
ap.add_argument("--limit", type=int, default=None, help="levels per set")
ap.add_argument("--samples", type=int, default=1, help="attempts per level (pass@k-style stats)")
ap.add_argument("--temperature", type=float, default=0.6)
ap.add_argument("--max-tokens", type=int, default=8192)
ap.add_argument("--concurrency", type=int, default=32)
ap.add_argument("--out", default="results")
ap.add_argument("--timeout", type=float, default=600, help="per-request seconds (raise for long thinking budgets)")
ap.add_argument("--prompt", default="modulo", choices=sorted(PROMPTS), help="prompt variant")
ap.add_argument("--no-think", action="store_true",
                help="disable thinking via chat_template_kwargs enable_thinking=false (Qwen3)")
a = ap.parse_args()


def chat(prompt):
    payload = {"model": a.model, "temperature": a.temperature, "max_tokens": a.max_tokens,
               "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}]}
    if a.no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(a.base_url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {a.api_key}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=a.timeout) as r:
                j = json.load(r)
            ch = j["choices"][0]
            return ch["message"]["content"] or "", j.get("usage", {}), ch.get("finish_reason")
        except Exception as e:  # retry transient errors
            if attempt == 3:
                return f"<error>{e}</error>", {}, "error"
            time.sleep(2 ** attempt)


def run_one(args):
    idx, sample, level = args
    text, usage, finish = chat(PROMPTS[a.prompt](level))
    moves = parse_moves(text)
    r = simulate(level, moves) if moves is not None else None
    # No parseable answer: distinguish running out of tokens / server errors from real format errors.
    reason = r.reason if r else {"length": "truncated", "error": "error"}.get(finish, "bad_format")
    return {"idx": idx, "sample": sample, "seed": level.seed, "difficulty": level.difficulty,
            "success": bool(r and r.success), "reason": reason, "prompt": a.prompt, "finish_reason": finish,
            "progress": r.progress if r else 0.0, "moves": "".join(moves or []),
            "optimal_moves": level.optimal_moves,
            "completion_tokens": usage.get("completion_tokens"), "completion": text}


paths = sorted(glob.glob(os.path.join(a.levels, "d*.jsonl")), key=lambda s: int(os.path.basename(s)[1:-6]))
wanted = set(a.sets.split(",")) if a.sets else None
os.makedirs(a.out, exist_ok=True)
print(f"{'set':>4} {'solve':>6} {'pass@k':>7} {'truncated':>9} {'bad_fmt':>7} {'hazard':>7} {'illegal':>8} {'avg_tok':>8}")
for p in paths:
    name = os.path.basename(p)[:-6]
    if wanted and name not in wanted:
        continue
    levels = [Level.from_json(l) for l in open(p)][: a.limit]
    jobs = [(i, s, l) for i, l in enumerate(levels) for s in range(a.samples)]
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        rows = list(ex.map(run_one, jobs))
    out = os.path.join(a.out, f"{a.model.replace('/', '_')}_{name}.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = len(rows)
    solved_levels = {r["idx"] for r in rows if r["success"]}
    toks = [r["completion_tokens"] for r in rows if r["completion_tokens"]]
    cnt = lambda k: sum(r["reason"] == k for r in rows) / n
    print(f"{name:>4} {sum(r['success'] for r in rows)/n:6.0%} {len(solved_levels)/len(levels):7.0%} "
          f"{cnt('truncated'):9.0%} {cnt('bad_format') + cnt('error'):7.0%} {cnt('hazard'):7.0%} {cnt('illegal_move'):8.0%} "
          f"{(sum(toks)/len(toks) if toks else 0):8.0f}")
