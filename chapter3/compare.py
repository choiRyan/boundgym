"""Paired comparison of two eval result dirs (same levels, any number of samples).

  python chapter3/compare.py chapter3/results/base chapter3/results/<tag> [--model SUBSTR]

A dir holding several models' files needs --model to pick one (it refuses to mix them).

Per set: solve rate for each (mean over all samples), difference, and a paired
bootstrap 95% CI over levels (each level's per-sample mean is the unit).
"""
import glob, json, os, random, sys
from collections import defaultdict

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("a_dir"); ap.add_argument("b_dir")
ap.add_argument("--model", default=None, help="only files whose name contains this")
args = ap.parse_args()
a_dir, b_dir = args.a_dir, args.b_dir


def load(d):
    out = {}
    for p in glob.glob(os.path.join(d, "*.jsonl")):
        if args.model and args.model not in os.path.basename(p):
            continue
        name = os.path.basename(p).rsplit("_", 1)[1][:-6]
        if name in out:
            sys.exit(f"{d}: several models have {name}; pass --model")
        per = defaultdict(list)
        toks = []
        for l in open(p):
            r = json.loads(l)
            per[r["idx"]].append(r["success"])
            toks.append(r.get("completion_tokens") or 0)
        out[name] = ({k: sum(v) / len(v) for k, v in per.items()}, sum(toks) / max(len(toks), 1))
    return out


A, B = load(a_dir), load(b_dir)
rng = random.Random(0)
print(f"{'set':>4} {os.path.basename(a_dir.rstrip('/')):>12} {os.path.basename(b_dir.rstrip('/'))[:24]:>24} {'diff':>7} {'95% CI':>16} {'tok A':>7} {'tok B':>7}")
for s in sorted(set(A) & set(B), key=lambda x: int(x[1:])):
    (a, ta), (b, tb) = A[s], B[s]
    idx = sorted(set(a) & set(b))
    diffs = [b[i] - a[i] for i in idx]
    boots = sorted(sum(rng.choice(diffs) for _ in idx) / len(idx) for _ in range(5000))
    lo, hi = boots[125], boots[4874]
    ma, mb = sum(a[i] for i in idx) / len(idx), sum(b[i] for i in idx) / len(idx)
    print(f"{s:>4} {ma:12.0%} {mb:24.0%} {mb - ma:+7.0%} {f'[{lo:+.0%}, {hi:+.0%}]':>16} {ta:7.0f} {tb:7.0f}")
