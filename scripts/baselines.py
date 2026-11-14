"""Success rate of simple non-LLM players per difficulty: oracle (exact BFS,
must be 100%), naive (shortest path ignoring hazards), wait-then-run
(waits a random 0..cycle-1 turns, then runs straight), random walker."""
import argparse, glob, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boundgym import Level, generate, naive_moves, simulate, solve

ap = argparse.ArgumentParser()
ap.add_argument("--levels", default=None, help="dir of d*.jsonl (default: generate on the fly)")
ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()

def level_sets():
    if a.levels:
        for p in sorted(glob.glob(os.path.join(a.levels, "d*.jsonl")), key=lambda s: int(os.path.basename(s)[1:-6])):
            yield os.path.basename(p)[:-6], [Level.from_json(l) for l in open(p)]
    else:
        for d in range(1, 11):
            yield f"d{d}", [generate(d, 2_000_000 + i) for i in range(a.n)]

rng = random.Random(0)
print(f"{'set':>4} {'oracle':>7} {'naive':>7} {'wait+run':>9} {'random':>7} {'opt_len':>8}")
for name, levels in level_sets():
    res = {"oracle": 0, "naive": 0, "wait": 0, "random": 0}
    for l in levels:
        res["oracle"] += simulate(l, solve(l)).success
        nm = naive_moves(l)
        res["naive"] += simulate(l, nm).success
        res["wait"] += simulate(l, ["W"] * rng.randrange(l.cycle) + nm).success
        res["random"] += simulate(l, [rng.choice("UDLRW") for _ in range(l.max_moves)]).success
    n = len(levels)
    avg = sum(l.optimal_moves for l in levels) / n
    print(f"{name:>4} {res['oracle']/n:7.0%} {res['naive']/n:7.0%} {res['wait']/n:9.0%} {res['random']/n:7.0%} {avg:8.1f}")
