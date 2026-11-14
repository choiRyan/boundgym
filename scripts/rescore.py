"""Re-score saved eval results with the current verifier (no model calls).

  python scripts/rescore.py results/think32k/*.jsonl results/nothink/*.jsonl results/*.jsonl

Rewrites each file in place and prints the eval table. Rows whose outcome changes
keep their previous reason in "reason_before_rescore".
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boundgym import Level, parse_moves, simulate

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--levels", default=os.path.join(os.path.dirname(__file__), "..", "levels"))
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

cache = {}
def levels_for(d):
    if d not in cache:
        cache[d] = [Level.from_json(l) for l in open(os.path.join(a.levels, f"d{d}.jsonl"))]
    return cache[d]

print(f"{'file':<40} {'solve':>6} {'before':>7} {'changed':>7} {'truncated':>9} {'bad_fmt':>7} {'hazard':>7} {'illegal':>8}")
for path in a.files:
    rows = [json.loads(l) for l in open(path)]
    before = sum(r["success"] for r in rows)
    changed = 0
    for r in rows:
        lvl = levels_for(r["difficulty"])[r["idx"]]
        assert lvl.seed == r["seed"], f"{path}: level mismatch at idx {r['idx']}"
        moves = parse_moves(r["completion"])
        res = simulate(lvl, moves) if moves is not None else None
        if res:
            reason = res.reason
        elif r["reason"] in ("truncated", "error"):
            reason = r["reason"]
        else:
            reason = "bad_format"
        if reason != r["reason"]:
            changed += 1
            r.setdefault("reason_before_rescore", r["reason"])
        r.update(reason=reason, success=bool(res and res.success),
                 progress=res.progress if res else 0.0, moves="".join(moves or []))
    if not a.dry_run:
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    n = len(rows)
    cnt = lambda k: sum(r["reason"] == k for r in rows) / n
    name = os.path.relpath(path)[-40:]
    print(f"{name:<40} {sum(r['success'] for r in rows)/n:6.0%} {before/n:7.0%} {changed:7d} "
          f"{cnt('truncated'):9.0%} {cnt('bad_format') + cnt('error'):7.0%} {cnt('hazard'):7.0%} {cnt('illegal_move'):8.0%}")
