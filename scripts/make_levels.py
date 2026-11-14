"""Freeze eval level sets: levels/d{d}.jsonl, N levels per difficulty.
Eval seeds start at 1_000_000 so they never overlap with training seeds (0..)."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boundgym import generate

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "levels"))
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--difficulties", default="1,2,3,4,5,6,7,8")
ap.add_argument("--seed-base", type=int, default=1_000_000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
for d in map(int, a.difficulties.split(",")):
    path = os.path.join(a.out, f"d{d}.jsonl")
    with open(path, "w") as f:
        for i in range(a.n):
            f.write(generate(d, a.seed_base + i).to_json() + "\n")
    print("wrote", path)
