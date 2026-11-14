"""Ch 5 step 1: what kind of mistakes does the model make? (Is there one systematic error worth editing out?)

For each eval results file (chapter3/results/<name>/*.jsonl) it reports:
  - outcome counts: goal / hazard / illegal_move / out_of_moves / did_not_reach_goal / bad_format / truncated
  - for hazard deaths, timing diagnostics:
      off_before: the hazard was OFF one tick earlier (t-1). If the model checks hazards at the pre-move time
                  (a classic off-by-one), deaths should be enriched for this.
      off_after:  the hazard was OFF one tick later (t+1), i.e. waiting one more turn would have survived.
      first_wait: the plan contains no W before the death (it never waited).
    Each is compared to a chance baseline for the same hazard: the share of its active phases with the
    previous / next phase inactive. A systematic timing error shows as observed >> baseline.
  - "ignored hazards": the plan's path up to the death equals a static shortest path (walls only).

Usage: .venv/bin/python chapter5/failure_taxonomy.py [results_dir ...]   (default: all of chapter3/results)
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boundgym.level import MOVES, Level  # noqa: E402
from boundgym.solve import static_distances  # noqa: E402
from boundgym.verify import parse_moves, simulate  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_levels(difficulty: int) -> list[Level]:
    with open(os.path.join(ROOT, "levels", f"d{difficulty}.jsonl")) as f:
        return [Level.from_dict(json.loads(l)) for l in f if l.strip()]


def analyse(path: str) -> dict:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        return {}
    levels = load_levels(rows[0]["difficulty"])
    out = Counter()
    hz = Counter()
    for r in rows:
        if r.get("finish_reason") == "length":
            out["truncated"] += 1
            continue
        out[r["reason"]] += 1
        if r["reason"] not in ("hazard", "illegal_move"):
            continue
        lv = levels[r["idx"]]
        moves = parse_moves(r["completion"]) or list(r.get("moves", ""))
        res = simulate(lv, moves)
        if r["reason"] == "illegal_move":
            if res.reason != "illegal_move":
                continue
            t, pos = res.moves_used, res.final_pos
            dr, dc = MOVES[moves[t]]
            nxt = (pos[0] + dr, pos[1] + dc)
            il = hz  # same counter, "il_" keys
            il["il_n"] += 1
            il["il_wall" if lv.in_bounds(nxt) else "il_edge"] += 1
            il["il_first3"] += t < 3
            il["il_prev_was_hazard_cell"] += lv.hazard_at(pos) is not None
            # would the transposed move (U<->L, D<->R: row/col swap) or the opposite move have been legal?
            swap = {"U": "L", "L": "U", "D": "R", "R": "D", "W": "W"}[moves[t]]
            opp = {"U": "D", "D": "U", "L": "R", "R": "L", "W": "W"}[moves[t]]
            for key, m in (("il_swap_legal", swap), ("il_opp_legal", opp)):
                ddr, ddc = MOVES[m]
                il[key] += lv.passable((pos[0] + ddr, pos[1] + ddc))
            continue
        if res.reason != "hazard":  # stored label from an older scorer; skip rather than guess
            hz["relabelled"] += 1
            continue
        t, pos = res.moves_used, res.final_pos
        h = next(h for h in lv.hazards if pos in h.cells and h.active(t))
        hz["n"] += 1
        hz["off_before"] += not lv.hazard_active(pos, t - 1)
        hz["off_after"] += not lv.hazard_active(pos, t + 1)
        on = list(h.on)
        hz["base_before"] += sum(((p - 1) % h.period) not in h.on for p in on) / len(on)
        hz["base_after"] += sum(((p + 1) % h.period) not in h.on for p in on) / len(on)
        hz["no_wait"] += "W" not in moves[:t]
        # did the path so far follow static (wall-only) shortest-path steps toward the goal?
        dist = static_distances(lv)
        p, greedy = lv.start, True
        for m in moves[:t]:
            dr, dc = MOVES[m]
            nxt = (p[0] + dr, p[1] + dc)
            if dist.get(nxt, 1e9) != dist.get(p, 1e9) - 1:
                greedy = False
                break
            p = nxt
        hz["greedy_path"] += greedy
    return {"n": len(rows), "out": out, "hz": hz}


def main():
    dirs = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "chapter3", "results", "*")))
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            a = analyse(f)
            if not a:
                continue
            n, out, hz = a["n"], a["out"], a["hz"]
            name = f"{os.path.basename(d)}/{os.path.basename(f)}"
            print(f"\n{name}  (n={n})")
            print("  outcomes: " + ", ".join(f"{k} {v} ({v / n:.0%})" for k, v in out.most_common()))
            k = hz["n"]
            if k:
                print(f"  hazard deaths analysed: {k}" + (f" (+{hz['relabelled']} relabelled)" if hz["relabelled"] else ""))
                print(f"    hazard was off at t-1: {hz['off_before'] / k:.0%}  (chance {hz['base_before'] / k:.0%})")
                print(f"    hazard was off at t+1: {hz['off_after'] / k:.0%}  (chance {hz['base_after'] / k:.0%})")
                print(f"    never waited before dying: {hz['no_wait'] / k:.0%}")
                print(f"    path so far was a static shortest path (ignored hazards): {hz['greedy_path'] / k:.0%}")
            k = hz["il_n"]
            if k:
                print(f"  illegal moves analysed: {k}: into a wall {hz['il_wall'] / k:.0%}, off the grid {hz['il_edge'] / k:.0%},"
                      f" within the first 3 moves {hz['il_first3'] / k:.0%}, from a hazard cell {hz['il_prev_was_hazard_cell'] / k:.0%}")
                print(f"    row/col-swapped move would be legal {hz['il_swap_legal'] / k:.0%}, opposite move legal {hz['il_opp_legal'] / k:.0%}")


if __name__ == "__main__":
    main()
