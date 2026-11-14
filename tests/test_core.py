import glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from boundgym import (Hazard, Level, generate, naive_moves, parse_moves, render_prompt,
                      reward, simulate, solve)


def tiny(max_moves=6):
    # S . A . G   hazard A at (0,2), active when t % 2 == 0
    return Level(5, 1, frozenset(), (0, 0), (0, 4),
                 (Hazard("A", ((0, 2),), 2, (0,)),), max_moves=max_moves)


def test_timing_semantics():
    l = tiny()
    r = simulate(l, list("RRRR"))           # reaches A at t=2 (active) -> dies
    assert not r.success and r.reason == "hazard" and r.moves_used == 2
    assert simulate(l, list("RWRRR")).success   # reaches A at t=3 (safe)
    assert solve(l) is not None and len(solve(l)) == 5


def test_failure_reasons():
    l = tiny()
    assert simulate(l, ["U"]).reason == "illegal_move"
    assert simulate(l, ["R"]).reason == "did_not_reach_goal"
    assert simulate(tiny(max_moves=4), list("RWRRR")).reason == "out_of_moves"
    assert solve(tiny(max_moves=4)) is None


def test_parse_and_reward():
    l = tiny()
    assert parse_moves("think... <moves>R W R R R</moves>") == list("RWRRR")
    assert parse_moves("<moves>bad</moves><moves>R,W,R,R,R</moves>") == list("RWRRR")  # last block wins
    assert parse_moves("<moves>R R and then go left</moves>") is None   # prose inside -> rejected
    assert parse_moves("R W R R R") is None
    # stray opening tag in the reasoning must not swallow the final answer
    assert parse_moves("I'll answer in <moves> format.\n</think>\n<moves>R W R R R</moves>") == list("RWRRR")
    assert reward(l, "<moves>RWRRR</moves>") == 1.0
    assert reward(l, "<moves>RRRR</moves>") == 0.0
    assert 0 < reward(l, "<moves>RRRR</moves>", shaped=True) <= 0.2
    assert reward(l, "no tags", shaped=True) == 0.0


@pytest.mark.parametrize("d", range(1, 11))
def test_generated_levels_are_valid(d):
    for seed in range(15):
        l = generate(d, seed)
        sol = solve(l)
        assert sol is not None and len(sol) == l.optimal_moves <= l.max_moves
        assert simulate(l, sol).success
        if d >= 2:
            assert not simulate(l, naive_moves(l)).success
        assert l.start not in l.walls and l.goal not in l.walls
        assert l.hazard_at(l.start) is None and l.hazard_at(l.goal) is None
        names = {h.name for h in l.hazards}
        assert not names & set("UDLRWSG")


def test_deterministic_and_roundtrip():
    a, b = generate(5, 42), generate(5, 42)
    assert a.to_json() == b.to_json()
    c = Level.from_json(a.to_json())
    assert c.to_json() == a.to_json() and solve(c) == solve(a)
    assert "<moves>" in render_prompt(a)


def test_gym_env_oracle_rollout():
    pytest.importorskip("gymnasium")
    from boundgym.env import ACTIONS, BoundGridEnv
    env = BoundGridEnv(difficulty=(1, 6))
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        assert env.observation_space.contains(obs)
        total, done = 0.0, False
        for m in solve(env.level):
            obs, r, term, trunc, info = env.step(ACTIONS.index(m))
            total += r
            done = term or trunc
        assert done and total == 1.0 and info["reason"] == "goal"


def test_trl_hooks():
    from boundgym.trl_hooks import boundgym_reward, make_records
    recs = make_records([2], 3)
    lv = Level.from_json(recs[0]["level"])
    good = "<moves>" + " ".join(solve(lv)) + "</moves>"
    conv = [[{"role": "assistant", "content": good}], [{"role": "assistant", "content": "nope"}]]
    assert boundgym_reward(conv, [recs[0]["level"]] * 2) == [1.0, 0.0]


def test_timeline_prompt_matches_hazard_rule():
    from boundgym.render import hazard_timeline
    l = generate(4, 7)
    lines = hazard_timeline(l).splitlines()
    assert len(lines) == l.max_moves
    for t, line in enumerate(lines, start=1):
        listed = set(line.split(":", 1)[1].split()) - {"-"}
        assert listed == {h.name for h in l.hazards if h.active(t)}


def test_viz_trace_matches_verifier(tmp_path):
    pytest.importorskip("PIL")
    from boundgym.viz import render_frames, save_gif, trace
    l = generate(3, 5)
    for moves in (solve(l), naive_moves(l), list("UUUUUUUU"), list("W")):
        tr = trace(l, moves)                       # asserts agreement with simulate()
        assert tr.reason == simulate(l, moves).reason
    frames = render_frames(l, trace(l, solve(l)), title="t")
    assert len(frames) == l.optimal_moves + 2      # one per t, plus held final banner
    save_gif(frames, str(tmp_path / "x.gif"))
    assert (tmp_path / "x.gif").stat().st_size > 0


def _brute_optimal(l):
    """Independent of solve(): BFS over the full time-expanded graph (cell, t), no t % cycle dedup."""
    from collections import deque
    from boundgym.level import MOVES
    seen, q = {(l.start, 0)}, deque([(l.start, 0)])
    while q:
        pos, t = q.popleft()
        if t >= l.max_moves:
            continue
        for dr, dc in MOVES.values():
            n = (pos[0] + dr, pos[1] + dc)
            if not l.passable(n) or l.hazard_active(n, t + 1):
                continue
            if n == l.goal:
                return t + 1
            if (n, t + 1) not in seen:
                seen.add((n, t + 1))
                q.append((n, t + 1))
    return None


def _some_static_shortest_path_survives(l):
    """True if ANY hazard-free shortest path (not just naive_moves' tie-break) survives the hazards."""
    from boundgym.level import MOVES
    from boundgym.solve import static_distances
    dist = static_distances(l)
    frontier = {l.start}
    for t in range(1, dist[l.start] + 1):
        frontier = {n for p in frontier for dr, dc in (MOVES[m] for m in "RDLU")
                    for n in [(p[0] + dr, p[1] + dc)]
                    if dist.get(n) == dist[p] - 1 and not l.hazard_active(n, t)}
    return l.goal in frontier


FROZEN = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "levels*", "d*.jsonl")))


@pytest.mark.parametrize("path", FROZEN, ids=lambda p: os.path.relpath(p, os.path.join(os.path.dirname(p), "..")))
def test_frozen_sets_optimal_and_reproducible(path):
    """Every frozen eval level: solvable, optimal_moves confirmed by an independent search,
    identical to the generator's output, and from d3 no hazard-free shortest path survives."""
    with open(path) as f:
        levels = [Level.from_json(line) for line in f]
    assert levels
    for l in levels:
        assert _brute_optimal(l) == l.optimal_moves == len(solve(l))
        assert generate(l.difficulty, l.seed).to_json() == l.to_json()
        if l.difficulty >= 3:
            assert not _some_static_shortest_path_survives(l)


def test_verifier_format_rules():
    """Documented leniencies: they relax format, never correctness."""
    l = tiny()
    assert reward(l, "<moves>r w r r r</moves>") == 1.0             # case-insensitive
    assert simulate(l, list("RWRRR") + list("UUUU")).success         # moves after reaching G are ignored
    assert reward(l, "<think><moves>RWRRR</moves></think> unsure") == 1.0  # last block counts, wherever it is
    assert reward(l, "<moves>RWRRR</moves> then <moves>RRRR</moves>") == 0.0  # ...and only the last
