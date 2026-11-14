"""Local project dashboard: chapter pipeline, todo list, training runs, eval results, GPU.

  .venv/bin/python scripts/dashboard.py --background   # start detached → http://127.0.0.1:8765
  .venv/bin/python scripts/dashboard.py --status | --stop | --restart [--background]
  .venv/bin/python scripts/dashboard.py                # run in the foreground (Ctrl+C stops it)
  .venv/bin/python scripts/dashboard.py --once out.html   # write one snapshot and exit
Full instructions: docs/dashboard.md

Standard library only. Everything is read fresh on each request from:
  progress.md (chapter table, current status), todo.md, chapter3/runs/*, chapter3/results/*, nvidia-smi.
Nothing here writes to the project.
"""
import argparse, glob, html, json, os, random, re, subprocess, time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUNS = os.path.join(ROOT, "chapter3", "runs")
RESULTS = os.path.join(ROOT, "chapter3", "results")
RESULTS_REP = os.path.join(ROOT, "chapter3", "results_rep")  # fresh levels_rep/; never pair with RESULTS


# ---------------------------------------------------------------- data

def read(path, default=""):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return default


def chapters():
    """Rows of the chapter table in progress.md: (num, goal, status_text, state)."""
    out = []
    for m in re.finditer(r"(?m)^\| (\d+) \| (.+?) \| (.+?) \|$", read(os.path.join(ROOT, "progress.md"))):
        n, goal, st = m.groups()
        low = st.lower()
        state = "done" if ("done" in low.split(":")[0] and "in progress" not in low) else \
                "active" if "in progress" in low else "todo"
        out.append((int(n), goal, st, state))
    return out


def status_section():
    s = read(os.path.join(ROOT, "progress.md"))
    m = re.search(r"(?ms)^## (Current status[^\n]*)\n(.*?)(?=^## |\Z)", s)
    return (m.group(1), m.group(2).strip()) if m else ("Current status", "")


def todos():
    sec, out = None, defaultdict(list)
    for line in read(os.path.join(ROOT, "todo.md")).splitlines():
        if line.startswith("## "):
            sec = line[3:].strip()
        m = re.match(r"\s*- \[( |x)\] (?:\((\w+)\) )?(.*)", line)
        if m and sec:
            out[sec].append({"done": m.group(1) == "x", "tag": m.group(2) or "", "text": m.group(3)})
    return out


def running_cmdlines():
    cmds = []
    for p in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            c = open(p, "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        exe = os.path.basename(c.split(" ", 1)[0])
        if not exe.startswith(("python", "vllm")):  # skip wrapper shells whose command text mentions these
            continue
        if "train_grpo.py" in c or "vllm serve" in c or "eval_openai.py" in c:
            cmds.append(c)
    return cmds


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def run_metrics(name):
    """Per-step logs: metrics.jsonl if the run wrote one, else the dict lines of chapter3/<prefix>.log."""
    mj = os.path.join(RUNS, name, "metrics.jsonl")
    if os.path.exists(mj):
        rows = [json.loads(l) for l in read(mj).splitlines() if l.strip()]
        # a resume redoes the steps after the checkpoint and logs them again: keep the latest entry per step
        last = {r["step"]: r for r in rows if "reward" in r}
        return [last[k] for k in sorted(last)]
    log = os.path.join(ROOT, "chapter3", name.split("-")[0] + ".log")
    rows = []
    for m in re.finditer(r"\{'loss': [^}]*\}", read(log)):
        d = {k: _num(v) for k, v in re.findall(r"'([^']+)': '([^']*)'", m.group(0))}
        if d.get("reward") is not None:
            d["step"] = len(rows) + 1
            rows.append(d)
    return rows


def runs(cmds):
    out = []
    for d in sorted(glob.glob(os.path.join(RUNS, "*")), key=os.path.getmtime, reverse=True):
        name = os.path.basename(d)
        args = json.loads(read(os.path.join(d, "args.json"), "{}") or "{}")
        res = [json.loads(l) for l in read(os.path.join(d, "resumes.jsonl")).splitlines() if l.strip()]
        total = (res[-1]["args"] if res else args).get("max_steps")
        ck = sorted((int(x.rsplit("-", 1)[1]) for x in os.listdir(d) if x.startswith("checkpoint-")))
        met = run_metrics(name)
        step = max([len(met)] + ck[-1:])
        live = any(f"--run {name} " in c + " " for c in cmds if "train_grpo.py" in c)
        # pace from profile.jsonl wall clock: seconds between the first records of recent steps
        first = {}
        for l in read(os.path.join(d, "profile.jsonl")).splitlines():
            try:
                r = json.loads(l)
            except ValueError:
                continue
            first.setdefault(r["step"], r["wall"])
        ks = sorted(first)[-8:]
        pace = (first[ks[-1]] - first[ks[0]]) / (ks[-1] - ks[0]) if len(ks) > 2 else None
        if live:
            state = "running"
        elif total and step >= total and os.path.isdir(os.path.join(d, "final")):
            state = "finished"
        else:
            state = "stopped"
        out.append({"name": name, "args": args, "total": total, "step": step, "ckpts": ck, "state": state,
                    "pace": pace, "eta": (total - step) * pace if (live and pace and total) else None,
                    "reward": [(r["step"], r["reward"]) for r in met],
                    "clipped": [(r["step"], r.get("completions/clipped_ratio")) for r in met
                                if r.get("completions/clipped_ratio") is not None],
                    "resumes": len(res)})
        out[-1].update(solved_finished(out[-1]["reward"], out[-1]["clipped"]))
    return out


def evals(root=RESULTS):
    """Per result dir and set: solve rate over all samples, levels, samples/level, mean tokens; diff vs base."""
    data = {}
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        sets = {}
        for p in glob.glob(os.path.join(d, "*.jsonl")):
            s = os.path.basename(p).rsplit("_", 1)[1][:-6]
            per, toks = defaultdict(list), []
            for l in read(p).splitlines():
                r = json.loads(l)
                per[r["idx"]].append(r["success"])
                toks.append(r.get("completion_tokens") or 0)
            sets[s] = {"per": {k: sum(v) / len(v) for k, v in per.items()},
                       "samples": max(len(v) for v in per.values()), "tok": sum(toks) / max(len(toks), 1)}
        data[os.path.basename(d)] = sets
    rng = random.Random(0)

    def paired(a, b):
        """b − a over shared levels: mean and 95% paired bootstrap CI (levels are the unit)."""
        idx = sorted(set(a) & set(b))
        diffs = [b[i] - a[i] for i in idx]
        boots = sorted(sum(rng.choice(diffs) for _ in idx) / len(idx) for _ in range(2000))
        return {"diff": sum(diffs) / len(idx), "lo": boots[50], "hi": boots[1949]}

    base = data.get("base", {})
    rows = []
    for name, sets in data.items():
        for s, v in sorted(sets.items(), key=lambda x: int(x[0][1:])):
            rate = sum(v["per"].values()) / len(v["per"])
            row = {"run": name, "set": s, "rate": rate, "n": len(v["per"]), "samples": v["samples"], "tok": v["tok"]}
            if name != "base" and s in base:
                row.update(paired(base[s]["per"], v["per"]))
            rows.append(row)
    # every pair of trained checkpoints (later name minus earlier name), per shared set
    names = [n for n in data if n != "base"]
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for s in sorted(set(data[a]) & set(data[b]), key=lambda x: int(x[1:])):
                pairs.append({"a": a, "b": b, "set": s, **paired(data[a][s]["per"], data[b][s]["per"])})
    return rows, pairs


def difficulty_rows():
    """Per frozen eval set d1..d8: grid size, hazards, schedule periods, rules, median optimal moves."""
    import importlib, statistics, sys
    sys.path.insert(0, ROOT)
    params_for = importlib.import_module("boundgym.generate").params_for
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "levels", "d*.jsonl")), key=lambda x: int(os.path.basename(x)[1:-6])):
        d = int(os.path.basename(p)[1:-6])
        lv = [json.loads(l) for l in read(p).splitlines() if l.strip()]
        q = params_for(d)
        rows.append({"d": d, "w": q.width, "h": q.height, "haz": q.n_hazards, "pmax": q.max_period,
                     "naive_dies": q.naive_must_die, "must_wait": q.must_wait,
                     "opt": statistics.median(x["optimal_moves"] for x in lv), "n": len(lv)})
    return rows


def gpu():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        name, used, tot, util = [x.strip() for x in q.strip().splitlines()[0].split(",")]
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        big = []
        for l in apps.strip().splitlines():
            pid, pname, mem = [x.strip() for x in l.split(",", 2)]
            if float(mem) >= 1024:
                big.append((pid, os.path.basename(pname.split()[0]), float(mem)))
        return {"name": name, "used": float(used), "total": float(tot), "util": float(util), "apps": big}
    except Exception as e:  # no GPU / driver hiccup: show it, don't crash the page
        return {"error": str(e)}


# ---------------------------------------------------------------- rendering

E = html.escape


def md_inline(t):
    t = E(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<span class=lnk title='\2'>\1</span>", t)
    return t


def md_bullets(text):
    items, cur = [], None
    for line in text.splitlines():
        if line.startswith("- "):
            cur = [line[2:]]
            items.append(cur)
        elif cur is not None and line.strip():
            cur.append(line.strip())
    return "<ul>" + "".join(f"<li>{md_inline(' '.join(i))}</li>" for i in items) + "</ul>"


STATE = {  # status chips always pair icon + label with the colour
    "done": ("✓", "done", "good"), "active": ("●", "in progress", "info"), "todo": ("○", "not started", "muted"),
    "running": ("▶", "running", "good"), "stopped": ("⏸", "stopped", "warning"), "finished": ("✓", "finished", "info"),
}


def chip(state):
    icon, label, cls = STATE[state]
    return f"<span class='chip {cls}'><span aria-hidden=true>{icon}</span> {label}</span>"


def hms(sec):
    if sec is None:
        return "–"
    h, m = divmod(int(sec) // 60, 60)
    return f"{h} h {m:02d} m" if h else f"{m} m"


def solved_finished(reward, clipped):
    """Solve rate among completions that finished inside the budget = reward / (1 - cut off).
    Exact because a cut-off completion always scores 0. Per step it is undefined when everything was cut off,
    so the rolling value is a ratio of sums over the last 10 steps (not a mean of noisy per-step ratios)."""
    c = dict(clipped)
    rows = [(x, r, c[x]) for x, r in reward if x in c and r is not None]
    pts, roll = [], {}
    for i, (x, r, k) in enumerate(rows):
        pts.append((x, r / (1 - k) if k < 1 else None))
        win = rows[max(0, i - 9):i + 1]
        fin = sum(1 - w[2] for w in win)
        roll[x] = sum(w[1] for w in win) / fin if fin > 0 else None
    fin = sum(1 - k for _, _, k in rows)
    return {"solvedfin": pts, "solvedfin_roll": roll,
            "solvedfin_all": sum(r for _, r, _ in rows) / fin if fin > 0 else None}


def line_chart(series, title, total, fmt="{:.2f}", ymax=1.0, w=520, h=150, roll_map=None):
    """Single-series line chart, steps on x, 0..ymax on y; per-point hover via <title>.
    roll_map (step -> value) overrides the default 10-step rolling mean of the plotted values."""
    pad_l, pad_r, pad_t, pad_b = 34, 10, 10, 22
    pts = [(x, y) for x, y in series if y is not None]
    if not pts:
        return f"<div class=empty>{E(title)}: no data yet</div>"
    xmax = max(total or 0, pts[-1][0], 1)
    X = lambda x: pad_l + (w - pad_l - pad_r) * (x - 1) / max(xmax - 1, 1)
    Y = lambda y: pad_t + (h - pad_t - pad_b) * (1 - y / ymax)
    grid = "".join(f"<line class=grid x1={pad_l} x2={w - pad_r} y1={Y(v):.1f} y2={Y(v):.1f} />"
                   f"<text class=tick x={pad_l - 6} y={Y(v) + 4:.1f} text-anchor=end>{v:g}</text>"
                   for v in (0, ymax / 2, ymax))
    xt = "".join(f"<text class=tick x={X(v):.1f} y={h - 6} text-anchor=middle>{v}</text>"
                 for v in sorted({1, xmax // 2, xmax}))
    # 10-step rolling mean as the line; raw points as faint dots (raw per-step reward is very noisy)
    ys = [y for _, y in pts]
    roll = [sum(ys[max(0, i - 9):i + 1]) / len(ys[max(0, i - 9):i + 1]) for i in range(len(ys))]
    if roll_map is not None:
        keep = [i for i, (x, _) in enumerate(pts) if roll_map.get(x) is not None]
        pts, roll = [pts[i] for i in keep], [roll_map[pts[i][0]] for i in keep]
        if not pts:
            return f"<div class=empty>{E(title)}: no data yet</div>"
    path = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(r):.1f}" for i, ((x, _), r) in enumerate(zip(pts, roll)))
    dots = "".join(f"<circle class=raw cx={X(x):.1f} cy={Y(y):.1f} r=1.6 />" for x, y in pts)
    hits = "".join(f"<circle class=hit cx={X(x):.1f} cy={Y(y):.1f} r=6><title>step {x}: {fmt.format(y)}"
                   f" (10-step mean {fmt.format(r)})</title></circle>" for (x, y), r in zip(pts, roll))
    last = (f"<text class=lab x={X(pts[-1][0]) + 4:.1f} y={Y(roll[-1]) - 6:.1f}>"
            f"{fmt.format(roll[-1])}</text>") if pts[-1][0] < xmax * 0.9 else ""
    return (f"<figure><figcaption>{E(title)} <span class=muted>(line: 10-step mean · dots: per step)</span>"
            f"</figcaption><svg viewBox='0 0 {w} {h}' role=img aria-label='{E(title)}'>{grid}{xt}"
            f"<path class=series d='{path}' />{dots}{last}{hits}</svg></figure>")


def page():
    cmds = running_cmdlines()
    ch, (st_title, st_body), td, rs, (ev, pairs), g = chapters(), status_section(), todos(), runs(cmds), evals(), gpu()
    rep_ev, _ = evals(RESULTS_REP)

    try:
        drows = difficulty_rows()
    except Exception as e:
        drows, derr = [], repr(e)
    else:
        derr = ""
    used = {"2": "train", "3": "train · eval", "4": "train", "5": "eval"}
    dtr = "".join(
        f"<tr><td><b>d{r['d']}</b></td><td class=num>{r['w']}×{r['h']}</td><td class=num>{r['haz']}</td>"
        f"<td class=num>2–{r['pmax']}</td><td class=num>{r['opt']:g}</td>"
        f"<td>{'running straight dies' if r['naive_dies'] else 'straight run can work'}"
        f"{'; must wait or detour' if r['must_wait'] else ''}</td>"
        f"<td>{used.get(str(r['d']), '')}</td></tr>" for r in drows)
    glossary = f"""<div class='card glossary'>
<div class=gl-grid><div>
<p><b>boundgym</b> is a puzzle environment for testing whether a language model can plan against time. A unit walks
from <b>S</b> to <b>G</b> on a grid. Hazard bars (letters) switch on and off on fixed repeating schedules, and the unit
dies if it stands on an active one. Each move (up, down, left, right, wait) takes one time step. The model reads the map
and schedules as text and answers with a move list. An exact solver proves every level solvable, and a verifier
scores the answer (1 = reached the goal alive within the move limit, else 0). That score is also the training reward.</p>
<p><b>Difficulty d1–d8.</b> One knob controls grid size, number of hazards and how long their cycles are. Each eval set
holds 100 frozen levels; training uses separately generated levels of the same difficulty.</p>
<table class=dtab><thead><tr><th>set</th><th class=num>grid</th><th class=num>hazards</th><th class=num>cycle length</th>
<th class=num>shortest solution (median moves)</th><th>what it requires</th><th>chapter 3 use</th></tr></thead><tbody>{dtr}</tbody></table>
{'<div class=muted>difficulty table unavailable: ' + E(derr) + '</div>' if derr else ''}
<dl>
<dt>base</dt><dd>Qwen3-4B with no training: the reference everything is compared against.</dd>
<dt>run names</dt><dd><code>g2-4b-d234-8k-nomask</code> = GRPO run 2, Qwen3-<b>4B</b>, trained on <b>d2–d4</b>,
<b>8k</b>-token answer limit in training, <b>nomask</b> = cut-off answers count as failures. <code>lr5e-5</code> = learning rate.</dd>
<dt>checkpoint-N / @N</dt><dd>The model's weights saved after N training steps. One step = 4 levels × 8 tries (g1: 2 levels × 8).</dd>
<dt>tokens</dt><dd>Length of the model's written reasoning + answer. More tokens means slower and costlier answers.</dd>
</dl></div>
<figure class=demo><img src='/media/timing.gif' alt='Animated example level: a naive straight run dies on a hazard; the solver waits and survives'>
<figcaption class=muted>Same level, two plans: running straight (left) dies on a hazard; the solver's plan (right)
waits for it to switch off.</figcaption></figure></div></div>"""

    # chapter pipeline with per-chapter task progress from todo.md
    per = defaultdict(lambda: [0, 0])
    for sec, items in td.items():
        for it in items:
            if it["tag"].startswith("ch"):
                per[it["tag"]][1] += 1
                per[it["tag"]][0] += it["done"]
    cards = ""
    for n, goal, st, state in ch:
        d, t = per[f"ch{n}"]
        bar = (f"<div class=bar title='{d} of {t} tasks done'><span style='width:{100 * d / t:.0f}%'></span></div>"
               f"<div class=muted>{d}/{t} tasks</div>") if t else "<div class=muted>no tasks listed</div>"
        cards += (f"<div class='card ch {state}'><div class=chn>Ch {n}</div><div class=goal>{md_inline(goal)}</div>"
                  f"{chip(state)}{bar}<details><summary>status</summary><p>{md_inline(st)}</p></details></div>")

    # todo
    tcols = ""
    for sec in ("Now", "Next", "Later", "Done"):
        items = td.get(sec, [])
        lis = "".join(f"<li class={'x' if it['done'] else 'o'}><span aria-hidden=true>{'☑' if it['done'] else '☐'}</span>"
                      f" <span class=tag>{E(it['tag'])}</span> {md_inline(it['text'])}</li>" for it in items)
        tcols += f"<div class='card todo {sec.lower()}'><h3>{sec} <span class=muted>{len(items)}</span></h3><ul>{lis}</ul></div>"

    # runs
    rcards = ""
    for r in rs:
        if r["name"] in ("smoke",) or r["name"].startswith("prof"):
            continue
        a = r["args"]
        frac = r["step"] / r["total"] if r["total"] else 0
        cfg = (f"LR {a.get('lr')}, {a.get('prompts_per_step')}×{a.get('num_generations')} per step, "
               f"budget {a.get('max_completion')}, d{a.get('difficulties')}, "
               f"{'cut-offs scored 0' if a.get('no_mask_truncated') else 'cut-offs masked'}")
        eta = f" · ETA {hms(r['eta'])} (~{time.strftime('%a %H:%M', time.localtime(time.time() + r['eta']))})" if r["eta"] else ""
        mr = [y for _, y in r["reward"]]
        mc = [y for _, y in r["clipped"]]
        summ = (f" · whole run: reward <b>{sum(mr) / len(mr):.2f}</b>, cut off <b>{sum(mc) / len(mc):.0%}</b>"
                + (f", solved of finished <b>{r['solvedfin_all']:.2f}</b>" if r["solvedfin_all"] is not None else "")
                if mr and mc else "")
        rcards += (f"<div class='card run'><div class=runhead><b>{E(r['name'])}</b> {chip(r['state'])}</div>"
                   f"<div class=muted>{E(cfg)}</div>"
                   f"<div class='bar big' title='{r['step']} of {r['total']} steps'><span style='width:{100 * frac:.1f}%'></span></div>"
                   f"<div>step <b>{r['step']}</b> / {r['total']} · {hms(r['pace'])}/step{eta}"
                   f" · checkpoints {', '.join(map(str, r['ckpts'])) or 'none yet'}"
                   f"{' · resumed ' + str(r['resumes']) + '×' if r['resumes'] else ''}{summ}</div>"
                   f"<div class=charts>{line_chart(r['reward'], 'training reward', r['total'])}"
                   f"{line_chart(r['clipped'], 'share cut off at budget', r['total'], '{:.0%}')}"
                   f"{line_chart(r['solvedfin'], 'solved, of those that finished', r['total'], roll_map=r['solvedfin_roll'])}"
                   f"</div></div>")

    # evals
    def dcell(e):
        sig = e["lo"] > 0 or e["hi"] < 0
        return (f"{round(e['diff'] * 100) + 0:+d} <span class=muted>[{round(e['lo'] * 100) + 0:+d}, "
                f"{round(e['hi'] * 100) + 0:+d}]</span> {'<b>significant</b>' if sig else '<span class=muted>n.s.</span>'}")

    def eval_rows(rows):
        out = ""
        for e in rows:
            d = dcell(e) if "diff" in e else "<span class=muted>reference</span>"
            out += (f"<tr><td>{E(e['run'])}</td><td>{e['set']}</td><td class=num>{e['rate']:.0%}</td>"
                    f"<td class=num>{d}</td><td class=num>{e['tok']:,.0f}</td><td class=num>{e['n']}×{e['samples']}</td></tr>")
        return out

    trs = eval_rows(ev)
    rep_trs = eval_rows(rep_ev)

    ptrs = "".join(f"<tr><td>{E(p['b'])}</td><td>{E(p['a'])}</td><td>{p['set']}</td><td class=num>{dcell(p)}</td></tr>"
                   for p in pairs)

    # gpu
    if "error" in g:
        gp = f"<span class=muted>nvidia-smi unavailable: {E(g['error'])}</span>"
    else:
        frac = g["used"] / g["total"]
        apps = ", ".join(f"{E(n)} {m / 1024:.1f} GB" for _, n, m in g["apps"]) or "no process over 1 GB"
        gp = (f"<b>{E(g['name'])}</b> · {g['used'] / 1024:.1f} / {g['total'] / 1024:.1f} GB · util {g['util']:.0f}%"
              f"<div class='bar big gpu' title='{frac:.0%} of memory'><span style='width:{100 * frac:.1f}%'></span></div>"
              f"<div class=muted>{apps}</div>")
    procs = []
    for c in cmds:
        if "train_grpo.py" in c:
            m = re.search(r"--run (\S+)", c)
            procs.append(f"training <code>{E(m.group(1) if m else '?')}</code>")
        elif "eval_openai.py" in c:
            m = re.search(r"--model (\S+)", c)
            procs.append(f"eval <code>{E(m.group(1) if m else '?')}</code>")
        elif "vllm serve" in c and "train_grpo" not in c:
            procs.append("vLLM server")
    procs = sorted(set(procs))

    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta http-equiv=refresh content=30><title>boundgym dashboard</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root {{ color-scheme: light; --bg:#f4f3f0; --surface-1:#fcfcfb; --line:#e2e1dc; --text-primary:#0b0b0b;
  --text-secondary:#52514e; --muted:#7a7974; --series-1:#2a78d6; --raw:#9ec5f4;
  --good:#0ca30c; --warning:#fab219; --info:#2a78d6; --critical:#d03b3b; }}
@media (prefers-color-scheme: dark) {{ :root {{ color-scheme: dark; --bg:#121211; --surface-1:#1a1a19; --line:#2e2e2b;
  --text-primary:#f0efec; --text-secondary:#c3c2b7; --muted:#8f8e86; --series-1:#3987e5; --raw:#1c5cab;
  --info:#3987e5; }} }}
* {{ box-sizing: border-box }}
body {{ margin:0; padding:20px 24px 40px; background:var(--bg); color:var(--text-primary);
  font:14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif }}
h1 {{ font-size:20px; margin:0 }} h2 {{ font-size:15px; margin:26px 0 10px; color:var(--text-secondary);
  text-transform:uppercase; letter-spacing:.05em }} h3 {{ font-size:14px; margin:0 0 6px }}
header {{ display:flex; gap:16px; align-items:baseline; flex-wrap:wrap }}
.muted {{ color:var(--muted) }} code {{ font-size:12.5px; background:var(--line); padding:0 4px; border-radius:4px }}
.lnk {{ text-decoration:underline dotted }}
.grid4 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px }}
.card {{ background:var(--surface-1); border:1px solid var(--line); border-radius:10px; padding:12px 14px }}
.ch.active {{ border-color:var(--info); box-shadow:0 0 0 1px var(--info) inset }}
.chn {{ font-weight:700; font-size:13px; color:var(--text-secondary) }} .goal {{ margin:2px 0 8px; min-height:40px }}
.chip {{ display:inline-block; font-size:12px; font-weight:600; padding:1px 8px; border-radius:20px;
  border:1px solid currentColor; margin-bottom:6px }}
.chip.good {{ color:var(--good) }} .chip.warning {{ color:#b07a00 }} .chip.info {{ color:var(--info) }}
.chip.muted {{ color:var(--muted) }}
@media (prefers-color-scheme: dark) {{ .chip.warning {{ color:var(--warning) }} }}
.bar {{ height:6px; background:var(--line); border-radius:4px; overflow:hidden; margin:4px 0 2px }}
.bar span {{ display:block; height:100%; background:var(--series-1); border-radius:4px }}
.bar.big {{ height:10px; margin:8px 0 6px }} .bar.gpu span {{ background:var(--text-secondary) }}
details summary {{ cursor:pointer; color:var(--muted); font-size:12px; margin-top:6px }}
details p {{ font-size:12.5px; margin:4px 0 0 }}
.status ul {{ margin:0; padding-left:18px }} .status li {{ margin:3px 0 }}
.todo ul {{ list-style:none; margin:0; padding:0 }} .todo li {{ margin:4px 0; font-size:13px }}
.todo li.x {{ color:var(--muted) }} .todo.done ul {{ max-height:220px; overflow:auto }}
.todo.now {{ border-color:var(--info) }}
.tag {{ font-size:11px; font-weight:700; color:var(--text-secondary); background:var(--line); padding:0 5px; border-radius:4px }}
.run {{ margin-bottom:10px }} .runhead {{ display:flex; gap:10px; align-items:center }}
.charts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:12px; margin-top:8px }}
figure {{ margin:0 }} figcaption {{ font-size:12.5px; font-weight:600 }}
svg {{ width:100%; height:auto; display:block }}
.grid {{ stroke:var(--line); stroke-width:1 }} .tick {{ fill:var(--muted); font-size:10px }}
.series {{ fill:none; stroke:var(--series-1); stroke-width:2; stroke-linejoin:round }}
.raw {{ fill:var(--raw) }} .hit {{ fill:transparent }} .hit:hover {{ fill:var(--series-1); fill-opacity:.35 }}
.lab {{ fill:var(--text-primary); font-size:11px; font-weight:600 }}
table {{ border-collapse:collapse; width:100%; font-size:13px }} th, td {{ padding:5px 10px; border-bottom:1px solid var(--line); text-align:left }}
th {{ color:var(--text-secondary); font-weight:600 }} td.num {{ text-align:right; font-variant-numeric:tabular-nums }}
dl {{ margin:8px 0 0; display:grid; grid-template-columns:max-content 1fr; gap:6px 14px }}
dt {{ font-weight:600 }} dd {{ margin:0; color:var(--text-secondary) }}
.explain p, .note {{ font-size:13px; color:var(--text-secondary); margin:10px 0 4px }}
.explain {{ margin-bottom:10px }}
.whatis > summary {{ cursor:pointer; list-style-position:outside }} .whatis > summary h2 {{ display:inline-block }}
.gl-grid {{ display:grid; grid-template-columns:1fr minmax(260px, 420px); gap:18px; align-items:start }}
@media (max-width: 1000px) {{ .gl-grid {{ grid-template-columns:1fr }} }}
.glossary p {{ margin:0 0 10px }} .demo img {{ width:100%; border-radius:8px; border:1px solid var(--line) }}
.dtab {{ margin:4px 0 10px }}
.empty {{ color:var(--muted); font-size:12.5px; padding:20px 0 }}
</style></head><body>
<header><h1>boundgym</h1><span class=muted>updated {time.strftime('%Y-%m-%d %H:%M:%S')} · refreshes every 30 s</span></header>

<details class=whatis open><summary><h2>What you're looking at</h2></summary>{glossary}</details>

<h2>Chapters</h2><div class=grid4>{cards}</div>

<h2>{E(st_title)}</h2><div class='card status'>{md_bullets(st_body)}</div>

<h2>Todo</h2><div class=grid4>{tcols}</div>

<h2>Machine</h2><div class=card>{gp}<div>Active: {', '.join(procs) or '<span class=muted>nothing from this project</span>'}</div></div>

<h2>Training runs (chapter 3)</h2>
<div class='card explain'>
<b>How to read these charts, and why they don't predict solve rate</b>
<dl>
<dt>Training reward</dt><dd>Share of a step's completions (4 levels × 8 tries) that solved the level within the
<b>training budget</b> (8k tokens for g1–g3, 16k for g4). Strict 0/1. Each step draws <i>different</i> training levels (d2–d4), so
step-to-step swings are mostly level difficulty, not learning; watch the 10-step mean.</dd>
<dt>Share cut off at budget</dt><dd>Share of completions that hit the training budget before giving an answer. A cut-off
always scores 0, so <b>training reward can rise only because fewer answers get cut off</b>, with no better
solving. When reward rises while this falls, suspect brevity, not skill.</dd>
<dt>Solved, of those that finished</dt><dd>Training reward ÷ (1 − share cut off): accuracy on the attempts that
gave an answer inside the budget. It separates the two ways reward can rise:
<code>reward = (1 − cut off) × solved-of-finished</code>. Finishing more often moves the first factor; <b>better
planning moves this one</b>. A successful run raises reward without this falling. In g2 it fell (0.72 → 0.56) while
reward rose: all of the gain came from finishing more. Caveat: newly finishing attempts tend to be the harder ones,
so a mild fall can be selection rather than the model getting worse. Line: ratio of sums over the last 10 steps.</dd>
<dt>Compare runs at the same step, not a run with its own start</dt><dd>Every run trains on <b>the same levels in
the same order</b>, and the difficulty mix drifts by chance (steps 21–27 are hard for everyone). So all three curves
can fall in every run without the model getting worse. Line runs up against each other at the same step; that's a
fair comparison (budgets aside).</dd>
<dt>Solve rate (evaluations below)</dt><dd>The real measure: <b>held-out</b> levels (the frozen d3 and d5 sets, see the difficulty table at the top), 4 tries each,
with a <b>32k budget</b>. At 32k almost nothing gets cut off, so getting shorter doesn't raise solve rate; it shows up
as fewer <i>mean tokens</i> instead.</dd>
</dl>
<p>Consequence: compare runs by <b>solve rate and its confidence interval</b>, not by these curves. Different settings
also shift the curves without changing skill (g1 used 2 levels per step, g2 used 4). Masking cut-offs (g1) only changes the loss,
not the reward shown here.</p>
</div>
{rcards or '<div class=empty>no runs</div>'}

<h2>Evaluations (chapter 3, paired vs base)</h2>
<div class=card><table><thead><tr><th>checkpoint</th><th>set</th><th class=num>solve rate</th>
<th class=num>vs base, pts [95% CI]</th><th class=num>mean tokens</th><th class=num>levels×samples</th></tr></thead>
<tbody>{trs}</tbody></table>
<p class=note><b>Reading it:</b> "vs base" is paired over the same levels; the bracket is the 95% range the true
difference plausibly lies in. <b>n.s.</b> (the range includes 0) means the two can't be told apart with 50 levels × 4
tries, even if the point estimates differ by several points. Example: an apparent 6-point gap between two checkpoints
is within noise unless its range excludes 0; check the comparisons below.
<b>Many comparisons:</b> at 95%, about 1 in 20 comparisons with no real difference still comes out "significant".
With {len(pairs) + sum('diff' in e for e in ev)} comparisons on this page, a single borderline hit (a range ending near 0)
is a lead to re-test with more levels, not a result.</p>
<details{' open' if pairs else ''}><summary>Checkpoints compared with each other (paired, B − A)</summary>
<table><thead><tr><th>B</th><th>A</th><th>set</th><th class=num>B − A, pts [95% CI]</th></tr></thead>
<tbody>{ptrs}</tbody></table></details></div>

<h2>Replication on fresh levels (chapter 3)</h2>
<div class=card><p class=note>A second, independent test of the best checkpoint: 50 <b>new</b> d3 and d5 levels
(seeds from 2,000,000, generated after g4 was chosen, never used for training or any earlier eval), same protocol
(4 tries, 32k). If "vs base" stays significant here, the gain above wasn't luck on one particular set of levels.
Results appear as each model finishes (base first).</p>
{('<table><thead><tr><th>checkpoint</th><th>set</th><th class=num>solve rate</th><th class=num>vs base, pts [95% CI]</th>'
  '<th class=num>mean tokens</th><th class=num>levels×samples</th></tr></thead><tbody>' + rep_trs + '</tbody></table>')
 if rep_trs else '<div class=empty>running: no results yet</div>'}</div>
</body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/media/"):
            name = os.path.basename(self.path[len("/media/"):])
            f = os.path.join(ROOT, "docs", "media", name)
            if name.endswith(".gif") and os.path.isfile(f):
                body = open(f, "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "image/gif")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "max-age=3600")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)
            return
        try:
            body, code = page().encode(), 200
        except Exception as e:
            body, code = f"<pre>dashboard error: {E(repr(e))}</pre>".encode(), 500
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


PIDFILE = os.path.join(ROOT, ".dashboard.pid")  # kill by PID, never by name pattern (see notes/lessons.md)
LOGFILE = os.path.join(ROOT, ".dashboard.log")


def running_pid():
    """PID of the dashboard server from the pidfile, if that process is alive and is really the dashboard."""
    try:
        pid = int(read(PIDFILE).strip())
    except ValueError:
        return None
    return pid if "dashboard.py" in read(f"/proc/{pid}/cmdline") else None


def stop():
    import signal
    pid = running_pid()
    if pid is None:
        return False
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if not os.path.exists(f"/proc/{pid}"):
            break
        time.sleep(0.1)
    if os.path.exists(PIDFILE):
        os.remove(PIDFILE)
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="boundgym progress dashboard (see docs/dashboard.md)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--once", metavar="OUT.html", help="write one snapshot and exit")
    ap.add_argument("--background", action="store_true", help="start detached (log: .dashboard.log) and return")
    ap.add_argument("--restart", action="store_true", help="stop the running dashboard, then start")
    ap.add_argument("--stop", action="store_true", help="stop the running dashboard")
    ap.add_argument("--status", action="store_true", help="say whether it is running, and where")
    a = ap.parse_args()

    if a.once:
        open(a.once, "w").write(page())
        print("wrote", a.once)
    elif a.status:
        pid = running_pid()
        print(f"running (pid {pid}) at http://127.0.0.1:{a.port}" if pid else "not running")
    elif a.stop:
        print("stopped" if stop() else "not running")
    else:
        if a.restart:
            stop()
        elif running_pid():
            raise SystemExit(f"already running (pid {running_pid()}); use --restart to replace it")
        if a.background:
            import subprocess, sys
            child = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--port", str(a.port)],
                                     stdout=open(LOGFILE, "a"), stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True, cwd=ROOT)
            for _ in range(30):  # wait for it to bind and write its pidfile
                if running_pid() == child.pid:
                    break
                if child.poll() is not None:
                    raise SystemExit(f"failed to start; see {LOGFILE}")
                time.sleep(0.1)
            print(f"started (pid {child.pid}) at http://127.0.0.1:{a.port}  · log {LOGFILE}")
        else:
            srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
            open(PIDFILE, "w").write(str(os.getpid()))
            print(f"boundgym dashboard on http://127.0.0.1:{a.port}  (Ctrl+C to stop)", flush=True)
            try:
                srv.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                if running_pid() == os.getpid():
                    os.remove(PIDFILE)
