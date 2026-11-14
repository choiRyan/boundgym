# Progress dashboard

A local web page showing where the project stands. It covers:
- what boundgym is, with the difficulty levels explained
- the chapter pipeline
- the current status and the todo list
- GPU use
- the training runs, with progress, an ETA and charts
- the evaluation results, with confidence intervals

Code: `scripts/dashboard.py`. It uses only the standard library and the kit venv (`.venv`).
It only **reads** the project, fresh on every page load, from:
- `progress.md`
- `todo.md`
- `chapter3/runs/*` and `chapter3/results/*`
- `levels/`
- `docs/media/`
- `nvidia-smi`

It binds to `127.0.0.1` only, so it can't be reached from other machines.

## Commands

Run from the project root (`boundgym/`):

| task | command |
|---|---|
| **start** (detached, keeps running after the terminal closes) | `.venv/bin/python scripts/dashboard.py --background` |
| **open** | http://127.0.0.1:8765 (refreshes itself every 30 s) |
| **check** | `.venv/bin/python scripts/dashboard.py --status` |
| **stop** | `.venv/bin/python scripts/dashboard.py --stop` |
| **restart** (e.g. after editing `dashboard.py`) | `.venv/bin/python scripts/dashboard.py --restart --background` |
| run in the foreground instead | `.venv/bin/python scripts/dashboard.py` (Ctrl+C stops it) |
| another port | add `--port 8800` to any of the above |
| one static snapshot, no server | `.venv/bin/python scripts/dashboard.py --once snapshot.html` |

Behaviour worth knowing:
- Starting while one is already running refuses with a message. Use `--restart` to replace it.
- `--stop` and `--status` use the pidfile `.dashboard.pid`, and check that the PID really is the dashboard before
  acting. Background output goes to `.dashboard.log`. Both files are git-ignored.
- **Editing `todo.md`, `progress.md` or results needs no restart**, since they're re-read on every request. Only edits to
  `scripts/dashboard.py` itself need `--restart`.
- **Don't stop it with `pkill -f dashboard`.** A name pattern also matches any shell whose command line contains it,
  which has killed the calling shell several times here. See `notes/lessons.md`. Use `--stop`.

## After a reboot

It does not start on its own. Run the start command again. To have it start at login, add a systemd user unit
(optional, not installed):

```ini
# ~/.config/systemd/user/boundgym-dashboard.service
[Unit]
Description=boundgym progress dashboard

[Service]
WorkingDirectory=/path/to/boundgym
ExecStart=/path/to/boundgym/.venv/bin/python scripts/dashboard.py
Restart=on-failure

[Install]
WantedBy=default.target
```

Then: `systemctl --user enable --now boundgym-dashboard`. Under systemd, use `systemctl --user stop|restart`
instead of the flags above.

## Where its content comes from

| section | source | to change it |
|---|---|---|
| What you're looking at | text in `dashboard.py`; difficulty table computed from `boundgym/generate.py` + `levels/` | edit `dashboard.py` |
| Chapters | chapter table in `progress.md`; task bars from `todo.md` tags `(chN)` | edit those files |
| Current status | the `## Current status` section of `progress.md` | edit `progress.md` |
| Todo | `todo.md` (`## Now/Next/Later/Done`, `- [ ] (chN) text`) | edit `todo.md` |
| Machine | `nvidia-smi`, running python processes | n/a |
| Training runs | `chapter3/runs/<run>/`: `args.json`, `metrics.jsonl` (or the `chapter3/<g?>.log` dict lines for g1–g3), `profile.jsonl` (pace), checkpoints | written by `train_grpo.py` |
| Evaluations | `chapter3/results/<name>/*.jsonl`, paired bootstrap vs `base` and between checkpoints | written by `eval_ckpt.sh` |

## Troubleshooting

- **Page doesn't load:** run `--status`. If it says not running, start it and read `.dashboard.log` if that fails.
- **"Address already in use":** something else holds the port. Either pick `--port`, or find the owner with
  `ss -ltnp | grep 8765` and stop it by PID.
- **A run shows "stopped" while it's training:** detection looks for a python process with `--run <name>` in its
  command line. A run launched another way won't be detected.
- **The page shows only `dashboard error: ...`:** some input failed to parse (e.g. a malformed line in `todo.md`
  or a results file mid-write). The message names the exception; reload in 30 s, or fix the file. The difficulty table
  and GPU sections fail softly instead (an inline note, rest of the page intact).
