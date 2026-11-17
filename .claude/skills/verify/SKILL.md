---
name: verify
description: Run SimSteer headless on a synthetic clip with the real ONNX models and check a change end-to-end, with no game, no virtual device and no wheel, before committing or opening a PR.
---

# Verify SimSteer

"Verified" means the real loop (`python -m simsteer --source`) replays a clip through the pinned driving
models and writes sane, repeatable per-frame steering, and that the lint and test suites pass. All of it runs
on Linux without a game, a virtual pad, vJoy or a wheel. Total runtime is about 1 to 2 minutes after the first
model download.

## Never

- Never open a physical wheel, vJoy or ViGEm device, and never apply force feedback. Do not run
  `tools/ffb_sender.py` or `python -m simsteer.runtime.ffb.bridge --attended`; they need Windows and hardware.
- Never launch the live screen-capture path (plain `python -m simsteer` without `--source`, or `--record`).
  Replay mode forces no device output and uses a throwaway state dir.
- Never read or write the real per-user state dir (`%LOCALAPPDATA%\SimSteer`). Point `SIMSTEER_DATA_DIR` at a
  temp dir for every run.
- Never commit `models/*.onnx`, clips or logs (they are git-ignored; keep it that way).

## 1. Cheap checks

```bash
timeout 10m scripts/check.sh   # ruff, pytest, `--help`; creates .venv with uv on first run (~15 s, ~10 s after)
```

Needs `uv` and Python 3.11. Run only the tests you touched while iterating, for example
`.venv/bin/python -m pytest tests/test_replay.py -q -p no:cacheprovider`; the full file set is small, so
`check.sh` is fine before every push. `tests/test_model_contract.py` skips unless the models from step 2
are present, so run step 2 before trusting a green result. Windows-only tests skip on Linux; list them in the PR.

## 2. Start it for real

```bash
T=$(mktemp -d); export SIMSTEER_DATA_DIR=$T/data; mkdir -p "$SIMSTEER_DATA_DIR"
timeout 10m .venv/bin/python tools/fetch_model.py --check   # pinned, SHA256-checked; ~60 MB once, then cached
timeout 5m .venv/bin/python tools/make_synthetic_clip.py --out "$T/clip" --frames 200   # ~3 s, deterministic
timeout 5m .venv/bin/python -m simsteer --source "$T/clip" --headless --max-frames 150 --log-jsonl "$T/a.jsonl"
echo "exit=$?"
```

Ready when the run exits 0 after printing `replay: <clip>`, `vision: CPUExecutionProvider` and `frames: 150`,
about 10 s on CPU. The model fetch ends with `contract: inputs, outputs and output slices match constants.py`.
Nothing keeps running afterwards, so there is no server to stop.

## 3. Check the change

Pick the checks that exercise what you changed.

1. Sane output: the log has one line per frame and finite steering within range.
   ```bash
   .venv/bin/python -c "
   import json,math,sys
   r=[json.loads(l) for l in open(sys.argv[1])]
   assert len(r)==150 and all(math.isfinite(x['steer_axis']) and abs(x['steer_axis'])<=1 for x in r)
   assert all(x['telemetry']=='ok' for x in r); print('ok', len(r), 'frames, last steer', r[-1]['steer_axis'])" "$T/a.jsonl"
   ```
2. Determinism: a second replay of the same clip must be byte-identical (`--log-timing` is off, so no wall-clock
   fields). Any diff means the change broke replay determinism or leaked state.
   ```bash
   timeout 5m .venv/bin/python -m simsteer --source "$T/clip" --headless --max-frames 150 --log-jsonl "$T/b.jsonl" >/dev/null && cmp "$T/a.jsonl" "$T/b.jsonl" && echo identical
   ```
3. Behaviour change in control, calibration or preprocess: compare `steer_axis` and `curvature` against a replay
   of the base branch (`git stash` or a second worktree; same clip, same frame count) and state the difference.
   For loop timing use `python tools/replay_bench.py --trunk origin/main --head HEAD --runs 3`; it is
   informational, not a gate.
4. FFB code (`simsteer/core/ffb_model.py`, `simsteer/runtime/ffb/`): the pure decoder and bridge logic run in
   `tests/test_ffb_*.py` on recorded packets with fake sinks. The refusal path also runs on any host:
   `.venv/bin/python -m simsteer.runtime.ffb.bridge` must exit 2 with "Refusing to start: --ffb-test-cap PERCENT is required".

## 4. Evidence

Put in the PR body, from the exact commit you push (re-run after any further change): the `scripts/check.sh`
tail (`All checks passed!`, `N passed, M skipped`), the replay command with `exit=0` and the `frames: 150` line,
the output of check 1 and the `identical` line from check 2, plus any base-versus-head comparison from check 3.

## 5. Clean up

```bash
rm -rf "$T"; unset SIMSTEER_DATA_DIR T
```

Replay removes its own throwaway state dir. Nothing is left running; if you started anything in the background,
stop it by the PID you captured at spawn, never by name or pattern. Leave `.venv/` and the model cache
(`~/.cache/simsteer`); both are ignored.

## If verify cannot run

Say so explicitly in the PR with the reason and the step that failed (no network for `fetch_model.py`, no `uv`,
no Python 3.11). Never claim "verified". Changes that need a Windows desktop session, vJoy/ViGEm drivers,
screen capture, a game or a physical wheel (device output, global hotkeys, the overlay and tuner windows, FFB on
real hardware) cannot be checked here: list what ran locally and name the part left for the owner to test on the
Windows host (`scripts/check.ps1` runs the same checks plus the Windows import check there).
