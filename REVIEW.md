# Review guide

SimSteer drives sim-racing games with comma.ai's openpilot model and steers through a virtual pad (ViGEm) or
wheel (vJoy), with an optional force feedback bridge to a physical wheel. It must never steer wildly, apply
unbounded force, or corrupt a user's calibration.

Severity tags: **Important** = blocks merge. **Nit** = mention, never blocks. **Pre-existing** = problem already on the base branch; note it, never block this PR for it.

## Important
- Wheel force and the FFB bridge (`simsteer/runtime/ffb/`, `simsteer/core/ffb_model.py`, `tools/ffb_sender.py`): the force cap (`--ffb-test-cap`, 1 to 15 percent) must stay required, validated and applied to the device gain before any effect plays. A refused gain must stop forwarding and close the wheel. Without `--attended` the bridge must never open a physical wheel. Effects with no end must stay re-armed short effects so force stops soon after the process dies. Flag any path that raises force, skips the cap, or leaves an effect running on exit or error.
- Steering output (`simsteer/runtime/output/`, `simsteer/core/control.py`, `simsteer/runtime/loop.py`): axis values stay clamped to [-1, 1], NaN or inf never reaches a device, and engagement stays gated (calibration, FPS, telemetry) with auto-disengage on low FPS. `--force-engage` is dev-only and must never default to on or be enabled implicitly. A new code path that writes to the device without going through the engage state is a bug.
- Replay isolation (`simsteer/runtime/replay.py`, `simsteer/app/main.py`): `--source` must use a throwaway state dir, open no device, and never read or write the user's calibration. `--log-jsonl` output without `--log-timing` must stay byte-identical across two replays of one clip; nondeterminism (wall-clock, unseeded randomness, dict or thread order) in the loop is a regression.
- Persistent user state (`simsteer/paths.py`, `settings.py`, `core/learners/livecalib.py`, `core/learners/liveparams.py`, wizard flags): state files are per game and device; a change must not mix games, drop the cold-start seeding, or delete or overwrite a user's file without an explicit reset (`R`, `--reset-calib`). `SIMSTEER_DATA_DIR` must keep overriding the state dir.
- Online learners: only close a loop on a signal independent of what is being tuned (see `docs/ARCHITECTURE.md`, "Removed auto-tuners"). Re-adding an auto-tuner for FOV, lookahead, authority or anticipation is Important. Camera FOV stays static and manual.
- Model contract and pinned models (`tools/fetch_model.py`, `simsteer/core/constants.py`, `simsteer/core/model.py`): the openpilot commit and SHA256 pins change only together with a passing `check_contract`; downloads stay HTTPS and hash-verified. Changing slice constants without the model pin (or the reverse) breaks decoding silently.
- Platform imports: Windows-only packages (dxcam, vgamepad, pyvjoy, DirectInput via ctypes) must be imported lazily so Linux, `--help` and the test suite keep working. A top-level import that fails off Windows is Important.
- Tests for fixes: a bug fix needs a regression test that fails on the base and passes on the head. Tests must not sleep, retry, or depend on wall-clock timing.

## Nit
- New or changed CLI flags missing from `docs/SETUP.md` (CLI flags table) or `--help` text.
- Dead code, unused imports or names that restate the type, in code the PR touches.
- Ruff only enforces syntax errors and undefined names; style cleanups unrelated to the change are a nit, not a request to expand scope.

## Pre-existing
- The legacy `pilot/` package, `debug/` and several `tools/test_*.py` scripts still exist beside `simsteer/`, and docs and `RELEASE.md` still reference `pilot/` paths and `python -m pilot.main`.
- Ruff selects only `E9`, `F63`, `F7`, `F82`; broader lint debt in the legacy tree is known.
- Windows-only behaviour (drivers, capture, hotkeys, overlay, real FFB) has no Linux coverage; those tests skip off Windows.
- Report other pre-existing problems as Pre-existing, not Important.

## Rules the reviewer enforces
- The PR ran the verify skill (`.claude/skills/verify/SKILL.md`) or says why it could not; claims of "verified" need evidence in the PR body.
- A PR never changes the checks or rules that judge it (REVIEW.md, merge-gate or check config, CI recipes, test thresholds) to make itself pass. The merge gate reads REVIEW.md from the base branch, never from the PR head.
- No secrets, tokens or credentials in code, logs, fixtures, commits or PR text.
