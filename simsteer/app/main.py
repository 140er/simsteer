"""SimSteer entry point.

Resolves settings (saved settings + CLI overrides for the current run),
then hands off to the driving loop. CLI flags win for this run only —
anything the user picks in the tuner's Setup tab is what persists.

Phase 1 runs a single pinned game (default ETS2). Live game detection
and no-restart hot-swap arrive in Phase 2; the guided onboarding UI
and one-click installer come later. For now:

    python -m simsteer.app.main                 # drive ETS2
    python -m simsteer.app.main --no-gamepad     # overlay-only (parity A/B)
    python -m simsteer.app.main --device wheel    # vJoy output

Record a drive, then replay it headless and deterministically:

    python -m simsteer --record clips/drive1 --seconds 900
    python -m simsteer --source clips/drive1 --headless --log-jsonl out.jsonl
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from simsteer.app.settings import Settings
from simsteer.games.base import all_profiles


def _build_parser(game_ids: list[str]) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="simsteer",
                                 description="Drive sim games with the "
                                             "openpilot model.")
    ap.add_argument("--game", choices=game_ids, default=None,
                    help="which game to drive (default: saved setting, "
                         "else ets2). Live auto-detect lands in Phase 2.")
    ap.add_argument("--device", choices=["gamepad", "wheel"], default=None,
                    help="output device (default: saved setting).")
    ap.add_argument("--no-gamepad", action="store_true",
                    help="overlay only; no virtual-pad output. Use for the "
                         "v1-vs-v2 parity diff.")
    ap.add_argument("--force-engage", action="store_true",
                    help="DEV: bypass the engagement gate (won't track "
                         "without calibration).")
    ap.add_argument("--passive-fit-ets2", action="store_true",
                    help="allow LiveParams to fit while disengaged on ETS2.")
    ap.add_argument("--vjoy-device", type=int, default=None,
                    help="vJoy device index when --device wheel.")
    ap.add_argument("--no-tuner", action="store_true",
                    help="don't open the tuner window.")
    ap.add_argument("--max-width", type=int, default=None,
                    help="cap the overlay window width in px.")
    ap.add_argument("--source", metavar="CLIP",
                    help="replay a recorded clip (folder or video) instead of "
                         "the screen. No device output.")
    ap.add_argument("--headless", action="store_true",
                    help="no window, no tuner, no keys.")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="stop after this many frames.")
    ap.add_argument("--log-jsonl", metavar="PATH",
                    help="write one JSON line per frame (time, speed, steer, "
                         "curvature, telemetry state).")
    ap.add_argument("--log-timing", action="store_true",
                    help="add per-stage times in ms to --log-jsonl lines. "
                         "Wall-clock times differ between runs.")
    ap.add_argument("--record", metavar="DIR",
                    help="record screen frames and telemetry to DIR while "
                         "you drive.")
    ap.add_argument("--seconds", type=float, default=900.0,
                    help="how long --record runs (default 900).")
    return ap


def _replay_data_dir(clip: Path) -> str:
    """A throwaway state dir seeded from the clip's `state/`, so replay
    never reads or writes the user's calibration and runs repeat exactly."""
    d = tempfile.mkdtemp(prefix="simsteer-replay-")
    state = (clip if clip.is_dir() else clip.parent) / "state"
    if state.is_dir():
        for p in state.iterdir():
            if p.is_file():
                shutil.copy2(p, d)
    return d


def main(argv: list[str] | None = None) -> int:
    profiles = all_profiles()
    game_ids = [p.id for p in profiles]
    args = _build_parser(game_ids).parse_args(argv)
    if args.source and args.record:
        print("--source and --record are exclusive")
        return 2
    if args.source:
        clip = Path(args.source).expanduser()
        if not clip.exists():
            print(f"no clip at {clip}")
            return 2
        replay_dir = _replay_data_dir(clip)
        user_dir = os.environ.get("SIMSTEER_DATA_DIR")
        os.environ["SIMSTEER_DATA_DIR"] = replay_dir
        try:
            return _main(args, clip)
        finally:
            if user_dir is None:
                os.environ.pop("SIMSTEER_DATA_DIR", None)
            else:
                os.environ["SIMSTEER_DATA_DIR"] = user_dir
            shutil.rmtree(replay_dir, ignore_errors=True)
    return _main(args, None)


def _main(args: argparse.Namespace, clip: Path | None) -> int:

    # Saved settings are the baseline; CLI flags override for this run.
    settings = Settings.load()
    if args.device is not None:
        settings.device = args.device
    if args.no_gamepad:
        settings.no_gamepad = True
    if args.force_engage:
        settings.force_engage = True
    if args.passive_fit_ets2:
        settings.passive_fit_ets2 = True
    if args.vjoy_device is not None:
        settings.vjoy_device = args.vjoy_device
    if args.no_tuner:
        settings.no_tuner = True
    if args.max_width is not None:
        settings.max_width = args.max_width

    # Game: CLI > saved setting > ets2. "auto" isn't resolvable until the
    # Phase 2 detector exists, so fall back to ets2 for now.
    game_id = args.game or settings.game
    if clip is not None and not args.game:
        from simsteer.runtime.replay import clip_meta
        game_id = clip_meta(clip if clip.is_dir() else clip.parent).get("game", game_id)
    if game_id in (None, "", "auto"):
        game_id = "ets2"

    if args.record:
        from simsteer.runtime.replay import record
        return record(args.record, args.seconds, game_id)

    # Deferred so `--help` works without onnxruntime and dxcam.
    from simsteer.runtime.loop import run
    loop_args = dict(headless=args.headless, max_frames=args.max_frames,
                     log_path=args.log_jsonl, log_timing=args.log_timing)
    if clip is None:
        return run(settings, game_id=game_id, **loop_args)

    from simsteer.runtime.loop import WindowKeys
    from simsteer.runtime.replay import ReplayClock, ReplaySource, ReplayTelemetry
    settings.no_gamepad = True
    clock = ReplayClock()
    tel = ReplayTelemetry(clip if clip.is_dir() else clip.parent, clock)
    print(f"replay: {clip}")
    return run(settings, game_id=game_id, source=ReplaySource(clip, clock),
               make_telemetry=lambda: tel, clock=clock,
               keys=None if args.headless else WindowKeys(), **loop_args)


if __name__ == "__main__":
    raise SystemExit(main())
