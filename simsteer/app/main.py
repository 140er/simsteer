"""SimSteer entry point.

Resolves settings (saved settings + CLI overrides for the current run),
then hands off to the driving loop. CLI flags win for this run only —
anything the user picks in the tuner's Setup tab is what persists.

Games are TOML profiles (`docs/PROFILES.md`). `--game auto` drives the
first profile whose process is running at launch; switching games needs
a restart. For now:

    python -m simsteer --list-games             # the known games
    python -m simsteer                          # drive the running game
    python -m simsteer --no-gamepad             # overlay-only (parity A/B)
    python -m simsteer --device wheel           # vJoy output

Record a drive, then replay it headless and deterministically:

    python -m simsteer --record clips/drive1 --seconds 900
    python -m simsteer --source clips/drive1 --headless --log-jsonl out.jsonl
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from simsteer.app.settings import Settings
from simsteer.games.base import Registry, registry
from simsteer.games.profile import GameProfile


def _build_parser(game_ids: list[str]) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="simsteer",
                                 description="Drive sim games with the "
                                             "openpilot model.")
    ap.add_argument("--game", choices=["auto", *game_ids], default=None,
                    help="game profile to drive, or auto for the running "
                         "game (default: saved setting).")
    ap.add_argument("--list-games", action="store_true",
                    help="print the game profiles and exit.")
    ap.add_argument("--device", choices=["gamepad", "wheel"], default=None,
                    help="output device (default: saved setting).")
    ap.add_argument("--no-gamepad", action="store_true",
                    help="overlay only; no virtual-pad output. Use for the "
                         "v1-vs-v2 parity diff.")
    ap.add_argument("--force-engage", action="store_true",
                    help="DEV: bypass the engagement gate (won't track "
                         "without calibration).")
    ap.add_argument("--passive-fit", "--passive-fit-ets2", action="store_true",
                    help="let LiveParams fit while disengaged in games "
                         "whose profile does not allow it.")
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


def _list_games(games: Registry) -> None:
    if not games.profiles:
        print("no game profiles")
        return
    rows = [("id", "game", "process", "telemetry", "")]
    rows += [(p.id, p.display_name, " ".join(p.process_names) or "-",
              p.telemetry.kind,
              " ".join(([] if p.tested else ["untested"])
                       + ([] if p.builtin else [f"user: {p.path}"])))
             for p in games.profiles]
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths + [0])).rstrip())


def _pick_game(games: Registry, cli: str | None, saved: str,
               clip: Path | None) -> GameProfile | None:
    """CLI, else the clip's game when replaying, else the saved setting.
    `auto` is the first running game, else `Registry.fallback()`. A replay
    never looks at running processes, so it stays deterministic."""
    if saved != "auto" and games.get(saved) is None:
        if saved:
            print(f"warning: saved game {saved!r} has no profile, using auto")
        saved = "auto"
    wanted = cli or saved
    if clip is not None and not cli:
        from simsteer.runtime.replay import clip_meta
        clip_game = clip_meta(clip if clip.is_dir() else clip.parent).get("game")
        if clip_game and games.get(clip_game) is None:
            print(f"warning: clip game {clip_game!r} has no profile, using auto")
            wanted = "auto"
        elif clip_game:
            wanted = clip_game
    if wanted != "auto":
        return games.get(wanted)
    if clip is None:
        from simsteer.games.processes import running_process_names
        found = games.detect(running_process_names())
        if found is not None:
            print(f"auto: {found.display_name} is running")
            return found
    found = games.fallback()
    if found is not None:
        print(f"auto: no known game running, using {found.display_name}")
    return found


def main(argv: list[str] | None = None) -> int:
    # Loaded before a replay swaps the data dir, so user profiles count.
    games = registry()
    for err in games.errors:
        print(f"game profile skipped: {err}", file=sys.stderr)
    args = _build_parser(games.ids).parse_args(argv)
    if args.list_games:
        _list_games(games)
        return 0
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
            return _main(args, games, clip)
        finally:
            if user_dir is None:
                os.environ.pop("SIMSTEER_DATA_DIR", None)
            else:
                os.environ["SIMSTEER_DATA_DIR"] = user_dir
            shutil.rmtree(replay_dir, ignore_errors=True)
    return _main(args, games, None)


def _main(args: argparse.Namespace, games: Registry, clip: Path | None) -> int:

    # Saved settings are the baseline; CLI flags override for this run.
    settings = Settings.load()
    if args.device is not None:
        settings.override(device=args.device)
    if args.no_gamepad:
        settings.override(no_gamepad=True)
    if args.force_engage:
        settings.override(force_engage=True)
    if args.passive_fit:
        settings.override(passive_fit=True)
    if args.vjoy_device is not None:
        settings.override(vjoy_device=args.vjoy_device)
    if args.no_tuner:
        settings.override(no_tuner=True)
    if args.max_width is not None:
        settings.override(max_width=args.max_width)

    game = _pick_game(games, args.game, settings.game, clip)
    if game is None:
        print("no game profiles found")
        return 2

    if args.record:
        from simsteer.runtime.replay import record
        return record(args.record, args.seconds, game)

    # Deferred so `--help` works without onnxruntime and dxcam.
    from simsteer.runtime.loop import run
    loop_args = dict(headless=args.headless, max_frames=args.max_frames,
                     log_path=args.log_jsonl, log_timing=args.log_timing)
    if clip is None:
        return run(settings, game, **loop_args)

    from simsteer.runtime.loop import WindowKeys
    from simsteer.runtime.replay import ReplayClock, ReplaySource, ReplayTelemetry
    settings.override(no_gamepad=True)
    clock = ReplayClock()
    tel = ReplayTelemetry(clip if clip.is_dir() else clip.parent, clock)
    print(f"replay: {clip}")
    return run(settings, game, source=ReplaySource(clip, clock),
               make_telemetry=lambda: tel, clock=clock,
               keys=None if args.headless else WindowKeys(), **loop_args)


if __name__ == "__main__":
    raise SystemExit(main())
