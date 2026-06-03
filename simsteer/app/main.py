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
"""

from __future__ import annotations

import argparse

from simsteer.app.settings import Settings
from simsteer.games.base import all_profiles
from simsteer.runtime.loop import run


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
    ap.add_argument("--max-width", type=int, default=None,
                    help="cap the overlay window width in px.")
    return ap


def main(argv: list[str] | None = None) -> int:
    profiles = all_profiles()
    game_ids = [p.id for p in profiles]
    args = _build_parser(game_ids).parse_args(argv)

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
    if args.max_width is not None:
        settings.max_width = args.max_width

    # Game: CLI > saved setting > ets2. "auto" isn't resolvable until the
    # Phase 2 detector exists, so fall back to ets2 for now.
    game_id = args.game or settings.game
    if game_id in (None, "", "auto"):
        game_id = "ets2"

    return run(settings, game_id=game_id)


if __name__ == "__main__":
    raise SystemExit(main())
