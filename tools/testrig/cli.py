import argparse
import json
import os
from pathlib import Path
import re
import sys

from . import RigError
from .guard import require_free
from .session import Session
from .state import lock
from .windows import Windows


def parser():
    command = argparse.ArgumentParser(description="Guarded, disposable Windows game test sessions")
    command.add_argument("game", choices=("guard", "ac", "ets2"))
    command.add_argument("action", nargs="?", choices=("prepare", "start", "ready", "stop", "restore", "status"))
    command.add_argument("--session", default="default", help="unique name for this test run")
    command.add_argument("--state-dir", type=Path, help="private state/backup root (default: LocalAppData)")
    command.add_argument("--timeout", type=int, default=180, help="readiness deadline, 1–600 seconds")
    command.add_argument("--idle-seconds", type=int, help="minimum user inactivity (default: 120; minimum: 60)")
    command.add_argument("--track", help="AC installed track directory (default: magione)")
    command.add_argument("--layout", help="AC track layout directory")
    command.add_argument("--car", help="AC installed car directory (default: lotus_elise_sc)")
    command.add_argument("--hfov", type=float, help="AC horizontal FOV in degrees (default: 120)")
    command.add_argument("--width", type=int, help="test display/game width (default: 1920)")
    command.add_argument("--height", type=int, help="test display/game height (default: 1080)")
    command.add_argument("--xinput-slot", type=int, choices=range(4), help="ETS2 virtual pad slot, 0–3")
    command.add_argument("--profile-source", type=Path, help="ETS2 offline local test profile seed")
    command.add_argument("--telemetry-plugin", type=Path, help="trusted ETS2 SCS telemetry DLL to copy temporarily")
    command.add_argument("--display", choices=("vdd",), help="extend desktop with VDD as primary")
    command.add_argument("--vdd-device", help="exact installed virtual display adapter PnP instance ID")
    command.add_argument("--overlay", action="store_true", default=None, help="black capture-excluded physical display cover")
    command.add_argument("--max-seconds", type=int, help="display worker lifetime (default/max: 1800)")
    command.add_argument("--launch-cancelled", action="store_true", help="restore only: confirm an unresolved Steam launch was cancelled")
    return command


def main(argv=None, *, backend=None):
    command = parser()
    args = command.parse_args(argv)
    if not 1 <= args.timeout <= 600:
        command.error("--timeout must be 1–600 seconds")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.session):
        command.error("--session must contain 1–64 letters, numbers, underscores or hyphens")
    for name, low, high in (("idle_seconds", 60, 3600), ("width", 640, 16384),
                            ("height", 480, 16384), ("hfov", 30, 160), ("max_seconds", 30, 1800)):
        value = getattr(args, name)
        if value is not None and not low <= value <= high:
            command.error(f"--{name.replace('_', '-')} must be {low}–{high}")
    if args.launch_cancelled and args.action != "restore":
        command.error("--launch-cancelled is only valid with restore")
    if args.game == "ac" and args.xinput_slot is not None:
        command.error("--xinput-slot is supported by ETS2; AC uses its default X360 pad selection")
    if args.game == "guard" and args.action:
        command.error("guard takes no game action")
    if args.game != "guard" and not args.action:
        command.error("a game action is required")
    options = {key: value for key, value in vars(args).items() if value is not None and key not in
               ("game", "action", "session", "state_dir", "timeout", "launch_cancelled")}
    for key in ("profile_source", "telemetry_plugin"):
        if key in options:
            options[key] = str(options[key].absolute())
    if args.game == "guard":
        if set(options) - {"idle_seconds"}:
            command.error("guard only accepts --idle-seconds as a configuration option")
    elif args.action not in ("prepare", "start") and options:
        command.error("session configuration options are only valid with prepare or start")
    session = None
    try:
        backend = backend or Windows()
        if args.game == "guard":
            require_free(backend.snapshot(), idle_seconds=args.idle_seconds or 120)
            print(json.dumps({"guard": "clear"}))
            return 0
        root = args.state_dir
        if root is None:
            local = os.environ.get("LOCALAPPDATA")
            if not local:
                raise RigError("LocalAppData is unavailable; specify a private --state-dir")
            root = Path(local) / "SimSteer/testrig"
        root = root.absolute()
        with lock(root):
            session = Session(args.game, root / args.game / args.session, backend)
            try:
                if args.action in ("prepare", "start"):
                    result = (session.prepare(options) if args.action == "prepare" else
                              session.start(options, args.timeout))
                elif args.action == "ready":
                    result = session.ready(args.timeout)
                elif args.action == "status":
                    result = session.status(live=True)
                else:
                    if args.launch_cancelled and session.data:
                        session.data["launch"] = None
                        session.save()
                    result = getattr(session, args.action)()
            except (Exception, KeyboardInterrupt) as exc:
                if args.action in ("prepare", "start", "ready"):
                    session.failure_cleanup(exc)
                raise
        print(json.dumps(result, indent=2))
        return 0
    except (RigError, OSError, ValueError, KeyboardInterrupt) as exc:
        print(f"testrig: {exc or 'interrupted'}", file=sys.stderr)
        for note in getattr(exc, "__notes__", []):
            print(note, file=sys.stderr)
        return 2
