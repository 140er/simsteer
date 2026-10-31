"""python -m simsteer.testing.fanatec {enumerate,bridge,report}

enumerate  what SimSteer's device detection sees on a simulated base
bridge     the real FFB bridge (vJoy receiver -> model -> DirectInput sink)
           forwarding game packets to a simulated base; `--events` streams
           every force event to a JSONL file as it happens, so a run that is
           killed can still be judged
report     judge an event log: peak force, cap, and whether the force was
           off by a deadline
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import threading
import time
from typing import Sequence

from simsteer.testing.fanatec.ffb import FfbLog
from simsteer.testing.fanatec.profiles import PROFILES
from simsteer.testing.fanatec.sim import FanatecSim


def _live(text: str):
    return text if text == "both" else int(text)


def cmd_enumerate(args) -> int:
    from simsteer.runtime.ffb import dinput_out as di

    with FanatecSim(args.profile, live_instance=args.live) as sim:
        print(json.dumps(di.list_ff_devices(), indent=1))
        sim.assert_no_leaks()
    return 0


def cmd_bridge(args) -> int:
    from simsteer.runtime.ffb import bridge as br
    from simsteer.runtime.ffb import dinput_out as di
    from simsteer.runtime.ffb import vjoy_ffb as vj

    playing = threading.Event()

    def on_event(event: dict) -> None:
        if writer is not None:
            writer(event)
        if event["ev"] == "play" and event["level"] > 0 and not playing.is_set():
            playing.set()
            print(f"playing {event['t']:.6f}", flush=True)

    writer = FfbLog.writer(args.events) if args.events else None
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, lambda *_: stop.set())
    sim = FanatecSim(args.profile, events=on_event).install()
    rx = vj.VJoyFfbReceiver(1)
    check = rx.open()
    if not check.ok:
        print(f"vJoy FFB: {check.row}", file=sys.stderr)
        return 1
    bridge = br.FfbBridge(rx, di.DirectInputSink(), args.cap, say=lambda line: print(line, flush=True))
    bridge.start()
    sender = sim.game.send(args.packets, realtime=True)
    print("bridge running", flush=True)
    t0 = time.monotonic()
    try:
        while not stop.is_set() and bridge.running and time.monotonic() - t0 < args.seconds:
            stop.wait(0.05)
    finally:
        sender.stop()
        bridge.stop()
        rx.close()
        sim.uninstall()
        if writer is not None:
            writer.close()
    print(json.dumps({k: v for k, v in bridge.summary().items() if not k.endswith("_log")}), flush=True)
    return 1 if bridge.failure else 0


def report(log: FfbLog, cap: float | None = None, deadline: float | None = None) -> dict:
    gains = [g for _, g in log.gain_timeline()]
    out = {
        "plays": len(log.plays()),
        "peak_force": round(log.peak_force(), 6),
        "peak_gain": max(gains, default=None),
        "commands": log.commands(),
        "force_ended": log.last_force_time(),
        "failures": [],
    }
    if out["force_ended"] == math.inf:
        out["force_ended"] = "never"
    checks = []
    if cap is not None:
        checks.append(("cap", lambda: log.assert_force_never_above(cap / 100)))
        checks.append(("gain first", lambda: log.assert_gain_set_before_first_start(cap / 100)))
    if deadline is not None:
        checks.append(("stopped", lambda: log.assert_all_stopped(by=deadline)))
    for name, check in checks:
        try:
            check()
        except AssertionError as e:
            out["failures"].append(f"{name}: {e}")
    return out


def cmd_report(args) -> int:
    result = report(FfbLog.load(args.events), args.cap,
                    None if args.killed_at is None else args.killed_at + args.within_ms / 1000)
    print(json.dumps(result, indent=1))
    return 1 if result["failures"] else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m simsteer.testing.fanatec", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enumerate", help="what device detection sees")
    e.add_argument("--profile", choices=list(PROFILES), default="dd2")
    e.add_argument("--live", type=_live, default=0, help="DD2 instance with live input: 0, 1 or both")
    b = sub.add_parser("bridge", help="run the real FFB bridge on a simulated base")
    b.add_argument("--profile", choices=list(PROFILES), default="dd2")
    b.add_argument("--cap", type=float, required=True, help="the bridge's --ffb-test-cap, 1 to 15 percent")
    b.add_argument("--packets", required=True, help="fixture name or file with the game's FFB packets")
    b.add_argument("--events", help="JSONL file for every force event, flushed per event")
    b.add_argument("--seconds", type=float, default=10.0)
    r = sub.add_parser("report", help="judge an event log")
    r.add_argument("events")
    r.add_argument("--cap", type=float, help="percent the force must never exceed")
    r.add_argument("--killed-at", type=float, help="clock time the process was killed")
    r.add_argument("--within-ms", type=float, default=250.0)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {"enumerate": cmd_enumerate, "bridge": cmd_bridge, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
