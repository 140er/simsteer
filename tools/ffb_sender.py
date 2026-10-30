"""Play a fixed force feedback script through DirectInput, standing in for a game.

Run it against vJoy to feed the bridge (bridged), or against the Fanatec
base to feel the same script directly (native), so the two can be compared:

    python tools\\ffb_sender.py --list
    python tools\\ffb_sender.py --device vjoy --log sender.txt
    python tools\\ffb_sender.py --device fanatec --ffb-test-cap 5 --log native.txt
    python tools\\ffb_sender.py --device fanatec --ffb-test-cap 5 --probe

The script covers every effect type vJoy carries (constant, ramp, square,
sine, triangle, both sawtooths, spring, damper, inertia, friction), live
parameter updates, an envelope, a start delay, effect and device gain,
solo start, repeat counts including INFINITE, pause and continue, stop
all, actuators off and on, and a device reset. Every effect uses the one
force axis named by object ID, as a wheel game does.

On a physical wheel the device gain is scaled by `--ffb-test-cap`
(1 to 15 percent, required), and the run asks for Enter at the console,
so it cannot start from a non-interactive shell. If the device refuses a
gain write, the script stops there and the sender exits with code 3.
`--probe` plays one low constant force for 1 s instead of the script, to
check that the wheel honours the capped device gain before the full
script runs on it. It goes through the same open, cap and gain path.
`--list` only enumerates and queries capabilities; it never acquires a
device or sends an effect.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simsteer.core import ffb_model as fm  # noqa: E402
from simsteer.runtime.ffb import dinput_out as di  # noqa: E402
from simsteer.runtime.ffb.bridge import CAP_MAX, CAP_MIN, capped_gain, validate_cap  # noqa: E402

ALL = (fm.DIEP_DURATION | fm.DIEP_SAMPLEPERIOD | fm.DIEP_GAIN | fm.DIEP_TRIGGERBUTTON
       | fm.DIEP_TRIGGERREPEATINTERVAL | fm.DIEP_AXES | fm.DIEP_DIRECTION
       | fm.DIEP_TYPESPECIFICPARAMS | fm.DIEP_STARTDELAY)
TS = fm.DIEP_TYPESPECIFICPARAMS
MS = 1000
EXIT_GAIN_REFUSED = 3
PROBE_EBI = 1
PROBE_SECONDS = 1.0
PROBE_MAGNITUDE = 2000
PROBE_MAGNITUDE_MAX = 3000

Step = tuple[float, fm.DiCall]


def _params(duration_us: int, ts: fm.TypeSpecific, **kw) -> fm.DiParams:
    return fm.DiParams(duration=duration_us, type_specific=ts, **kw)


class Script:
    def __init__(self) -> None:
        self.steps: list[Step] = []
        self.t = 0.0

    def at(self, dt: float, call: fm.DiCall) -> None:
        self.t += dt
        self.steps.append((round(self.t, 3), call))

    def effect(self, eid: int, kind: str, params: fm.DiParams, dt: float = 0.02) -> None:
        flags = ALL | (fm.DIEP_ENVELOPE if params.envelope else 0)
        self.at(dt, fm.DiCall("CreateEffect", eid, kind))
        self.at(0.01, fm.DiCall("SetParameters", eid, kind, flags, params))

    def update(self, dt: float, eid: int, kind: str, params: fm.DiParams, flags: int = TS) -> None:
        self.at(dt, fm.DiCall("SetParameters", eid, kind, flags, params))


def default_script() -> list[Step]:
    """About 6 s of effects; the order is the fixture recording's order."""
    s = Script()
    s.at(0.0, fm.DiCall("SetGain", value=10000))
    s.at(0.01, fm.DiCall("Command", flags=fm.DISFFC_SETACTUATORSON))

    # 1 constant, 400 ms, live magnitude updates through zero
    p = _params(400 * MS, fm.DiConstant(3000))
    s.effect(1, "ConstantForce", p)
    s.at(0.01, fm.DiCall("Start", 1, iterations=1))
    for mag in (-2500, 1500, -10000, 10000):
        p = replace(p, type_specific=fm.DiConstant(mag))
        s.update(0.05, 1, "ConstantForce", p)
    s.at(0.15, fm.DiCall("Stop", 1))
    s.at(0.01, fm.DiCall("Release", 1))

    # 2 constant without end, with an envelope, effect gain 7500
    env = fm.DiEnvelope(attack_level=0, attack_time=300 * MS, fade_level=1000, fade_time=100 * MS)
    p = _params(fm.INFINITE, fm.DiConstant(-4000), gain=7500, envelope=env)
    s.effect(2, "ConstantForce", p)
    s.at(0.01, fm.DiCall("Start", 2, iterations=1))
    s.update(0.30, 2, "ConstantForce", replace(p, gain=5000), fm.DIEP_GAIN)
    s.at(0.30, fm.DiCall("Stop", 2))
    s.at(0.01, fm.DiCall("Release", 2))

    # 3 ramp, played twice, with a 50 ms start delay
    p = _params(150 * MS, fm.DiRamp(-2000, 2000), start_delay=50 * MS)
    s.effect(3, "RampForce", p)
    s.at(0.01, fm.DiCall("Start", 3, iterations=2))
    s.at(0.40, fm.DiCall("Release", 3))

    # 4..8 periodic waves, 250 ms each
    waves = [
        (4, "Square", fm.DiPeriodic(2000, 0, 0, 50 * MS), None),
        (5, "Sine", fm.DiPeriodic(2500, 500, 9000, 40 * MS),
         fm.DiEnvelope(attack_level=500, attack_time=50 * MS, fade_level=0, fade_time=50 * MS)),
        (6, "Triangle", fm.DiPeriodic(1500, -300, 18000, 60 * MS), None),
        (7, "SawtoothUp", fm.DiPeriodic(1800, 0, 0, 30 * MS), None),
        (8, "SawtoothDown", fm.DiPeriodic(1800, 0, 27000, 30 * MS), None),
    ]
    for eid, kind, wave, env in waves:
        p = _params(250 * MS, wave, envelope=env)
        s.effect(eid, kind, p)
        s.at(0.01, fm.DiCall("Start", eid, iterations=1))
        s.at(0.28, fm.DiCall("Release", eid))

    # 9 sine looped without end: a finite effect repeated INFINITE times
    p = _params(100 * MS, fm.DiPeriodic(1000, 0, 0, 25 * MS))
    s.effect(9, "Sine", p)
    s.at(0.01, fm.DiCall("Start", 9, iterations=fm.INFINITE))
    s.at(0.50, fm.DiCall("Stop", 9))
    s.at(0.01, fm.DiCall("Release", 9))

    # 10..13 conditions without end; spring updated live
    spring = fm.DiCondition(0, 3000, 3000, 5000, 5000, 100)
    conds = [
        (10, "Spring", spring),
        (11, "Damper", fm.DiCondition(0, 2000, 2000, 10000, 10000, 0)),
        (12, "Inertia", fm.DiCondition(0, 1000, 1000, 10000, 10000, 0)),
        (13, "Friction", fm.DiCondition(0, 1500, 1500, 10000, 10000, 0)),
    ]
    for eid, kind, cond in conds:
        s.effect(eid, kind, _params(fm.INFINITE, cond))
        s.at(0.01, fm.DiCall("Start", eid, iterations=1))
    s.update(0.20, 10, "Spring", _params(fm.INFINITE, replace(spring, positive_coefficient=4500,
                                                               negative_coefficient=4500, offset=-500)))

    # device-level controls while the conditions play
    s.at(0.20, fm.DiCall("SetGain", value=8000))
    s.at(0.10, fm.DiCall("Command", flags=fm.DISFFC_PAUSE))
    s.at(0.20, fm.DiCall("Command", flags=fm.DISFFC_CONTINUE))
    s.at(0.10, fm.DiCall("Command", flags=fm.DISFFC_SETACTUATORSOFF))
    s.at(0.10, fm.DiCall("Command", flags=fm.DISFFC_SETACTUATORSON))
    s.at(0.10, fm.DiCall("SetGain", value=10000))
    s.at(0.10, fm.DiCall("Start", 11, flags=fm.DIES_SOLO, iterations=1))
    s.at(0.20, fm.DiCall("Command", flags=fm.DISFFC_STOPALL))
    for eid, _, _ in conds:
        s.at(0.01, fm.DiCall("Release", eid))

    # 14 constant left playing, then a device reset clears it
    s.effect(14, "ConstantForce", _params(fm.INFINITE, fm.DiConstant(2000)))
    s.at(0.01, fm.DiCall("Start", 14, iterations=1))
    s.at(0.30, fm.DiCall("Command", flags=fm.DISFFC_RESET))
    s.at(0.01, fm.DiCall("Release", 14))
    return s.steps


def _precise_timer(on: bool) -> None:
    if sys.platform == "win32":
        import ctypes

        winmm = ctypes.WinDLL("winmm")
        (winmm.timeBeginPeriod if on else winmm.timeEndPeriod)(1)


def _gain_refused(gain: int, hr: int) -> int:
    print(f"stopped: the device refused device gain {gain} ({di.hresult_name(hr)}); "
          "no effect was sent after it", file=sys.stderr)
    return EXIT_GAIN_REFUSED


def play(sink: di.FfbSink, steps: list[Step], seconds: float, cap: float | None, log_path: str | None) -> int:
    lines: list[str] = []
    errors = 0
    refused: tuple[int, int] | None = None
    t0 = time.perf_counter()
    loop = 0
    _precise_timer(True)
    try:
        while True:
            base = time.perf_counter()
            for t, call in steps:
                delay = base + t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                if call.op == "SetGain" and cap is not None:
                    call = fm.DiCall("SetGain", value=capped_gain(call.value, cap))
                c0 = time.perf_counter()
                hr = sink.execute(call)
                c1 = time.perf_counter()
                if di.failed(hr):
                    errors += 1
                lines.append(f"{(c0 - t0) * 1000:.3f}\t{call.text()}\t{di.hresult_name(hr)}\t{(c1 - c0) * 1e6:.0f}")
                if call.op == "SetGain" and di.failed(hr):
                    refused = (call.value, hr)
                    break
            loop += 1
            if refused or not seconds or time.perf_counter() - t0 >= seconds:
                break
            time.sleep(0.2)
    finally:
        _precise_timer(False)
        sink.execute(fm.DiCall("Command", flags=fm.DISFFC_STOPALL))
    if log_path:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("t_ms\tcall\tresult\tcall_us\n" + "\n".join(lines) + "\n")
    print(f"played {loop} loop(s), {len(lines)} calls, {errors} failed")
    if refused:
        return _gain_refused(*refused)
    return 1 if errors else 0


def probe_magnitude(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = 0
    if not 1 <= value <= PROBE_MAGNITUDE_MAX:
        raise argparse.ArgumentTypeError(f"must be a whole number from 1 to {PROBE_MAGNITUDE_MAX}, got {text!r}")
    return value


def probe_summary(cap: float | None, magnitude: int) -> str:
    gain = fm.DI_FFNOMINALMAX if cap is None else capped_gain(fm.DI_FFNOMINALMAX, cap)
    peak = magnitude * gain / fm.DI_FFNOMINALMAX**2 * 100
    limit = "no cap" if cap is None else f"cap {cap:g} percent"
    return (f"probe: one constant force for {PROBE_SECONDS:g} s, magnitude {magnitude} of {fm.DI_FFNOMINALMAX}, "
            f"{limit}, expected peak {peak:.3g} percent of the wheel's peak torque")


def probe(sink: di.FfbSink, magnitude: int) -> int:
    """One constant force in a fixed direction for PROBE_SECONDS, no envelope
    and no repeat, then stopped and unloaded on every exit path."""
    results: list[tuple[fm.DiCall, int]] = []

    def run(call: fm.DiCall) -> bool:
        hr = sink.execute(call)
        results.append((call, hr))
        return not di.failed(hr)

    params = _params(round(PROBE_SECONDS * 1e6), fm.DiConstant(magnitude))
    created = False
    try:
        created = run(fm.DiCall("CreateEffect", PROBE_EBI, "ConstantForce"))
        if (created and run(fm.DiCall("SetParameters", PROBE_EBI, "ConstantForce", ALL, params))
                and run(fm.DiCall("Start", PROBE_EBI, iterations=1))):
            time.sleep(PROBE_SECONDS)
    finally:
        if created:
            run(fm.DiCall("Stop", PROBE_EBI))
            run(fm.DiCall("Release", PROBE_EBI))
        run(fm.DiCall("Command", flags=fm.DISFFC_STOPALL))
    bad = [f"{call.op} {di.hresult_name(hr)}" for call, hr in results if di.failed(hr)]
    if bad:
        print(f"probe: {len(bad)} of {len(results)} calls failed: {', '.join(bad)}")
        return 1
    print(f"probe: all {len(results)} calls returned OK")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ffb_sender", description="Play a fixed FFB script through DirectInput.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="list force feedback devices and their effects (read only)")
    mode.add_argument("--probe", action="store_true",
                      help=f"play one constant force for {PROBE_SECONDS:g} s instead of the script")
    mode.add_argument("--print-script", action="store_true", help="print the script and exit")
    p.add_argument("--device", choices=("vjoy", "fanatec"), help="where to play the script")
    p.add_argument("--ffb-test-cap", type=float, metavar="PERCENT",
                   help=f"required on a physical wheel: device gain ceiling, {CAP_MIN:g} to {CAP_MAX:g} percent")
    p.add_argument("--seconds", type=float, default=0, help="repeat the script for this long (default: once)")
    p.add_argument("--log", metavar="FILE", help="write every call with its result and duration")
    p.add_argument("--probe-magnitude", type=probe_magnitude, metavar="N",
                   help=f"probe magnitude, 1 to {PROBE_MAGNITUDE_MAX} of {fm.DI_FFNOMINALMAX} (default "
                        f"{PROBE_MAGNITUDE}: at a 5 percent cap, 1 percent of peak torque, enough to see the rim move)")
    args = p.parse_args(argv)
    if args.probe:
        if args.seconds or args.log:
            p.error("--seconds and --log belong to the script, not --probe")
        if args.probe_magnitude is None:
            args.probe_magnitude = PROBE_MAGNITUDE
    elif args.probe_magnitude is not None:
        p.error("--probe-magnitude needs --probe")

    steps = default_script()
    if args.print_script:
        for t, call in steps:
            print(f"{t:7.3f}  {call.text()}")
        return 0
    if sys.platform != "win32":
        print("DirectInput needs Windows.", file=sys.stderr)
        return 2
    if args.list:
        print(json.dumps(di.list_ff_devices(), indent=1))
        return 0
    if args.device is None:
        p.error("--device or --list is required")

    cap = None
    if args.device == "fanatec":
        if args.ffb_test_cap is None:
            print("Refusing: --ffb-test-cap PERCENT is required on a physical wheel.", file=sys.stderr)
            return 2
        cap = validate_cap(args.ffb_test_cap)
        if not sys.stdin.isatty():
            print("Refusing: playing on a physical wheel needs someone at the console.", file=sys.stderr)
            return 2
    if args.probe:
        print(probe_summary(cap, args.probe_magnitude))
    if cap is None:
        sink = di.DirectInputSink(match=lambda d: d.is_vjoy, allow_vjoy=True)
    else:
        if args.probe:
            input("Probe on the Fanatec base. Hands off the rim, Enter to start, Ctrl+C to cancel: ")
        else:
            input(f"Script on the Fanatec base at {cap:g} percent. Hands near the rim, "
                  "Enter to start, Ctrl+C to cancel: ")
        sink = di.DirectInputSink()
    gain = fm.DI_FFNOMINALMAX if cap is None else capped_gain(fm.DI_FFNOMINALMAX, cap)
    try:
        opened = sink.open(gain)
    except di.GainRefused as e:
        return _gain_refused(gain, e.hr)
    if not opened:
        print(f"no device: {sink.last_error}", file=sys.stderr)
        return 1
    print(f"playing on {sink.device.name} ({sink.device.vid:04X}:{sink.device.pid:04X}), axis 0x{sink.axis_id:08X}")
    try:
        if args.probe:
            return probe(sink, args.probe_magnitude)
        return play(sink, steps, args.seconds, cap, args.log)
    finally:
        sink.close()


if __name__ == "__main__":
    raise SystemExit(main())
