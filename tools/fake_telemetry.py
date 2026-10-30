"""Send synthetic game telemetry, to test SimSteer without the game.

    python tools/fake_telemetry.py forza --port 7777
    python tools/fake_telemetry.py forza --format alternate --gap 10:20
    python tools/fake_telemetry.py ac
    python tools/fake_telemetry.py ets2

`forza` sends Data Out packets over UDP and works anywhere. `ac` and
`ets2` publish the games' shared-memory pages and need Windows. Speed is
a function of the wall clock, `--speed + --wave * sin(2 pi t / --period)`
with t in Unix seconds, so a SimSteer log with `tel_wall` (from
`--log-timing`) can be checked against it. The car turns gently left and
right on the same period.
"""

from __future__ import annotations

import argparse
import math
import socket
import struct
import sys
import time

FORZA_FORMATS = ("324", "311", "alternate")
WHEELBASE_M = 4.0


def signals(t: float, speed: float, wave: float, period: float) -> tuple[float, float, float]:
    """(speed m/s, yaw rate rad/s left positive, steer -1..1 right positive) at time t."""
    phase = 2 * math.pi * t / period
    v = speed + wave * math.sin(phase)
    yaw = 0.05 * math.sin(phase + 1.0)
    return v, yaw, -yaw * 8.0


def forza_packet(length: int, race_on: bool, v: float, yaw: float, steer: float) -> bytes:
    """A Data Out packet: 324 bytes as Forza Horizon 4 and 5 send it, or
    311 bytes in the Forza Motorsport 7 layout."""
    buf = bytearray(length)
    shift = 12 if length >= 324 else 0
    struct.pack_into("<i", buf, 0, 1 if race_on else 0)
    struct.pack_into("<3f", buf, 32, 0.0, 0.0, v)       # velocity, forward is z
    struct.pack_into("<3f", buf, 44, 0.0, yaw, 0.0)     # angular velocity
    struct.pack_into("<f", buf, 244 + shift, v)
    struct.pack_into("<b", buf, 308 + shift, max(-127, min(127, round(steer * 127))))
    return bytes(buf)


def ac_physics(packet_id: int, v: float, yaw: float, steer: float) -> bytes:
    buf = bytearray(312)
    struct.pack_into("<i", buf, 0, packet_id)
    struct.pack_into("<2f", buf, 24, steer, v * 3.6)
    struct.pack_into("<3f", buf, 296, 0.0, yaw, 0.0)
    return bytes(buf)


def scs_page(tick: int, v: float, yaw: float, steer: float) -> list[tuple[int, bytes]]:
    """(offset, bytes) writes into the SCS telemetry page, see scs_shm.py."""
    angle = math.atan(yaw * WHEELBASE_M / v) if abs(v) >= 1.0 else 0.0
    rot = angle / (2 * math.pi)
    return [(0, b"\x01"), (8, struct.pack("<Q", 1000 + tick * 16_666)),
            (948, struct.pack("<f", v)), (956, struct.pack("<f", steer)),
            (972, struct.pack("<f", steer)), (1200, struct.pack("<4f", rot, rot, 0.0, 0.0)),
            (1920, struct.pack("<f", yaw))]


def _gap(text: str) -> tuple[float, float]:
    try:
        a, b = (float(x) for x in text.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError("use START:END in seconds") from None
    if not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b):
        raise argparse.ArgumentTypeError("need 0 <= START < END")
    return a, b


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("game", choices=("forza", "ac", "ets2"))
    ap.add_argument("--port", type=int, default=7777, help="forza: UDP port")
    ap.add_argument("--host", default="localhost", help="forza: destination")
    ap.add_argument("--format", choices=FORZA_FORMATS, default="324",
                    help="forza: packet length, or alternate 324 and 311")
    ap.add_argument("--race-off", action="store_true",
                    help="forza: send IsRaceOn = 0, as in a menu")
    ap.add_argument("--hz", type=float, default=60.0)
    ap.add_argument("--speed", type=float, default=20.0, help="mean speed, m/s")
    ap.add_argument("--wave", type=float, default=5.0, help="speed swing, m/s")
    ap.add_argument("--period", type=float, default=30.0, help="seconds")
    ap.add_argument("--seconds", type=float, default=0.0, help="0 runs until Ctrl+C")
    ap.add_argument("--gap", metavar="START:END", type=_gap,
                    help="send nothing between these seconds after start")
    args = ap.parse_args(argv)
    gap = args.gap
    if not 0 < args.port < 65536:
        ap.error("--port must be between 1 and 65535")
    if not math.isfinite(args.hz) or args.hz <= 0:
        ap.error("--hz must be positive and finite")
    if not math.isfinite(args.period) or args.period <= 0:
        ap.error("--period must be positive and finite")
    if not math.isfinite(args.seconds) or args.seconds < 0:
        ap.error("--seconds must be nonnegative and finite")
    if not all(math.isfinite(x) for x in (args.speed, args.wave)):
        ap.error("--speed and --wave must be finite")

    sock = page = None
    if args.game == "forza":
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        where = f"UDP {args.host}:{args.port}, format {args.format}"
    else:
        if sys.platform != "win32":
            print(f"{args.game}: shared memory needs Windows")
            return 2
        import mmap
        if args.game == "ac":
            name, size = "Local\\acpmf_physics", 4 * 1024
        else:
            name, size = "Local\\SCSTelemetry", 32 * 1024
        page = mmap.mmap(-1, size, name)
        where = name
    print(f"fake {args.game} telemetry to {where} at {args.hz:g} Hz, "
          f"speed {args.speed:g} +- {args.wave:g} m/s over {args.period:g} s")

    start = time.monotonic()
    next_tick = start
    tick = 0
    sending = True
    next_report = 0.0
    try:
        while args.seconds <= 0 or time.monotonic() - start < args.seconds:
            now = time.time()
            el = time.monotonic() - start
            in_gap = gap is not None and gap[0] <= el < gap[1]
            if in_gap == sending:
                sending = not in_gap
                print(f"{el:7.2f} s  {'sending' if sending else 'paused'}", flush=True)
            v, yaw, steer = signals(now, args.speed, args.wave, args.period)
            if sending:
                tick += 1
                if sock is not None:
                    length = 324 if args.format == "324" or (
                        args.format == "alternate" and tick % 2) else 311
                    sock.sendto(forza_packet(length, not args.race_off, v, yaw, steer),
                                (args.host, args.port))
                elif args.game == "ac":
                    page.seek(0)
                    page.write(ac_physics(tick, v, yaw, steer))
                else:
                    for off, data in scs_page(tick, v, yaw, steer):
                        page.seek(off)
                        page.write(data)
            if el >= next_report:
                print(f"{el:7.2f} s  speed {v:6.3f} m/s  yaw {yaw:+.4f}  "
                      f"packets {tick}", flush=True)
                next_report += 1.0
            next_tick = max(next_tick + 1.0 / args.hz, time.monotonic())
            time.sleep(max(0.0, next_tick - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        if sock is not None:
            sock.close()
        if page is not None:
            if args.game == "ets2":
                page[0] = 0     # sdkActive off, as the plugin does on shutdown
            else:
                struct.pack_into("<i", page, 0, 0)
            page.close()
    print(f"sent {tick} updates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
