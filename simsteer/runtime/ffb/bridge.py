"""Pass the game's force feedback from vJoy to the Fanatec wheel.

The game binds vJoy as its wheel. vJoy hands every PID packet to
`VJoyFfbReceiver`, `FfbModel` turns it into the DirectInput call the game
would have made on the wheel, and `DirectInputSink` makes that call on the
Fanatec base. One thread does all of it, so DirectInput stays on the thread
that opened it.

While forwarding, the only intended difference from native play is safety:
effects that would never end play in a 200 ms window the model renews
every 100 ms, so if SimSteer dies the force stops within 200 ms.

This PR runs the bridge standalone and test-only:

    python -m simsteer.runtime.ffb.bridge --ffb-test-cap 5 --attended

`--ffb-test-cap` (1 to 15 percent) is required. It scales the device gain
on the wheel, so the strongest force the game can ask for is that share of
the base's peak torque (5 percent of the DD2 is about 1.25 Nm). The wheel
gets the capped gain before anything else; if it ever refuses a capped
gain, the bridge stops forwarding and closes the wheel, which stops and
unloads every effect. Without `--attended` the bridge never opens a
physical wheel; it reports what it finds and keeps waiting, which is what
unattended runs need. `--dry-run` logs the calls instead of making them.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from simsteer.core import ffb_model as fm
from simsteer.runtime.ffb import dinput_out as di
from simsteer.runtime.ffb.vjoy_ffb import Packet, crosscheck

CAP_MIN = 1.0
CAP_MAX = 15.0
REOPEN_S = 1.0
IDLE_WAIT_S = 0.02
ERROR_LOG_MAX = 50
LATENCY_WINDOW = 20_000


def validate_cap(cap: float) -> float:
    cap = float(cap)
    if not CAP_MIN <= cap <= CAP_MAX:  # also rejects NaN
        raise ValueError(f"--ffb-test-cap must be {CAP_MIN:g} to {CAP_MAX:g} percent, got {cap:g}")
    return cap


def capped_gain(game_gain: int, cap_percent: float) -> int:
    """Device gain on the wheel: the game's device gain times the cap."""
    return min(fm.DI_FFNOMINALMAX, round(game_gain * cap_percent / 100))


class CapNotApplied(RuntimeError):
    """The wheel refused the capped device gain, so nothing may play on it."""


class PacketSource(Protocol):
    def get(self, timeout: float | None) -> Packet | None: ...

    def note_mismatch(self, problems: list[str], packet: Packet) -> None: ...


@dataclass
class BridgeStats:
    """`latencies_ms` keeps the last LATENCY_WINDOW packets; the count and
    the maximum cover the whole run."""

    errors: int = 0
    error_log: list[str] = field(default_factory=list)
    device_events: list[str] = field(default_factory=list)
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=LATENCY_WINDOW))
    latency_count: int = 0
    latency_max: float | None = None

    def add_latency(self, ms: float) -> None:
        self.latencies_ms.append(ms)
        self.latency_count += 1
        if self.latency_max is None or ms > self.latency_max:
            self.latency_max = ms


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = min(len(ordered) - 1, max(0, round(q / 100 * (len(ordered) - 1))))
    return ordered[k]


class FfbBridge:
    """Forwarding thread. `start()`, read `status`, `stop()`."""

    def __init__(
        self,
        source: PacketSource,
        sink: di.FfbSink,
        cap_percent: float,
        *,
        attended: bool = True,
        list_devices: Callable[[], list[di.DeviceInfo]] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        reopen_s: float = REOPEN_S,
        say: Callable[[str], None] = lambda line: None,
    ) -> None:
        self.source = source
        self.sink = sink
        self.cap = validate_cap(cap_percent)
        self.attended = attended
        self.list_devices = list_devices
        self.clock = clock
        self.reopen_s = reopen_s
        self.say = say
        self.model = fm.FfbModel()
        self.stats = BridgeStats()
        self.device_open = False
        self.failure: BaseException | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reopen_at = 0.0
        self._last_seen: tuple[str, ...] | None = None

    # ---- lifecycle

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="ffb-bridge", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def status(self) -> str:
        """HUD line: `FFB pass 0 unknown`, or why nothing is forwarded."""
        if self.failure is not None:
            return f"FFB stopped: {self.failure}"
        if not self.device_open:
            return "FFB waiting for wheel"
        return self.model.status()

    def summary(self) -> dict:
        c = self.model.counters
        lat = self.stats.latencies_ms
        return {
            "packets": c.packets, "forwarded": c.forwarded, "unknown": c.unknown, "orphans": c.orphans,
            "adapted": c.adapted, "calls": c.calls, "rearms": c.rearms, "errors": self.stats.errors,
            "device": self.sink.device.name if self.sink.device else None,
            "cap_percent": self.cap,
            "failure": str(self.failure) if self.failure is not None else None,
            "latency_ms": {
                "n": self.stats.latency_count, "window": len(lat), "p50": percentile(lat, 50),
                "p95": percentile(lat, 95), "p99": percentile(lat, 99), "max": self.stats.latency_max,
            },
            "unknown_log": [f"{u.reason}: {u.data.hex()}" for u in self.model.unknown_log],
            "error_log": self.stats.error_log,
            "device_events": self.stats.device_events,
        }

    # ---- thread

    def _run(self) -> None:
        try:
            self._try_open(self.clock())
            while not self._stop.is_set():
                self._step()
        except BaseException as e:  # stop forwarding on anything unexpected
            self.failure = e
            self._event(f"bridge failed: {e!r}")
        finally:
            self._close()

    def _step(self) -> None:
        now = self.clock()
        wait = IDLE_WAIT_S
        due = self.model.next_rearm() if self.device_open else None
        if due is not None:
            wait = min(wait, max(0.0, due - now))
        packet = self.source.get(wait)
        if packet is not None:
            self._forward(packet)
        now = self.clock()
        if self.device_open:
            self._execute(self.model.rearm(now))
        elif now >= self._reopen_at:
            self._try_open(now)

    def _forward(self, packet: Packet) -> None:
        record = fm.decode_packet(packet.cmd, packet.data)
        problems = crosscheck(record, packet.dll)
        if problems:
            self.source.note_mismatch(problems, packet)
        calls = self.model.apply(record, self.clock())
        if self.device_open:
            self._execute(calls)
            self.stats.add_latency((self.clock() - packet.t) * 1000)

    def _execute(self, calls: list[fm.DiCall]) -> None:
        for call in calls:
            if not self.device_open:
                return
            if call.op == "SetGain":
                call = fm.DiCall("SetGain", value=capped_gain(call.value, self.cap))
            hr = self.sink.execute(call)
            if call.op == "Command" and call.flags == fm.DISFFC_RESET and not di.failed(hr):
                # A reset may restore the wheel's own gain; put the cap back.
                call = fm.DiCall("SetGain", value=capped_gain(self.model.device_gain, self.cap))
                hr = self.sink.execute(call)
            if hr in di.DEVICE_LOST:
                # Reopening sets the capped gain before anything else plays.
                self._event(f"wheel lost: {di.hresult_name(hr)}")
                self._close()
                self._reopen_at = self.clock() + self.reopen_s
                return
            if di.failed(hr) and call.op == "SetGain":
                # `_run` closes the wheel, which stops and unloads every effect.
                raise CapNotApplied(f"wheel refused the capped gain {call.value}: {di.hresult_name(hr)}")
            if di.failed(hr):
                self.stats.errors += 1
                if len(self.stats.error_log) < ERROR_LOG_MAX:
                    self.stats.error_log.append(f"{call.text()} -> {di.hresult_name(hr)}")

    def _try_open(self, now: float) -> None:
        self._reopen_at = now + self.reopen_s
        if not self.attended:
            self._report_devices()
            return
        gain = capped_gain(self.model.device_gain, self.cap)
        try:
            opened = self.sink.open(gain)
        except di.GainRefused as e:
            raise CapNotApplied(f"wheel refused the capped gain {gain}: {di.hresult_name(e.hr)}") from e
        if opened:
            self.device_open = True
            dev = self.sink.device
            self._event(f"wheel open: {dev.name if dev else '?'}")
            self._execute(self.model.replay(self.clock()))

    def _report_devices(self) -> None:
        names = tuple(f"{d.name} ({d.vid:04X}:{d.pid:04X})" for d in (self.list_devices() if self.list_devices else [])
                      if not d.is_vjoy)
        if names != self._last_seen:
            self._last_seen = names
            if names:
                self._event(f"wheel present but run is unattended, not opening it: {', '.join(names)}")
            else:
                self._event("no force feedback wheel attached; waiting")

    def _close(self) -> None:
        if self.device_open:
            self.device_open = False
            try:
                self.sink.close()
            except OSError as e:
                self._event(f"close failed: {e!r}")

    def _event(self, text: str) -> None:
        self.stats.device_events.append(f"{self.clock():.3f} {text}")
        self.say(text)


# --------------------------------------------------------------------------
# CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m simsteer.runtime.ffb.bridge",
        description="Forward the game's force feedback from vJoy to the Fanatec wheel (test build).",
    )
    p.add_argument("--ffb-test-cap", type=float, metavar="PERCENT",
                   help=f"required: device gain ceiling on the wheel, {CAP_MIN:g} to {CAP_MAX:g} percent")
    p.add_argument("--attended", action="store_true",
                   help="someone is at the wheel: open it and forward. Without this the bridge only reports")
    p.add_argument("--vjoy-device", type=int, default=1, help="vJoy device the game uses (default 1)")
    p.add_argument("--dry-run", action="store_true", help="log the calls instead of making them")
    p.add_argument("--log", metavar="FILE", help="with --dry-run: write the call log here")
    p.add_argument("--summary", metavar="FILE", help="write counters and latency as JSON on exit")
    p.add_argument("--seconds", type=float, default=0, help="stop after this many seconds (default: run until Ctrl+C)")
    p.add_argument("--check", action="store_true", help="print the vJoy force feedback check and exit")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.check:
        if args.ffb_test_cap is None:
            print("Refusing to start: --ffb-test-cap PERCENT is required in this build (1 to 15).", file=sys.stderr)
            return 2
        try:
            cap = validate_cap(args.ffb_test_cap)
        except ValueError as e:
            print(f"Refusing to start: {e}", file=sys.stderr)
            return 2
    if sys.platform != "win32":
        print("The FFB bridge needs Windows (vJoy and DirectInput).", file=sys.stderr)
        return 2
    from simsteer.runtime.ffb import vjoy_ffb

    if args.check:
        check = vjoy_ffb.vjoy_ffb_check(vjoy_ffb.probe_vjoy(device_id=args.vjoy_device), args.vjoy_device)
        print(f"vJoy FFB: {'OK' if check.ok else 'FAIL'} {check.row}" + (f". Fix: {check.fix}" if check.fix else ""))
        return 0 if check.ok else 1

    rx = vjoy_ffb.VJoyFfbReceiver(args.vjoy_device)
    try:
        check = rx.open()
    except vjoy_ffb.VJoyError as e:
        print(f"vJoy FFB: {e}. Fix: install vJoy 2.2.2.0 (BrunnerInnovation).", file=sys.stderr)
        return 1
    if not check.ok:
        print(f"vJoy FFB: {check.row}." + (f" Fix: {check.fix}" if check.fix else ""), file=sys.stderr)
        rx.close()
        return 1
    print(f"vJoy FFB: {check.row}", flush=True)

    sink: di.FfbSink = di.LogSink() if args.dry_run else di.DirectInputSink()
    bridge = FfbBridge(rx, sink, cap, attended=args.attended or args.dry_run,
                       list_devices=di.attached_ff_devices,
                       say=lambda line: print(line, flush=True))
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, lambda *_: stop.set())
    bridge.start()
    print(f"bridge running, cap {cap:g} percent" + (", dry run" if args.dry_run else ""), flush=True)
    t0 = time.monotonic()
    last_status = ""
    try:
        while not stop.is_set() and bridge.running:
            if args.seconds and time.monotonic() - t0 >= args.seconds:
                break
            stop.wait(0.5)
            if bridge.status != last_status:
                last_status = bridge.status
                print(last_status, flush=True)
    finally:
        bridge.stop()
        rx.close()
    summary = bridge.summary()
    summary["receiver"] = {"received": rx.stats.received, "other_device": rx.stats.other_device,
                           "dll_mismatches": rx.stats.mismatches, "mismatch_log": rx.stats.mismatch_log}
    if args.dry_run and args.log:
        with open(args.log, "w", encoding="utf-8") as f:
            f.write("\n".join(sink.lines()) + "\n")
    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("_log")}, indent=1))
    return 1 if bridge.failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
