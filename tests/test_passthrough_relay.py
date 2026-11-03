from dataclasses import replace
import json
from pathlib import Path
import sys
import threading
import time

import pytest

from simsteer.app.settings import Settings
from simsteer.core.override import Physical
from simsteer.runtime.output.passthrough import PassthroughWheel, percentile
from simsteer.runtime.output.vjoy_feeder import VJoyFeeder
from fakes_wheel import FakeVJoyDll, ScriptedPhysical


def relay(reader=None, dll=None):
    reader, dll = reader or ScriptedPhysical(Physical(0, 0, 0, 0, True)), dll or FakeVJoyDll()
    wheel = PassthroughWheel(Settings(), reader_factory=lambda: reader,
                             feeder_factory=lambda: VJoyFeeder(dll=dll))
    return wheel, reader, dll


def test_rate_latency_single_writer_and_close(record_property):
    interval = sys.getswitchinterval()
    wheel, reader, dll = relay(ScriptedPhysical(Physical(.1, .2, .3, .4, True)))
    try:
        threading.Event().wait(.15)
        start, before = time.monotonic(), wheel.io.ticks
        while time.monotonic() - start < .2:  # Deliberately contend for the GIL.
            sum(i * i for i in range(100))
        assert wheel.io.ticks - before >= 20, "relay stalled with model thread"
        stats = wheel.stats()
        record_property("relay_hz", stats["relay_hz"])
        record_property("latency_p95_ms", stats["latency_p95_ms"])
        assert stats["relay_hz"] >= 100  # Loaded runners use a functional floor.
        assert stats["latency_p95_ms"] <= 5
        assert stats["latency_max_ms"] >= stats["latency_p95_ms"]
        assert sys.getswitchinterval() == .001
    finally:
        start = time.monotonic()
        wheel.close()
    assert time.monotonic() - start < .5
    assert sys.getswitchinterval() == interval
    assert reader.opened_thread == reader.closed_thread == wheel.io.ident
    assert {c[3] for c in dll.calls} == {wheel.io.ident}
    assert dll.values == {0x30: 0x4000, 0x31: 1, 0x32: 1, 0x33: 1}
    assert dll.calls[-1][0] == "RelinquishVJD"


def test_reader_exception_neutralizes_and_thread_survives():
    wheel, reader, dll = relay()
    try:
        reader.reads.extend([RuntimeError("scripted reader error")] + [Physical()] * 30)
        threading.Event().wait(.025)
        assert wheel.io.is_alive() and wheel.source == "LOST"
        assert dll.values == {0x30: 0x4000, 0x31: 1, 0x32: 1, 0x33: 1}
    finally:
        wheel.close()


def test_stalled_ai_hands_back_on_relay_and_center_only_clears_overrides():
    wheel, reader, dll = relay()
    try:
        wheel.engage()
        threading.Event().wait(.015)
        assert wheel.engaged
        wheel.set_steering(.7)
        wheel.set_throttle_brake(.8, 0)
        threading.Event().wait(.3)
        assert not wheel.engaged and wheel.last_reason == "stale"
        assert dll.values[0x31] == 1
        wheel.set_steering(.9, force=True)
        threading.Event().wait(.01)
        before = wheel.io.output.steer
        wheel.center()
        threading.Event().wait(.01)
        assert abs(wheel.io.output.steer - before) < .02
    finally:
        wheel.close()


def test_ui_commands_never_write_vjoy_and_runtime_overrides_expire():
    wheel, reader, dll = relay()
    try:
        def tuner():
            wheel.set_steering(.4, force=True)
            wheel.set_throttle_brake(.2, .1, force=True)
        thread = threading.Thread(target=tuner)
        thread.start()
        thread.join()
        threading.Event().wait(.02)
        assert wheel.source == "OVERRIDE"
        assert {c[3] for c in dll.calls} == {wheel.io.ident}
        threading.Event().wait(.22)
        assert wheel.source == "HANDBACK" and dll.values[0x31] == 1
    finally:
        wheel.close()


def test_mapping_and_instance_persistence_happen_outside_io_thread():
    wheel, reader, dll = relay()
    try:
        reader.instance_proven = "00000001-0000-0000-0000-000000000000"
        threading.Event().wait(.01)
        wheel.stats()
        assert json.loads(Path(wheel.settings.path()).read_text())["wheel_instance"] == reader.instance_proven
        assert wheel.map_pedals()
        threading.Event().wait(.01)
        assert reader.status == "Press throttle fully"
    finally:
        wheel.close()


def test_latency_percentiles_from_known_samples():
    assert percentile([], .95) == 0
    assert percentile(range(1, 101), .95) == 95
    wheel, _, _ = relay()
    try:
        with wheel.io.lock:
            wheel.io.latencies.clear()
            wheel.io.latencies.extend(range(1, 101))
            wheel.io.gap_max = 7
        s = wheel.stats(consume=True)
        assert (s["latency_p95_ms"], s["latency_p99_ms"], s["latency_max_ms"], s["relay_gap_max_ms"]) == (95, 99, 100, 7)
    finally:
        wheel.close()
