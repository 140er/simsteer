"""Regressions for wheel re-engagement, identity changes and safe teardown."""
from dataclasses import replace
from pathlib import Path
import json
import threading
import time
from types import SimpleNamespace

import pytest

from fakes_wheel import FakeDi, FakeVJoyDll, ScriptedPhysical, profile
from simsteer.app.settings import Settings
from simsteer.core.override import Commands, Output, Override, Physical
from simsteer.runtime import wheel_session
from simsteer.runtime.ffb import dinput_out as di
from simsteer.runtime.input.physical import PhysicalWheel
from simsteer.runtime.output import passthrough
from simsteer.runtime.output.vjoy_feeder import VJoyFeeder


def test_reengage_does_not_replay_previous_ai_pedals():
    reader = ScriptedPhysical(Physical(0, 0, 0, 0, True))
    dll = FakeVJoyDll()
    wheel = passthrough.PassthroughWheel(Settings(), reader_factory=lambda: reader,
                                         feeder_factory=lambda: VJoyFeeder(dll=dll))
    try:
        wheel.engage()
        time.sleep(.02)
        wheel.set_steering(.3)
        wheel.set_throttle_brake(.8, .02)
        reader.physical = replace(reader.physical, brake=.5)
        time.sleep(.08)
        assert not wheel.engaged and wheel.last_reason == "brake"
        reader.physical = replace(reader.physical, brake=0)
        time.sleep(.01)
        start = len(dll.calls)
        wheel.engage()
        time.sleep(.06)
        pedals = [c[1][0] for c in dll.calls[start:] if c[0] == "SetAxis" and c[1][2] in (0x31, 0x32)]
        assert pedals and set(pedals) == {1}
        assert (wheel.io.commands.steer, wheel.io.commands.throttle, wheel.io.commands.brake) == (0, 0, 0)
        wheel.set_throttle_brake(.25, 0)
        time.sleep(.01)
        assert dll.values[0x31] > 1, "fresh AI commands must still work"
    finally:
        wheel.close()


def test_engage_starts_at_last_output_even_when_rim_moved():
    machine = Override()
    previous = machine.step(0, Physical(.2, ok=True), Commands())
    out = machine.step(.002, Physical(-.1, ok=True), Commands("engage", 1, ai_at=.002))
    assert out.steer == previous.steer


def test_instance_change_and_restart_cannot_restore_old_pedal_map(monkeypatch):
    old, new = profile(1), profile(2, {0: "Wheel Axis", 1: "Z Axis", 2: "Accelerator", 3: "Brake"})
    new.axes = [32768, 32767, 0, 0, 0, 0, 0, 0]
    settings = Settings(wheel_instance=str(old.inst.guidInstance), wheel_axes={
        "steer": 0, "throttle": {"slot": 1, "rest": 0, "full": 65535},
        "brake": {"slot": 2, "rest": 0, "full": 65535}})
    api = FakeDi(new)
    monkeypatch.setattr(di, "_dev", api.call)
    reader = PhysicalWheel(settings, di_factory=api)
    dll = FakeVJoyDll()
    wheel = passthrough.PassthroughWheel(settings, reader_factory=lambda: reader,
                                         feeder_factory=lambda: VJoyFeeder(dll=dll))
    try:
        time.sleep(.01)
        new.axes[0] += 1000
        time.sleep(.08)
        wheel.stats()
        saved = json.loads(Path(settings.path()).read_text())
        assert saved["wheel_instance"] == str(new.inst.guidInstance)
        assert not {"throttle", "brake", "clutch"} & saved["wheel_axes"].keys()
    finally:
        wheel.close()
    restarted = PhysicalWheel(Settings.load(), di_factory=api)
    try:
        assert restarted.open()
        sample = restarted.read(1)
        assert sample.throttle is sample.brake is None
    finally:
        restarted.close()


@pytest.mark.parametrize("slot,rest,full,value", [
    (1, 0, 65535, 5000), (2, 0, 65535, 5000), (3, 32767, 0, 30000),
    (1, 0, 65535, 32767), (1, 32767, 65535, 29000),
])
def test_every_mapped_pedal_must_reach_rest_before_startup_relay(monkeypatch, slot, rest, full, value):
    p = profile()
    settings = Settings(wheel_instance=str(p.inst.guidInstance), wheel_axes={
        "steer": 0, "throttle": {"slot": 1, "rest": 0, "full": 65535},
        "brake": {"slot": 2, "rest": 0, "full": 65535},
        "clutch": {"slot": 3, "rest": 32767, "full": 0}})
    name = {1: "throttle", 2: "brake", 3: "clutch"}[slot]
    settings.wheel_axes[name] = dict(slot=slot, rest=rest, full=full)
    p.axes[slot] = value
    api = FakeDi(p)
    monkeypatch.setattr(di, "_dev", api.call)
    reader = PhysicalWheel(settings, di_factory=api)
    try:
        assert reader.open()
        sample = reader.read(0)
        assert sample.throttle is sample.brake is sample.clutch is None
        assert reader.status == "Check pedals"
        p.axes[slot] = rest
        sample = reader.read(.002)
        assert reader.status != "Check pedals" and sample.brake is not None
        p.axes[slot] = full
        assert getattr(reader.read(.004), name) == 1
    finally:
        reader.close()


@pytest.mark.parametrize("speed,seconds", [(4, 5), (5.9, 5), (1, 35)])
def test_slow_takeover_and_absolute_hard_limit_cannot_be_absorbed(speed, seconds):
    machine = Override()
    machine.step(0, Physical(ok=True), Commands("engage", 1))
    for i in range(1, round((seconds + 1) * 500) + 1):
        now = i / 500
        degrees = max(0, now - 1) * speed
        out = machine.step(now, Physical(degrees / 450, ok=True), Commands("engage", 1, ai_at=now))
        if not out.engaged:
            break
    assert not out.engaged and out.reason == "steering"
    assert degrees <= 30.01


def test_close_failure_never_introduces_midpoint_pedals():
    dll = FakeVJoyDll()
    feeder = VJoyFeeder(dll=dll)
    dll.fail = 0x31
    with pytest.raises(Exception, match="write failed"):
        feeder.close()
    assert dll.values == {0x30: 0x4000, 0x31: 1, 0x32: 1, 0x33: 1}
    assert not any(c[0] == "ResetVJD" for c in dll.calls)
    assert dll.calls[-1][0] == "RelinquishVJD"


@pytest.mark.parametrize("status,reason", [("", "map pedals"), ("Check pedals", "check pedals"),
                                        ("Press brake fully", "map pedals")])
def test_engage_requires_working_brake_takeover(status, reason):
    pad = SimpleNamespace(physical=Physical(ok=True), pedal_status=status)
    assert wheel_session.engage_block(pad) == reason


def test_unconfirmed_instance_and_single_glitch_never_reach_output(monkeypatch):
    a, b = profile(1), profile(2)
    a.axes[0] = 60000
    api = FakeDi(a, b)
    monkeypatch.setattr(di, "_dev", api.call)
    reader = PhysicalWheel(di_factory=api)
    try:
        assert reader.open()
        assert not reader.read(0).ok
        b.axes[0] += 1000
        assert not reader.read(.002).ok
        b.axes[0] -= 1000
        assert not reader.read(.004).ok and not reader.instance_proven
        b.axes[0] += 1000
        reader.read(.01)
        sample = reader.read(.07)
        assert sample.ok and reader.instance_proven == str(b.inst.guidInstance)
        machine = Override()
        machine.step(0, Physical(), Commands())
        assert machine.step(.07, sample, Commands()).steer == 0
        assert machine.step(2.08, sample, Commands()).steer == sample.steer
    finally:
        reader.close()


@pytest.mark.parametrize("event", [2, 5, 6])
def test_console_close_logoff_shutdown_neutralize_on_owner(event):
    dll, reader = FakeVJoyDll(), ScriptedPhysical(Physical(0, 0, 0, 0, True))
    wheel = passthrough.PassthroughWheel(Settings(), reader_factory=lambda: reader,
                                         feeder_factory=lambda: VJoyFeeder(dll=dll))
    registrations = []
    api = SimpleNamespace(SetConsoleCtrlHandler=lambda callback, enabled: registrations.append((callback, enabled)) or 1)
    try:
        wheel._console = passthrough.ConsoleExit(wheel.close, api=api)
        wheel.set_throttle_brake(.8, 0, force=True)
        time.sleep(.01)
        results = []
        handler = registrations[0][0]
        assert handler(0) == 0  # Leave Ctrl+C to Python's normal exit path.
        thread = threading.Thread(target=lambda: results.append(handler(event)))
        thread.start()
        thread.join(.5)
        assert results == [1] and not wheel.io.is_alive()
        assert dll.values == {0x30: 0x4000, 0x31: 1, 0x32: 1, 0x33: 1}
        assert {c[3] for c in dll.calls} == {wheel.io.ident}
        assert registrations[-1][1] == 0
    finally:
        wheel.close()
