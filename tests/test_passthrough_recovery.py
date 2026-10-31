"""Regressions for held pedals during recovery and the first engage command."""
import threading
import time

import pytest

from fakes_wheel import FakeDi, FakeVJoyDll, ScriptedPhysical, profile
from simsteer.app.settings import Settings
from simsteer.core.override import Commands, Override, Physical
from simsteer.runtime import wheel_session
from simsteer.runtime.ffb import dinput_out as di
from simsteer.runtime.input.physical import PhysicalWheel
from simsteer.runtime.output.passthrough import PassthroughWheel
from simsteer.runtime.output.vjoy_feeder import VJoyFeeder


def mapped_reader(monkeypatch):
    p = profile()
    p.axes[1:4] = [0, 0, 0]
    axes = {name: dict(slot=slot, rest=0, full=65535)
            for slot, name in enumerate(("throttle", "brake", "clutch"), 1)}
    settings = Settings(wheel_instance=str(p.inst.guidInstance), wheel_axes={"steer": 0, **axes})
    api = FakeDi(p)
    monkeypatch.setattr(di, "_dev", api.call)
    reader = PhysicalWheel(settings, di_factory=api)
    assert reader.open()
    return reader, api, p


@pytest.mark.parametrize("failures", [1, 3])
@pytest.mark.parametrize("startup_held", [False, True])
def test_brake_survives_recovery_without_rearming_pedal_hold(monkeypatch, failures, startup_held):
    reader, api, p = mapped_reader(monkeypatch)
    machine, dll = Override(), FakeVJoyDll()
    feeder = VJoyFeeder(dll=dll)

    def tick(now):
        sample = reader.read(now)
        out = machine.step(now, sample, Commands())
        feeder.write(out)
        return sample, out

    try:
        if startup_held:
            p.axes[1:4] = [52428] * 3
        tick(0)
        p.axes[1:4] = [52428] * 3
        sample, out = tick(.002)
        assert out.brake == pytest.approx(.8)
        p.reads.append(di.DIERR_INPUTLOST)
        sample, out = tick(.004)
        assert not sample.ok
        assert (out.steer, out.throttle, out.brake, out.clutch) == (0, 0, 0, 0)
        if failures == 3:
            p.acquire_errors.extend([di.DIERR_INPUTLOST] * 2)
            assert not tick(.504)[0].ok
            assert not tick(1.004)[0].ok
        sample, out = tick(1.504)
        assert sample.ok and out.brake == pytest.approx(.8)
        assert dll.values[0x32] == round(1 + .8 * 32767)
        if startup_held:
            assert sample.throttle is sample.clutch is None
            assert reader.status == "Check pedals"
        else:
            assert sample.throttle == sample.clutch == pytest.approx(.8)
            assert reader.status != "Check pedals"
        assert api.enumerations == (2 if failures == 3 else 1)
        p.axes[1:4] = [0] * 3
        tick(1.506)
        p.axes[1:4] = [52428] * 3
        assert tick(1.508)[1].throttle == pytest.approx(.8)
    finally:
        reader.close()
        feeder.close()


def test_missing_device_during_reconnect_preserves_its_confirmed_map(monkeypatch):
    reader, api, p = mapped_reader(monkeypatch)
    try:
        reader.read(0)
        p.axes[1:4] = [52428] * 3
        p.reads.append(di.DIERR_INPUTLOST)
        p.acquire_errors.extend([di.DIERR_INPUTLOST] * 2)
        for now in (.002, .502, 1.002):
            assert not reader.read(now).ok
        api.profiles = []
        assert not reader.read(1.502).ok
        api.profiles = [p]
        sample = reader.read(2.002)
        assert sample.ok and sample.brake == pytest.approx(.8)
        assert sample.throttle == sample.clutch == pytest.approx(.8)
    finally:
        reader.close()


def test_remapping_holds_throttle_and_clutch_but_preserves_mapped_brake(monkeypatch):
    reader, _, p = mapped_reader(monkeypatch)
    try:
        reader.read(0)
        p.axes[1:4] = [52428] * 3
        assert reader.read(.002).throttle == pytest.approx(.8)
        reader.map_pedals(.004)
        sample = reader.read(.006)
        assert sample.throttle is sample.clutch is None
        assert sample.brake == pytest.approx(.8)
    finally:
        reader.close()


def test_pressed_brake_alone_does_not_arm_startup_hold(monkeypatch):
    reader, _, p = mapped_reader(monkeypatch)
    try:
        p.axes[2] = 52428
        sample = reader.read(0)
        assert sample.brake == pytest.approx(.8)
        assert sample.throttle == sample.clutch == 0
        assert reader.status != "Check pedals"
    finally:
        reader.close()


def test_map_request_preserves_brake_on_every_relay_tick():
    reader, dll = ScriptedPhysical(Physical(0, .4, .8, .2, True)), FakeVJoyDll()
    wheel = PassthroughWheel(Settings(), reader_factory=lambda: reader,
                             feeder_factory=lambda: VJoyFeeder(dll=dll))
    mapped, braking = threading.Event(), []

    def tick(now):
        braking.append(dll.values[0x32])
        if reader.status == "Press throttle fully":
            mapped.set()

    try:
        time.sleep(.01)
        wheel.io.hooks.append(tick)
        assert wheel.map_pedals()
        assert mapped.wait(.5)
        assert braking and set(braking) == {round(1 + .8 * 32767)}
    finally:
        wheel.close()


@pytest.mark.parametrize("path", ["engage", "toggle"])
def test_engage_holds_current_output_until_first_fresh_ai_post(path):
    reader, dll = ScriptedPhysical(Physical(.2, 0, 0, 0, True)), FakeVJoyDll()
    wheel = PassthroughWheel(Settings(), reader_factory=lambda: reader,
                             feeder_factory=lambda: VJoyFeeder(dll=dll))
    try:
        time.sleep(.01)
        held = wheel.io.output.steer
        assert held == .2
        getattr(wheel, path)()
        cmd = wheel.io.commands
        assert cmd.steer == held and cmd.throttle == cmd.brake == 0
        machine = Override()
        machine.step(cmd.ai_at - .002, reader.physical, Commands())
        for dt in (0, .05, .125, .249, .3):
            out = machine.step(cmd.ai_at + dt, reader.physical, cmd)
            assert out.steer == held and out.throttle == out.brake == 0
        assert out.reason == "stale"
    finally:
        wheel.close()


@pytest.mark.parametrize("physical,status,reason", [
    (Physical(), "", "wheel not found"),
    (Physical(), "Confirming wheel", "Confirming wheel"),
    (Physical(ok=True), "", "map pedals"),
    (Physical(0, None, 0, None, True), "Check pedals", "check pedals"),
    (Physical(0, None, 0, None, True), "Release pedals, press brake fully", "map pedals"),
    (Physical(0, 0, .8, 0, True), "", "release the brake"),
])
def test_refused_engage_returns_false_and_a_short_reason(physical, status, reason):
    reader, dll = ScriptedPhysical(physical), FakeVJoyDll()
    reader.status = status
    wheel = PassthroughWheel(Settings(), reader_factory=lambda: reader,
                             feeder_factory=lambda: VJoyFeeder(dll=dll))
    try:
        time.sleep(.01)
        serial = wheel.io.commands.serial
        assert wheel.engage() is False
        assert wheel.toggle() is False
        assert wheel.io.commands.serial == serial and not wheel.engaged
        assert wheel.last_reason == reason
        assert wheel_session.engage_block(wheel) == reason
    finally:
        wheel.close()


def test_stats_and_concurrent_close_serialize_settings_saves():
    reader, dll = ScriptedPhysical(Physical(0, 0, 0, 0, True)), FakeVJoyDll()
    wheel = PassthroughWheel(Settings(), reader_factory=lambda: reader,
                             feeder_factory=lambda: VJoyFeeder(dll=dll))
    saving, release, overlapping = threading.Event(), threading.Event(), threading.Event()
    calls, errors, threads = [], [], []

    def save():
        calls.append(wheel.settings.wheel_axes)
        if len(calls) == 1:
            saving.set()
            assert release.wait(1)
        else:
            overlapping.set()

    def run(func):
        try:
            func()
        except Exception as error:
            errors.append(error)

    wheel.settings.save = save
    try:
        with wheel.io.lock:
            wheel.io.pending = {"wheel_axes": {"steer": 1}}
        threads.append(threading.Thread(target=run, args=(wheel.stats,)))
        threads[0].start()
        assert saving.wait(.5)
        with wheel.io.lock:
            wheel.io.pending = {"wheel_axes": {"steer": 2}}
        for _ in range(2):
            thread = threading.Thread(target=run, args=(wheel.close,))
            threads.append(thread)
            thread.start()
        assert not overlapping.wait(.05), "close saved settings while stats was still saving"
    finally:
        release.set()
        for thread in threads:
            thread.join(1)
        wheel.close()
    assert not errors and all(not t.is_alive() for t in threads)
    assert calls == [{"steer": 1}, {"steer": 2}]
    assert {c[3] for c in dll.calls} == {wheel.io.ident}
