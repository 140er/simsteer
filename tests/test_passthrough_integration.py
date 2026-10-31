"""The physical relay shares the output manager's safety boundary."""
import threading
import time

import pytest

from fakes_wheel import FakeVJoyDll, ScriptedPhysical
from simsteer.app.settings import Settings
from simsteer.core.finite import NonFiniteError
from simsteer.core.override import Physical
from simsteer.runtime.input.physical import PhysicalWheel
from simsteer.runtime.output import passthrough
from simsteer.runtime.output.base import DeviceManager
from simsteer.runtime.output.safety import OutputGate
from simsteer.runtime.output.vjoy_feeder import VJoyFeeder
from simsteer.testing.fanatec import FanatecSim


def tick(io):
    done, before = threading.Event(), io.ticks

    def after(_):
        if io.ticks >= before + 2:
            done.set()

    io.hooks.append(after)
    try:
        assert done.wait(.5), io.error
    finally:
        io.hooks.remove(after)


def until(io, condition, post=lambda: None):
    deadline = time.monotonic() + 2.5
    while time.monotonic() < deadline:
        post()
        tick(io)
        if condition():
            return
    pytest.fail(f"relay did not reach expected state: {io.output}, {io.error}")


def simulated_manager(monkeypatch, sim):
    factory = PhysicalWheel
    monkeypatch.setattr(passthrough, "PhysicalWheel",
                        lambda settings: factory(settings, di_factory=sim.DirectInput))
    settings = Settings(device="wheel", wheel_instance=sim.wheel.instances[1].instance, wheel_axes={
        "steer": 0, "throttle": dict(slot=6, rest=0, full=65535),
        "brake": dict(slot=5, rest=0, full=65535), "clutch": dict(slot=7, rest=0, full=65535)})
    return DeviceManager("wheel", passthrough=True, settings=settings)


def test_simulator_takeover_reengage_and_nan_share_one_writer(monkeypatch):
    active, gates = threading.local(), []
    original = OutputGate.write_output

    def guarded(self, writer):
        def write(backend):
            previous = getattr(active, "gate", None)
            active.gate = self._output_lock
            try:
                return writer(backend)
            finally:
                active.gate = previous
        return original(self, write)

    monkeypatch.setattr(OutputGate, "write_output", guarded)
    with FanatecSim("dd2", live_instance=1) as sim:
        set_axis = sim.vjoy.SetAxis

        def checked(*args):
            gate = getattr(active, "gate", None)
            assert gate is not None and gate._is_owned(), "axis write bypassed write_output"
            gates.append(id(gate))
            return set_axis(*args)

        sim.vjoy.SetAxis = checked
        manager = simulated_manager(monkeypatch, sim)
        assert manager.passthrough
        io = manager.device.io
        try:
            until(io, lambda: manager.physical.ok)
            sim.wheel.set(steer_deg=8)
            tick(io)
            assert manager.engage()

            def ai():
                manager.set_steering(.04)
                manager.set_throttle_brake(.7, .02)

            until(io, lambda: manager.source == "AI", ai)
            assert sim.game.pedal("y") == pytest.approx(.7, abs=.0001)
            sim.wheel.set(brake=.8)
            until(io, lambda: not manager.engaged, ai)
            assert manager.last_reason == "brake"
            assert sim.game.pedal("z") == pytest.approx(.8, abs=.0001)
            serial = io.commands.serial
            manager.insert_disengage()  # The loop can observe the same edge later.
            assert io.commands.serial == serial and manager.last_reason == "brake"
            until(io, lambda: manager.source == "PLAYER")
            sim.wheel.set(brake=0)
            tick(io)
            assert manager.engage()
            tick(io)
            assert io.commands.throttle == io.commands.brake == 0
            assert sim.game.pedal("y") == sim.game.pedal("z") == 0
            ai()
            tick(io)
            assert sim.game.pedal("y") > .6
            with pytest.raises(NonFiniteError):
                manager.set_steering(float("nan"))
            for _ in range(3):
                tick(io)
                assert sim.game.axes == {"x": 0x4000, "y": 1, "z": 1, "rx": 1}
                assert not manager.engaged
            manager.resume_output()
            until(io, lambda: manager.source == "PLAYER")
            assert manager.engage()
            tick(io)
            assert sim.game.pedal("y") == sim.game.pedal("z") == 0
        finally:
            manager.shutdown_output()
        before = len(sim.vjoy.calls)
        manager.engage()
        manager.set_steering(.9, force=True)
        manager.write_output(lambda dev: dev.set_throttle_brake(1, 1, force=True))
        assert len(sim.vjoy.calls) == before and manager._closed
        assert sim.game.axes == {"x": 0x4000, "y": 1, "z": 1, "rx": 1}
        writes = [row for row in sim.vjoy.calls if row[0] in ("SetAxis", "SetBtn", "AcquireVJD", "RelinquishVJD")]
        assert {row[2] for row in writes} == {io.ident}
        assert gates and set(gates) == {id(manager._output_lock)}
        sim.assert_single_thread()
        sim.assert_no_leaks()
        sim.assert_no_force_writes()


def test_stop_during_a_read_cannot_replay_a_cached_command_after_resume():
    entered, release = threading.Event(), threading.Event()

    class Reader(ScriptedPhysical):
        block = False

        def read(self, now):
            if self.block:
                self.block = False
                entered.set()
                assert release.wait(1)
            return super().read(now)

    reader, dll = Reader(Physical(0, 0, 0, 0, True)), FakeVJoyDll()
    wheel = passthrough.PassthroughWheel(Settings(), reader_factory=lambda: reader,
                                         feeder_factory=lambda: VJoyFeeder(dll=dll))
    try:
        tick(wheel.io)
        assert wheel.engage()
        tick(wheel.io)
        wheel.set_throttle_brake(.8, 0)
        tick(wheel.io)
        reader.block = True
        assert entered.wait(.5)
        wheel.stop_output()
        wheel.resume_output()
        start = len(dll.calls)
        release.set()
        tick(wheel.io)
        pedals = [r[1][0] for r in dll.calls[start:] if r[0] == "SetAxis" and r[1][2] in (0x31, 0x32, 0x33)]
        assert pedals and set(pedals) == {1}
        assert not wheel.engaged
    finally:
        release.set()
        wheel.close()


def test_relay_release_for_swap_keeps_permanent_shutdown_separate(monkeypatch):
    from test_output_safety import gamepad
    import simsteer.runtime.output.base as devices
    with FanatecSim("dd2", live_instance=1) as sim:
        manager = simulated_manager(monkeypatch, sim)
        old = manager.device.io
        monkeypatch.setattr(devices, "Gamepad", gamepad)
        try:
            assert manager.set_kind("gamepad")
            assert not old.is_alive() and not manager._closed
            assert manager.kind == "gamepad"
            manager.engage()
            manager.set_steering(.4)
            assert manager.raw.visible[0] == .4
        finally:
            manager.shutdown_output()
        assert manager._closed and not manager.set_kind("wheel")
        sim.assert_no_leaks()
        sim.assert_no_force_writes()


def test_loop_nan_neutralizes_even_live_physical_controls(monkeypatch, tmp_path):
    import simsteer.runtime.loop as loop
    from test_replay import FakeModel, _make_clip
    from simsteer.runtime.replay import ReplayClock, ReplaySource, ReplayTelemetry

    class ReadyFlow(loop.CalibFlow):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.steering_fit = False
            self.step = loop.Step.DONE
            self.lc.cal_status = loop.CalStatus.CALIBRATED

    with FanatecSim("dd2", live_instance=1) as sim:
        manager = simulated_manager(monkeypatch, sim)
        io = manager.device.io
        until(io, lambda: manager.physical.ok)
        sim.wheel.set(steer_deg=15, brake=.08)
        tick(io)
        clip, clock = _make_clip(tmp_path / "clip", frames=4), ReplayClock()
        snapshots, frame = [], [-1]

        class Source(ReplaySource):
            def grab(self):
                tick(io)
                snapshots.append((manager.engaged, sim.game.axes))
                frame[0] += 1
                return super().grab()

        monkeypatch.setattr(loop, "DeviceManager", lambda **kw: manager)
        monkeypatch.setattr(loop, "CalibFlow", ReadyFlow)
        monkeypatch.setattr(loop.LateralController, "compute",
                            lambda *a, **kw: float("nan") if frame[0] == 1 else .3)
        monkeypatch.setattr(loop.LongitudinalController, "compute", lambda *a, **kw: (.6, 0))
        try:
            assert loop.run(Settings(device="wheel", no_tuner=True, no_probe=True, force_engage=True),
                            "ets2", source=Source(clip, clock), clock=clock, model=FakeModel(),
                            make_telemetry=lambda: ReplayTelemetry(clip, clock), headless=True) == 0
        finally:
            manager.shutdown_output()
        assert snapshots[1][0] and snapshots[1][1]["y"] > 1
        assert snapshots[2] == (False, {"x": 0x4000, "y": 1, "z": 1, "rx": 1})
        assert manager._closed and not io.is_alive()
        sim.assert_no_leaks()
        sim.assert_single_thread()
        sim.assert_no_force_writes()


def test_explicit_binding_can_resume_a_fault_stopped_relay(monkeypatch):
    with FanatecSim("dd2", live_instance=1) as sim:
        manager = simulated_manager(monkeypatch, sim)
        io = manager.device.io
        try:
            until(io, lambda: manager.physical.ok)
            manager.disengage()
            tick(io)
            assert manager.request_bind("steering")
            assert manager.tick_bind(0, 0)
            assert manager.tick_bind(4.1, 0)
            tick(io)
            assert sim.game.steer() == 1
        finally:
            manager.shutdown_output()
        sim.assert_no_leaks()
        sim.assert_no_force_writes()
