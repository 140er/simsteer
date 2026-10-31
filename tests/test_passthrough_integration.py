"""The physical relay shares the output manager's safety boundary."""
import threading
import time
from types import SimpleNamespace

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
from simsteer.runtime.output.wheel import WheelError
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

    def guarded(self, writer, **kwargs):
        def write(backend):
            previous = getattr(active, "gate", None)
            active.gate = self._output_lock
            try:
                return writer(backend)
            finally:
                active.gate = previous
        return original(self, write, **kwargs)

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
            sim.wheel.set(steer_deg=45, brake=.8)
            for _ in range(3):
                tick(io)
                assert sim.game.steer() == pytest.approx(.1, abs=.0001)
                assert sim.game.pedal("z") == pytest.approx(.8, abs=.0001)
                assert sim.game.pedal("y") == 0
                assert not manager.engaged
            sim.wheel.set(brake=0)
            tick(io)
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
        reader.physical = Physical(.1, .2, .8, .4, True)
        start = len(dll.calls)
        release.set()
        tick(wheel.io)
        for usage, expected in ((0x30, 18022), (0x31, 6554), (0x32, 26214), (0x33, 13107)):
            values = [r[1][0] for r in dll.calls[start:] if r[0] == "SetAxis" and r[1][2] == usage]
            assert values and all(abs(value - expected) <= 1 for value in values)
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


@pytest.mark.parametrize("fault", ["capture", "telemetry", "nan_idle", "nan_ai", "shutdown"])
def test_loop_fault_preserves_player_controls(monkeypatch, tmp_path, fault):
    import builtins
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
        sim.wheel.set(steer_deg=15, brake=0 if fault == "nan_ai" else .8, throttle=.2, clutch=.4)
        tick(io)
        clip, clock = _make_clip(tmp_path / "clip", frames=4), ReplayClock()
        snapshots, frame, saves, logs = [], [-1], [], []
        log = tmp_path / "run.jsonl"

        def open_log(path, *args, **kwargs):
            handle = builtins.open(path, *args, **kwargs)
            logs.append(handle)
            return handle

        for cls, method in ((loop.LiveParams, "save"), (loop.LiveCalib, "save_state"),
                            (loop.Calibration, "save"), (loop.CalibFlow, "save")):
            original = getattr(cls, method)

            def save(self, *args, _save=original, _name=cls.__name__, **kwargs):
                result = _save(self, *args, **kwargs)
                saves.append(_name)
                return result

            monkeypatch.setattr(cls, method, save)

        class Source(ReplaySource):
            closed = False

            def __exit__(self, *args):
                self.closed = True
                return super().__exit__(*args)

            def grab(self):
                tick(io)
                snapshots.append((manager.engaged, sim.game.steer(), sim.game.pedal("y"),
                                  sim.game.pedal("z"), sim.game.pedal("rx")))
                frame[0] += 1
                if frame[0] == 1:
                    sim.wheel.set(steer_deg=45, brake=.8)
                value = super().grab()
                if self.done:
                    saves.clear()
                return None if fault == "capture" and frame[0] == 1 else value

        class Telemetry(ReplayTelemetry):
            closed = False

            @property
            def available(self):
                return not (fault == "telemetry" and frame[0] == 1) and super().available

            def close(self):
                self.closed = True
                super().close()

        monkeypatch.setattr(loop, "DeviceManager", lambda **kw: manager)
        monkeypatch.setattr(loop, "CalibFlow", ReadyFlow)
        monkeypatch.setattr(loop, "open", open_log, raising=False)
        monkeypatch.setattr(loop.LateralController, "compute",
                            lambda *a, **kw: float("nan") if fault.startswith("nan") and frame[0] == 1 else .3)
        monkeypatch.setattr(loop.LongitudinalController, "compute", lambda *a, **kw: (.6, 0))
        shutdown = manager.shutdown_output
        if fault == "shutdown":
            def failed_shutdown():
                shutdown()
                raise WheelError("Wheel I/O did not stop")
            monkeypatch.setattr(manager, "shutdown_output", failed_shutdown)
        source, telemetry = Source(clip, clock), Telemetry(clip, clock)

        def run():
            return loop.run(Settings(device="wheel", no_tuner=True, no_probe=True,
                                     force_engage=fault == "nan_ai"),
                            "ets2", source=source, clock=clock, model=FakeModel(),
                            make_telemetry=lambda: telemetry, headless=True, log_path=str(log))

        try:
            if fault == "shutdown":
                with pytest.raises(WheelError, match="did not stop"):
                    run()
            else:
                assert run() == 0
        finally:
            shutdown()
        if fault == "nan_ai":
            assert snapshots[1][0] and snapshots[1][2] == pytest.approx(.6, abs=.0001)
        for engaged, steer, throttle, brake, clutch in snapshots[2:]:
            assert not engaged
            assert (steer, throttle, brake, clutch) == pytest.approx((.1, .2, .8, .4), abs=.0001)
        assert set(saves) == {"LiveParams", "LiveCalib", "Calibration", "CalibFlow"}
        assert source.closed and telemetry.closed and logs and all(f.closed for f in logs)
        assert manager._closed and not io.is_alive()
        sim.assert_no_leaks()
        sim.assert_single_thread()
        sim.assert_no_force_writes()


def test_shutdown_timeout_still_latches_and_allows_owner_neutral_cleanup(monkeypatch):
    import atexit
    entered, release, registered = threading.Event(), threading.Event(), []
    monkeypatch.setattr(atexit, "register", registered.append)
    monkeypatch.setattr(atexit, "unregister", lambda callback: registered.remove(callback))

    class Reader(ScriptedPhysical):
        block = False

        def read(self, now):
            if self.block:
                entered.set()
                assert release.wait(2)
            return super().read(now)

    reader, dll = Reader(Physical(0, 0, 0, 0, True)), FakeVJoyDll()
    monkeypatch.setattr(passthrough, "PhysicalWheel", lambda _: reader)
    monkeypatch.setattr(passthrough, "VJoyFeeder", lambda _: VJoyFeeder(dll=dll))
    manager = DeviceManager("wheel", passthrough=True)
    io = manager.device.io
    try:
        assert manager.engage()
        tick(io)
        manager.set_throttle_brake(.8, 0)
        tick(io)
        assert dll.values[0x31] > 1
        reader.block = True
        assert entered.wait(.5)
        with pytest.raises(WheelError, match="did not stop"):
            manager.shutdown_output()
        assert manager._closed and manager._shutdown_requested
        assert manager.shutdown_output not in registered
        before = len(dll.calls)
        manager.engage()
        manager.set_throttle_brake(1, 1, force=True)
        manager.write_output(lambda dev: dev.set_steering(1, force=True))
        manager.write_output(lambda dev: pytest.fail("cleanup bypassed the closed latch"), cleanup=True)
        assert len(dll.calls) == before
    finally:
        release.set()
        io.join(.5)
        manager.device.close()
        manager.shutdown_output()
    assert not io.is_alive() and not registered
    assert dll.values == {0x30: 0x4000, 0x31: 1, 0x32: 1, 0x33: 1}
    assert dll.calls[-1][0] == "RelinquishVJD"
    assert {row[3] for row in dll.calls} == {io.ident}


@pytest.mark.parametrize("event", [2, 5, 6])
def test_console_handlers_chain_to_permanent_manager_shutdown(monkeypatch, event):
    import ctypes
    callbacks, called = [], []

    def set_handler(callback, enabled):
        callbacks.append(callback) if enabled else callbacks.remove(callback)
        return 1

    kernel = SimpleNamespace(SetConsoleCtrlHandler=set_handler)
    monkeypatch.setattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE, raising=False)
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **kw: kernel, raising=False)
    console = passthrough.ConsoleExit
    monkeypatch.setattr(passthrough, "ConsoleExit", lambda close: console(close, api=kernel))
    monkeypatch.setattr(passthrough, "sys", SimpleNamespace(
        platform="win32", getswitchinterval=passthrough.sys.getswitchinterval,
        setswitchinterval=passthrough.sys.setswitchinterval))
    with FanatecSim("dd2", live_instance=1) as sim:
        manager = simulated_manager(monkeypatch, sim)
        io = manager.device.io
        try:
            assert callbacks == [manager._console_handler, manager.device._console.callback]
            for callback in reversed(tuple(callbacks)):
                called.append(callback)
                if callback(event):
                    break
            assert len(called) == 2 and not callbacks
            assert manager._closed and not io.is_alive()
            assert sim.game.axes == {"x": 0x4000, "y": 1, "z": 1, "rx": 1}
        finally:
            manager.shutdown_output()
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
