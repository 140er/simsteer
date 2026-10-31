"""Drive the real loop headlessly, with scripted physical ownership."""
from dataclasses import replace
import json
import subprocess
from types import ModuleType

import pytest

from simsteer.app.settings import Settings
from simsteer.calib.flow import CalibFlow, Step
from simsteer.core.override import Physical
from simsteer.runtime import wheel_session
from simsteer.runtime.replay import ReplayClock, ReplaySource, ReplayTelemetry
from simsteer.ui.hud import HudState
from simsteer.ui.state import status_label
from test_replay import FakeModel, _make_clip


class FakeDeviceManager:
    kind, last_error = "wheel", None
    physical = Physical(.02, .1, 0, .2, True)
    engaged, last_reason, source = False, "", "PLAYER"

    def __init__(self, **kwargs):
        self.writes, self.closed = [], False

    @property
    def passthrough_live(self):
        return self.physical.ok and self.source == "PLAYER"

    def stats(self, **kwargs):
        return dict(relay_hz=500, relay_gap_max_ms=2, latency_p95_ms=1, pedal_status="")

    def engage(self):
        self.engaged, self.source = True, "AI"

    def disengage(self):
        self.engaged, self.source, self.last_reason = False, "HANDBACK", "loop"

    def center(self):
        pass

    def set_steering(self, value, **kwargs):
        self.writes.append(value)

    def set_throttle_brake(self, *args, **kwargs):
        pass

    def close(self):
        self.closed = True


def test_player_fit_takeover_edge_controller_reset_and_log(monkeypatch, tmp_path):
    import simsteer.runtime.loop as loop
    clip, clock, holder = _make_clip(tmp_path / "clip", frames=16), ReplayClock(), {}
    device = FakeDeviceManager()
    class Flow(CalibFlow):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.lc.cal_status = loop.CalStatus.CALIBRATED
            self._finish_fov("kept")
            self.direction.model_sign = self.direction.axis_sign = 1
            self.output_direction.sign = 1
            self.step = Step.STEERING
            holder["flow"] = self
    updates, resets, banners, labels = [], [], [], []
    real_update = loop.LiveParams.update
    def update(self, *a, **kw):
        updates.append(clock.now())
        return real_update(self, *a, **kw)
    real_lat_reset, real_long_reset = loop.LateralController.reset, loop.LongitudinalController.reset
    def lat_reset(self):
        resets.append(("lat", clock.now()))
        real_lat_reset(self)
    def long_reset(self):
        resets.append(("long", clock.now()))
        real_long_reset(self)
    def sound(name):
        if name == "disengage":
            banners.append(name)
    class Keys:
        frame = 0
        def poll(self):
            self.frame += 1
            if self.frame == 6:
                device.engage()
            if self.frame == 9:
                device.engaged, device.source, device.last_reason = False, "HANDBACK", "steering"
            labels.append(status_label(HudState(driver=wheel_session.driver(device), engaged=device.engaged)))
            return []
    monkeypatch.setattr(loop, "DeviceManager", lambda **kw: device)
    monkeypatch.setattr(loop, "CalibFlow", Flow)
    monkeypatch.setattr(loop.LiveParams, "update", update)
    monkeypatch.setattr(loop.LateralController, "reset", lat_reset)
    monkeypatch.setattr(loop.LongitudinalController, "reset", long_reset)
    monkeypatch.setattr(loop.audio, "play", sound)
    log = tmp_path / "loop.jsonl"
    assert loop.run(Settings(device="wheel", no_tuner=True), "ets2", source=ReplaySource(clip, clock),
                    make_telemetry=lambda: ReplayTelemetry(clip, clock), clock=clock, keys=Keys(),
                    model=FakeModel(), headless=True, log_path=str(log)) == 0
    rows = [json.loads(row) for row in log.read_text().splitlines()]
    assert updates and min(updates) < .25, "ETS2 must fit player driving through vJoy"
    assert {"Player", "AI"} <= set(labels)
    edges = [r for r in rows if r["takeover_reason"]]
    assert len(edges) == 1 and edges[0]["takeover_reason"] == "steering"
    edge_t = edges[0]["t"]
    assert {name for name, t in resets if t == pytest.approx(edge_t)} == {"lat", "long"}
    assert "disengage" in banners and device.closed
    assert all({"source", "rim_axis", "phys_throttle", "phys_brake", "relay_hz", "relay_gap_max_ms"} <= r.keys() for r in rows)
    assert all(r["user_steer"] == 0 and r["rim_axis"] == .02 for r in rows)


@pytest.mark.parametrize("ok,brake,reason", [(False, 0, "wheel not found"), (True, .1, "release the brake"), (True, .08, "")])
def test_physical_engagement_gate(ok, brake, reason):
    device = FakeDeviceManager()
    device.physical = replace(device.physical, ok=ok, brake=brake)
    assert wheel_session.engage_block(device) == reason


def test_capture_entry_failure_closes_relay(monkeypatch, tmp_path):
    import simsteer.runtime.loop as loop
    device = FakeDeviceManager()
    monkeypatch.setattr(loop, "DeviceManager", lambda **kw: device)
    class BadSource:
        def __enter__(self):
            raise OSError("capture failed")
        def __exit__(self, *args):
            pass
    clock = ReplayClock()
    clip = _make_clip(tmp_path / "clip", frames=1)
    with pytest.raises(OSError, match="capture failed"):
        loop.run(Settings(device="wheel", no_tuner=True), "ets2", source=BadSource(),
                 make_telemetry=lambda: ReplayTelemetry(clip, clock), clock=clock,
                 model=FakeModel(), headless=True)
    assert device.closed


def test_engage_race_reports_facade_refusal(monkeypatch, tmp_path, capsys):
    import simsteer.runtime.loop as loop
    device, sounds = FakeDeviceManager(), []

    class ReadyFlow(CalibFlow):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.lc.cal_status = loop.CalStatus.CALIBRATED
            self._finish_fov("kept")
            self.direction.model_sign = self.direction.axis_sign = 1
            self.output_direction.sign = 1
            self.step = Step.STEERING

    class Keys:
        def poll(self):
            return [loop.KEY_INSERT]

    def refuse():
        device.last_reason = "release the brake"
        return False

    device.engage = refuse
    monkeypatch.setattr(loop, "DeviceManager", lambda **kw: device)
    monkeypatch.setattr(loop, "CalibFlow", ReadyFlow)
    monkeypatch.setattr(loop.audio, "play", sounds.append)
    clip, clock = _make_clip(tmp_path / "clip", frames=1), ReplayClock()
    assert loop.run(Settings(device="wheel", no_tuner=True), "ets2", source=ReplaySource(clip, clock),
                    make_telemetry=lambda: ReplayTelemetry(clip, clock), clock=clock, keys=Keys(),
                    model=FakeModel(), headless=True) == 0
    assert "denied" in sounds and "engage" not in sounds
    assert "ENGAGE BLOCKED: release the brake" in capsys.readouterr().out
    assert device.closed and not device.engaged


@pytest.mark.parametrize("action", ["test_fire", "periodic_center"])
@pytest.mark.parametrize("passthrough", [True, False])
def test_passthrough_skips_test_fire_and_periodic_center(monkeypatch, tmp_path, action, passthrough):
    import simsteer.runtime.loop as loop
    device = FakeDeviceManager()
    device.passthrough = passthrough
    engages, centers, sleeps = [], [], []
    def engage():
        engages.append(True)
        device.engaged = True
    device.engage = engage
    device.center = lambda: centers.append(True)
    monkeypatch.setattr(loop, "DeviceManager", lambda **kw: device)
    monkeypatch.setattr(loop.time, "sleep", lambda seconds: sleeps.append(seconds))
    class Keys:
        first = True
        def poll(self):
            if self.first and action == "test_fire":
                self.first = False
                return [ord("t")]
            return []
    clip = _make_clip(tmp_path / "clip", frames=21 if action == "periodic_center" else 1)
    clock = ReplayClock()
    assert loop.run(Settings(device="wheel", no_tuner=True), "ets2", source=ReplaySource(clip, clock),
                    make_telemetry=lambda: ReplayTelemetry(clip, clock), clock=clock, keys=Keys(),
                    model=FakeModel(), headless=True) == 0
    if action == "test_fire":
        assert bool(engages) == (not passthrough)
        assert bool(sleeps) == (not passthrough)
    else:
        assert bool(centers) == (not passthrough)


def test_passthrough_off_replay_matches_base_bytes(tmp_path, monkeypatch):
    import simsteer.runtime.loop as loop
    # Execute the exact parent loop against the same replay/model/state.
    original = subprocess.run(["git", "show", "1d276e0f1e0fc8a44813341f0f35e0b05fac4348:simsteer/runtime/loop.py"],
                              check=True, capture_output=True, text=True).stdout
    parent = ModuleType("parent_loop")
    exec(compile(original, "parent_loop.py", "exec"), parent.__dict__)
    clip = _make_clip(tmp_path / "clip", frames=12)
    results = []
    for name, module in (("parent", parent), ("current", loop)):
        state = tmp_path / name
        state.mkdir()
        monkeypatch.setenv("SIMSTEER_DATA_DIR", str(state))
        clock, log = ReplayClock(), tmp_path / f"{name}.jsonl"
        assert module.run(Settings(no_tuner=True, no_gamepad=True, wheel_passthrough=False), "ets2",
                          source=ReplaySource(clip, clock), make_telemetry=lambda: ReplayTelemetry(clip, clock),
                          clock=clock, model=FakeModel(), headless=True, log_path=str(log)) == 0
        results.append(log.read_bytes())
    assert results[0] == results[1]
