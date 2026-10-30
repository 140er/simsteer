"""The device binding and wheel telemetry have independent polarities."""

from collections import deque
import json
from types import SimpleNamespace

import numpy as np
import pytest

from simsteer.app.settings import Settings
from simsteer.calib.flow import CalibFlow, Step, flow_path
from simsteer.calib.steering import OutputDirection
from simsteer.core.learners.livecalib import CalStatus
from simsteer.core.learners.liveparams import TRUSTED_MIN_SAMPLES
from test_steering_direction import _bends, _flow, _plan, _probe_output


@pytest.mark.parametrize("pad_to_game", [1, -1])
@pytest.mark.parametrize("wheel_convention", [1, -1], ids=["left-positive", "right-positive"])
@pytest.mark.parametrize("input_sign", [1, -1], ids=["normal-input", "inverted-input"])
def test_runtime_measures_output_before_ai_and_disengages_on_contradiction(
        monkeypatch, tmp_path, pad_to_game, wheel_convention, input_sign):
    import simsteer.runtime.loop as loop

    holder = {}
    clock = SimpleNamespace(t=0.0)
    clock.now = lambda: clock.t
    decoded = _plan(0.004)  # The model always requests LEFT.
    decoded.pose = np.array([22.0, 0.0, 0.0])
    monkeypatch.setattr(loop, "decode", lambda *args: decoded)
    monkeypatch.setattr(loop, "_lane_offset", lambda d: 0.0)
    monkeypatch.setattr(loop, "LongitudinalController", lambda cfg: SimpleNamespace(
        compute=lambda *args: (0.0, 0.0), reset=lambda: None))
    monkeypatch.setattr(loop, "FrameQueue", lambda: SimpleNamespace(
        push=lambda *args: (None, None)))

    class Flow(CalibFlow):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.lc.cal_status = CalStatus.CALIBRATED
            self._finish_fov("kept")
            _bends(self, wheel_convention, input_sign)
            # Isolate output mapping from rack fitting, which has its own tests.
            self.lp.x[:] = (input_sign * 4.0, 0.0, 0.0)
            self.lp.samples = TRUSTED_MIN_SAMPLES
            self.lp.update = lambda *args, **kwargs: None
            holder["flow"] = self

    monkeypatch.setattr(loop, "CalibFlow", Flow)

    class Plant:
        kind = "gamepad"
        last_error = None
        engaged = False
        command = 0.0
        response = 0.0
        frame = 0
        flipped_at = None

        def __init__(self):
            self.commands = []
            self.history = deque([0.0] * 5, maxlen=5)

        def engage(self):
            self.engaged = True

        def disengage(self):
            self.engaged = False
            self.command = 0.0

        def center(self):
            self.command = 0.0

        def set_steering(self, command, **kwargs):
            flow = holder["flow"]
            measured = flow.log_fields().get("steering_output_sign")
            self.commands.append((self.frame, command, measured))
            self.command = command

        def set_throttle_brake(self, *args, **kwargs):
            pass

        @property
        def physical_wheel(self):
            return self.response / (input_sign * 4.0 * wheel_convention)

    plant = Plant()
    monkeypatch.setattr(loop, "DeviceManager", lambda **kwargs: plant)

    class Source:
        done = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def grab(self):
            plant.frame += 1
            clock.t += 0.05
            plant.response = pad_to_game * plant.history[0]
            plant.history.append(plant.command)
            # A binding change after successful AI steering must stop control.
            if plant.frame >= 240:
                plant.flipped_at = plant.flipped_at or plant.frame
                plant.response *= -1
            return np.zeros((2, 2, 3), np.uint8)

    tel = SimpleNamespace(
        available=True, speed_mps=lambda: 22.0, yaw_rate_rad_s=lambda: 0.0,
        wheel_angle_rad=lambda wb: wheel_convention * plant.physical_wheel,
        game_steer=lambda: plant.response, close=lambda: None)

    class Keys:
        def poll(self):
            return [loop.KEY_INSERT] if plant.frame == 1 else []

    log = tmp_path / "output.jsonl"
    assert loop.run(Settings(no_tuner=True), "ets2", source=Source(),
                    make_telemetry=lambda: tel, clock=clock, keys=Keys(),
                    model=SimpleNamespace(step=lambda *args, **kwargs: (None, None),
                                          active_provider="test", policy_provider="test"),
                    headless=True, max_frames=340, log_path=str(log)) == 0

    unverified = [(frame, command) for frame, command, sign in plant.commands if sign is None]
    assert len(unverified) >= 80
    # Only the existing small symmetric probe may reach the device before measurement.
    assert max(abs(command) for _, command in unverified) <= 0.03
    assert any(command < -0.005 for _, command in unverified)
    assert any(command > 0.005 for _, command in unverified)
    measured = [(frame, command) for frame, command, sign in plant.commands
                if sign == pad_to_game and 180 <= frame < 240]
    assert measured, "manual input evidence alone must never authorize AI output"
    for _, command in measured:
        game_axis = pad_to_game * command
        assert game_axis / (input_sign * 4.0 * wheel_convention) > 0  # LEFT
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert all(row["ai_steer_axis"] == 0 for row in rows
               if row.get("steering_output_sign") is None)
    assert any(not row["engaged"] for row in rows if 260 <= row["frame"] < 340)
    flow = holder["flow"]
    assert not flow.ai_steering_ready
    assert not flow.direction.ready
    assert flow.engage_block()
    assert flow.step is Step.STEERING


def test_recalibrate_clears_the_saved_input_sign():
    from simsteer.core.calibration import Calibration
    from simsteer.core.learners.livecalib import LiveCalib
    from simsteer.core.learners.liveparams import LiveParams

    lp = LiveParams(game="ets2", device_kind="gamepad")
    lp.set_axis_sign(-1)
    flow = CalibFlow("ets2", "gamepad", live_calib=LiveCalib(game="ets2"),
                     live_params=lp, calib=Calibration())
    flow.restart("recalibrate")
    lp.save()
    assert lp.axis_sign is None
    assert json.loads(lp._resolve_path("ets2", "gamepad").read_text())["axis_sign"] is None


@pytest.mark.parametrize("gate", ["disengaged", "not-probing", "slow", "fast",
                                  "lane-change", "missing", "single-turn", "noise"])
def test_output_direction_requires_independent_probe_evidence(gate):
    from simsteer.app.probe import SteeringProbe

    direction = OutputDirection(lag_frames=5, prior=-1)
    probe = SteeringProbe()
    for i in range(400):
        command = probe.tick(0.05, 0.0, 22.0, False, True, False, True)
        if gate == "single-turn":
            command = abs(command)
        observed = -command if gate != "noise" else (0.012 if i % 2 else -0.012)
        assert not direction.update(command=command,
                                    observed=None if gate == "missing" else observed,
                                    speed=3.0 if gate == "slow" else 35.0 if gate == "fast" else 22.0,
                                    engaged=gate != "disengaged", probing=gate != "not-probing",
                                    lane_change=gate == "lane-change", dt=0.05)
    assert not direction.ready


def test_output_evidence_resumes_but_cannot_be_bypassed_by_a_saved_sign():
    from simsteer.app.probe import SteeringProbe

    direction = OutputDirection(lag_frames=5)
    direction.load_dict({"sign": -1})
    assert not direction.ready
    probe = SteeringProbe()
    for i in range(240):
        command = probe.tick(0.05, 0.0, 22.0, False, True, False, True)
        direction.update(command=command, observed=command, speed=22.0,
                         engaged=True, probing=True, lane_change=False, dt=0.05)
        if i == 50:
            saved = direction.to_dict()
            direction = OutputDirection(lag_frames=5)
            direction.load_dict(saved)
            assert not direction.ready
    assert direction.sign == 1  # Evidence overrides the negative prior.


def test_output_verification_is_per_device_and_cleared_on_switch_and_reset():
    from simsteer.core.learners.livecalib import LiveCalib
    from simsteer.core.learners.liveparams import LiveParams

    lp = LiveParams(game="ets2", device_kind="gamepad")
    flow = _flow(lp)
    _bends(flow, 1, -1)
    _probe_output(flow, -1)
    flow.save()
    saved = CalibFlow("ets2", "gamepad", live_calib=flow.lc, live_params=lp, calib=flow.calib)
    assert saved.ai_steering_ready and saved.output_direction.sign == -1
    lp.switch_device("wheel")
    saved.switch_device("wheel")
    assert not saved.ai_steering_ready and not saved.output_direction.ready
    lp.switch_device("gamepad")
    saved.switch_device("gamepad")
    assert saved.direction.ready and not saved.output_direction.ready
    assert not saved.ai_steering_ready
    assert not CalibFlow("ats", "gamepad", live_calib=flow.lc,
                         live_params=lp, calib=flow.calib).output_direction.ready
    _probe_output(saved, 1)
    saved.lc = LiveCalib(game="ets2")
    saved.restart("R pressed")
    assert saved.output_direction.sign is None and lp.axis_sign is None
    assert flow_path("ets2", "gamepad").exists()


def test_previous_version_done_requires_output_verification():
    from simsteer.core.learners.liveparams import LiveParams

    lp = LiveParams(game="ets2", device_kind="gamepad")
    flow = _flow(lp)
    _bends(flow, 1, 1)
    state = flow.to_dict()
    state["version"] = 2
    state["step"] = "done"
    state["result"] = {"passed": True}
    state["steering"].pop("output_direction")
    flow_path("ets2", "gamepad").write_text(json.dumps(state))
    restored = CalibFlow("ets2", "gamepad", live_calib=flow.lc,
                         live_params=lp, calib=flow.calib)
    assert restored.step is Step.STEERING and not restored.calibrated
    assert restored.device_axis(0.2) == 0.0
    assert not restored.output_direction.ready
