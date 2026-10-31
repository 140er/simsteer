"""Regressions for the live calibration and virtual-pad findings."""

from __future__ import annotations

import json
import math
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from simsteer.calib.flow import CalibFlow, Obs, Step
from simsteer.core.calibration import Calibration
from simsteer.core.learners.livecalib import LiveCalib
from simsteer.core.learners.liveparams import LiveParams
from simsteer.runtime.output.gamepad import Gamepad, GamepadError
from simsteer.runtime.telemetry.scs_shm import OFF_CABIN_AV_Y, ScsTelemetry


class BootTransport:
    """The driver's zero cache and different first visible report."""

    def __init__(self, fail_at=(), fail_after_delivery=True):
        self.report = [0] * 6  # lx, ly, rx, ry, left trigger, right trigger
        self.cached = (0,) * 6
        self.visible = (-3356, 0, 0, 0, 0, 0)
        self.sent = []
        self.fail_at = fail_at
        self.fail_after_delivery = fail_after_delivery
        self.attached = True
        self.removed = False
        self._busp, self._devicep = 1, self
        self.update()  # vgamepad submits its zero report on construction

    def reset(self):
        self.report[:] = [0] * 6

    def left_joystick_float(self, x_value_float, y_value_float):
        self.report[:2] = [round(x_value_float * 32767), round(y_value_float * 32767)]

    def right_joystick_float(self, x_value_float, y_value_float):
        self.report[2:4] = [round(x_value_float * 32767), round(y_value_float * 32767)]

    def right_joystick(self, x_value, y_value):
        self.report[2:4] = [x_value, y_value]

    def left_trigger_float(self, value_float):
        self.report[4] = round(value_float * 255)

    def right_trigger_float(self, value_float):
        self.report[5] = round(value_float * 255)

    def update(self):
        report = tuple(self.report)
        self.sent.append(report)
        if len(self.sent) in self.fail_at and not self.fail_after_delivery:
            raise OSError("report transport failed")
        if report != self.cached:
            self.cached = self.visible = report
        # Transport errors may arrive after delivery, too.
        if len(self.sent) in self.fail_at:
            raise OSError("report transport failed")


def install_transport(monkeypatch, transport):
    monkeypatch.setitem(sys.modules, "vgamepad", SimpleNamespace(VX360Gamepad=lambda: transport))

    def remove(bus, device):
        device.attached = False
        device.removed = True

    monkeypatch.setitem(sys.modules, "vgamepad.win", SimpleNamespace(
        vigem_client=SimpleNamespace(vigem_target_remove=remove)))


def test_item1_repeated_zero_cannot_replace_boot_report():
    transport = BootTransport()
    for _ in range(5):
        transport.update()
    assert transport.visible[0] == -3356
    assert transport.cached == (0,) * 6


@pytest.mark.parametrize("recreate", [False, True])
def test_item1_creation_forces_change_then_full_neutral(monkeypatch, recreate):
    for _ in range(2 if recreate else 1):
        transport = BootTransport()
        install_transport(monkeypatch, transport)
        pad = Gamepad()
        assert not pad.engaged
        assert transport.visible == (0,) * 6
        assert transport.sent[-2:] == [(0, 0, 1, 0, 0, 0), (0,) * 6]
        assert all(r[0] == r[1] == r[3] == r[4] == r[5] == 0 for r in transport.sent)
        pad.set_steering(1)
        pad.set_throttle_brake(1, 1)
        assert transport.visible == (0,) * 6


@pytest.mark.parametrize("fail_at", [(2,), (3,), (2, 3)])
@pytest.mark.parametrize("fail_after_delivery", [False, True])
def test_item1_transport_failure_neutralizes_and_releases(monkeypatch, fail_at, fail_after_delivery):
    transport = BootTransport(fail_at, fail_after_delivery)
    install_transport(monkeypatch, transport)
    with pytest.raises(GamepadError):
        Gamepad()
    assert transport.sent[-1] == (0,) * 6  # finally always tries neutral
    assert transport.removed and not transport.attached


def scs_yaw(turns):
    tel = ScsTelemetry.__new__(ScsTelemetry)
    tel._mm = None if turns is None else bytearray(OFF_CABIN_AV_Y + 4)
    if turns is not None:
        struct.pack_into("<f", tel._mm, OFF_CABIN_AV_Y, turns)
    return tel.yaw_rate_rad_s()


@pytest.mark.parametrize("turns", [0.01, -0.01, 0.0, None])
def test_item2_scs_yaw_is_radians_per_second(turns):
    yaw = scs_yaw(turns)
    if turns is None:
        assert yaw is None
    else:
        assert yaw == pytest.approx(turns * math.tau)


def test_item2_truncated_mapping_has_no_yaw():
    tel = ScsTelemetry.__new__(ScsTelemetry)
    tel._mm = bytearray(OFF_CABIN_AV_Y)
    assert tel.yaw_rate_rad_s() is None


def test_item2_scs_yaw_crosses_camera_and_fov_thresholds():
    from simsteer.runtime.fov import _usable
    lc = LiveCalib()
    lc.update(Calibration(), np.array([10., 0., 0.]), None, scs_yaw(0.01), 10.)
    assert lc.samples == 0 and lc.rej_turning == 1
    assert not _usable(10., scs_yaw(0.01))


def run_frames(monkeypatch, tmp_path, frames, observations, *, log=True, on_push=None):
    import simsteer.runtime.loop as loop

    state = SimpleNamespace(i=-1)
    clock = SimpleNamespace(now=lambda: max(0, state.i) * .05)
    lc = LiveCalib(clock=clock.now)
    monkeypatch.setattr(loop, "LiveCalib", lambda **kw: lc)
    monkeypatch.setattr(loop, "decode", lambda *args: observations[state.i]["decoded"])

    def push(frame, calib):
        if on_push:
            on_push(frame, calib)
        return None, None

    monkeypatch.setattr(loop, "FrameQueue", lambda: SimpleNamespace(push=push))

    class Source:
        done = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def grab(self):
            state.i += 1
            if state.i == len(frames):
                self.done = True
                return None
            return frames[state.i]

    tel = SimpleNamespace(
        available=True, speed_mps=lambda: observations[state.i]["speed"],
        yaw_rate_rad_s=lambda: observations[state.i]["yaw"],
        wheel_angle_rad=lambda wb: 0., game_steer=lambda: 0., close=lambda: None)
    path = tmp_path / "frames.jsonl"
    assert loop.run(loop.Settings(no_tuner=True, no_gamepad=True), "ets2",
                    source=Source(), make_telemetry=lambda: tel, clock=clock,
                    model=SimpleNamespace(step=lambda *a, **k: (None, None)),
                    headless=True, log_path=str(path) if log else None) == 0
    return lc, [json.loads(s) for s in path.read_text().splitlines()] if log else []


def healthy_decoded():
    from simsteer.core.postprocess import decode
    from test_steering_direction import _plan
    d = decode(np.zeros(1576, np.float32), np.zeros(1000, np.float32))
    d.plan = _plan(0., 10.).plan
    d.pose = np.array([10., 0., 0.])
    d.pose_std = np.array([.01, .02, .01])
    d.road_transform = np.array([0., 0., 1.2])
    d.road_transform_std = np.array([.01, .01, .02])
    d.wide_from_device_euler = None
    return d


def test_item3_capture_resize_precedes_warp_and_vertical_fov_entry(monkeypatch, tmp_path):
    seen = []

    def push(frame, calib):
        seen.append((calib.image_w, calib.image_h))
        calib.fov_v_deg = 72.
        aspect = frame.shape[1] / frame.shape[0]
        expected = math.degrees(2 * math.atan(math.tan(math.radians(36)) * aspect))
        assert calib.fov_h_deg == pytest.approx(expected)
        assert calib.fov_v_deg == pytest.approx(72.)

    frames = [np.zeros((1440, 3440, 3), np.uint8), np.zeros((1440, 2560, 3), np.uint8)]
    run_frames(monkeypatch, tmp_path, frames,
               [dict(speed=10., yaw=0., decoded=healthy_decoded())] * 2, on_push=push)
    assert seen == [(3440, 1440), (2560, 1440)]


def test_item4_manual_fov_survives_restart_with_history(monkeypatch, data_dir):
    cal = Calibration(fov_h_deg=90.)
    cal.save(game="ets2")
    lc = LiveCalib(game="ets2")
    lp = LiveParams(game="ets2", device_kind="gamepad")
    flow = CalibFlow("ets2", "gamepad", live_calib=lc, live_params=lp, calib=cal)
    for _ in range(120):
        lc.update(cal, np.array([10., .02, -.03]), np.array([0., 0., 1.2]), 0., 10.)
    assert lc.blocks == 1 and lc._block_n == 20
    save_flow = flow.save

    def save_after_calibration():
        assert Calibration.load(game="ets2").fov_h_deg == pytest.approx(cal.fov_h_deg)
        save_flow()

    monkeypatch.setattr(flow, "save", save_after_calibration)
    cal.fov_h_deg = 120.10235969224264
    flow.tick(Obs(now=1., dt=.05))
    before = json.loads((data_dir / "livecalib_state_ets2.json").read_text())
    restored_cal = Calibration.load(game="ets2")
    restored_lc = LiveCalib(game="ets2")
    restored = CalibFlow("ets2", "gamepad", live_calib=restored_lc,
                        live_params=lp, calib=restored_cal)
    assert restored_cal.fov_h_deg == pytest.approx(cal.fov_h_deg)
    assert restored.step is Step.CAMERA and restored.fov_source == "manual"
    assert restored_lc.samples == 120 and restored_lc.blocks == 1 and restored_lc._block_n == 20
    np.testing.assert_allclose(restored_lc._rpys, before["rpys"])
    np.testing.assert_allclose(restored_lc._block_rpy_sum, before["block_rpy_sum"])


REJECTION_CASES = [
    ("missing_pose", {"pose": None, "v_ego": None}, 0),
    ("missing_speed", {"v_ego": None}, 0),
    ("speed", {"v_ego": 6., "actual_yaw_rate": .1}, 1),
    ("turning", {"actual_yaw_rate": .04}, 1),
    ("lateral_acceleration", {"v_ego": 100., "actual_yaw_rate": .02}, 1),
    ("human_steering", {"game_steer": .2}, 1),
    ("model_speed", {"pose": np.array([6., 0., 0.])}, 1),
    ("lateral_uncertainty", {"pose_std": np.array([.01, .1, .01])}, 1),
    ("height_uncertainty", {"road_transform_std": np.array([.01, .01, .04])}, 1),
    ("pitch_bounds", {"pose": np.array([10., 0., -5.])}, 1),
    ("yaw_bounds", {"pose": np.array([10., 2., 0.])}, 1),
    ("height_bounds", {"road_transform": np.array([0., 0., .4])}, 1),
    ("height_bounds", {"road_transform": np.array([0., 0., 4.1])}, 1),
    (None, {}, 0),
]


@pytest.mark.parametrize("reason,changes,rejected", REJECTION_CASES)
def test_item5_first_failing_reason_and_unchanged_acceptance(reason, changes, rejected):
    lc, cal = LiveCalib(), Calibration()
    lc._last_game_steer = 0.
    inputs = dict(pose=np.array([10., 0., 0.]), road_transform=np.array([0., 0., 1.2]),
                  actual_yaw_rate=0., v_ego=10., pose_std=np.array([.01, .02, .01]),
                  road_transform_std=np.array([.01, .01, .02]), game_steer=0.)
    lc.update(cal, **(inputs | changes))
    assert lc.samples == int(reason is None) and lc.rejected == rejected
    assert lc.last_rejection_reason == reason
    lc._suppress_frames_remaining = 0
    lc._last_game_steer = 0.
    lc.update(cal, **inputs)
    assert lc.last_rejection_reason is None


def test_item5_jsonl_records_actual_gate_inputs_without_affecting_acceptance(monkeypatch, tmp_path):
    observations = [dict(speed=None, yaw=.01, decoded=healthy_decoded()),
                    dict(speed=6., yaw=.1, decoded=healthy_decoded()),
                    dict(speed=10., yaw=0., decoded=healthy_decoded())]
    frames = [np.zeros((180, 320, 3), np.uint8)] * 3
    lc, rows = run_frames(monkeypatch, tmp_path, frames, observations)
    unlogged, _ = run_frames(monkeypatch, tmp_path, frames, observations, log=False)
    assert lc.samples == unlogged.samples == 1
    assert lc.rejected == unlogged.rejected == 1
    assert [r["camera_rejection"] for r in rows] == ["missing_speed", "speed", None]
    assert rows[0]["actual_speed_mps"] is None  # distinct from the control fallback
    assert rows[0]["actual_yaw_rad_s"] == .01
    assert rows[0]["pose_vx"] == 10.
    assert rows[0]["pose_lateral_std"] == .02
    assert rows[0]["road_height_m"] == 1.2 and rows[0]["road_height_std_m"] == .02
    assert rows[0]["fov_h_deg"] == 90.
    assert (rows[0]["frame_width"], rows[0]["frame_height"]) == (320, 180)


@pytest.mark.parametrize("polarity", [-1, 1])
def test_item6_relative_saturation_rejects_with_setup_hint(polarity):
    from test_steering_direction import _bends, _flow
    flow = _flow(LiveParams(game="ets2", device_kind="gamepad"))
    _bends(flow, 1, 1)
    assert flow.output_direction.start(speed=0., observed=0., user_axis=0.)
    assert flow.measure_output(speed=0., observed=0., user_axis=0., dt=.05) == .1
    assert flow.measure_output(speed=0., observed=0., user_axis=polarity * 1., dt=.05) == 0.
    assert not flow.output_direction.ready and not flow.output_direction.active
    hint = flow.hint(0.)
    assert "likely relative" in hint and "Wheel (absolute)" in hint and "0% deadzone" in hint
    assert "\n" not in hint
    flow.game = "ats"
    assert "relative" not in flow.hint(0.)  # game-specific advice


def test_item6_absolute_response_succeeds_and_genuine_takeover_still_rejects():
    from test_steering_direction import _bends, _flow, _probe_output
    flow = _flow(LiveParams(game="ets2", device_kind="gamepad"))
    _bends(flow, 1, 1)
    _probe_output(flow, -1)
    assert flow.output_direction.ready and "relative" not in flow.hint(0.)
    flow.output_direction.sign = None
    assert flow.output_direction.start(speed=0., observed=0., user_axis=0.)
    assert flow.measure_output(speed=0., observed=0., user_axis=0., dt=.05)
    assert flow.measure_output(speed=0., observed=-.1, user_axis=.4, dt=.05) == 0.
    assert not flow.output_direction.active and not flow.output_direction.ready
    assert "Release steering" in flow.hint(0.) and "relative" not in flow.hint(0.)
