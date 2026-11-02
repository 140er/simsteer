from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from simsteer.app.manual import ManualInputs
from simsteer.app.settings import Settings
from simsteer.core.calibration import Calibration
from simsteer.core.constants import IDX_N, PLAN_WIDTH, T_IDXS
from simsteer.runtime.replay import ClipWriter, ReplayClock, ReplaySource, ReplayTelemetry
from simsteer.ui.hud import HudState
from simsteer.ui.overlay import draw_developer_panel, draw_overlay


def road_prediction():
    plan = np.zeros((IDX_N, PLAN_WIDTH), np.float32)
    plan[:, 0] = np.maximum(1, np.asarray(T_IDXS) * 20)
    plan[:, 3] = 20
    lines = np.zeros((4, IDX_N, 2), np.float32)
    lines[:, :, 0] = np.asarray([5.4, 1.8, -1.8, -5.4])[:, None]
    edges = np.zeros((2, IDX_N, 2), np.float32)
    edges[:, :, 0] = np.asarray([7, -7])[:, None]
    return SimpleNamespace(
        plan=plan,
        lane_lines=lines,
        lane_lines_prob=np.ones(4),
        road_edges=edges,
        lead_prob=np.zeros(3),
        desire_state=np.eye(8)[0],
    )


def test_disengaged_path_is_white_and_tint_requires_engagement():
    frame = np.full((540, 960, 3), 70, np.uint8)
    calib = Calibration()
    decoded = road_prediction()
    off = draw_overlay(frame, decoded, calib)
    on = draw_overlay(frame, decoded, calib, engaged=True)
    # A centre-line pixel is white while disengaged; a wide patch is green only when engaged.
    assert ((off == 255).all(axis=2)).sum() > 100
    off16, on16 = off.astype(np.int16), on.astype(np.int16)
    green_off = (off16[:, :, 1] > off16[:, :, 0] + 15) & (off16[:, :, 1] > off16[:, :, 2] + 15)
    green_on = (on16[:, :, 1] > on16[:, :, 0] + 15) & (on16[:, :, 1] > on16[:, :, 2] + 15)
    assert not green_off.any()
    assert green_on.sum() > 1000
    assert np.array_equal(frame, np.full_like(frame, 70))


def test_road_diagnostics_are_developer_only():
    frame = np.zeros((540, 960, 3), np.uint8)
    calib = Calibration()
    decoded = road_prediction()
    normal = draw_overlay(frame, decoded, calib)
    developer = draw_overlay(frame, decoded, calib, mode="dev")
    red_normal = (normal[:, :, 2] > 180) & (normal[:, :, 1] < 100)
    red_dev = (developer[:, :, 2] > 180) & (developer[:, :, 1] < 100)
    assert not red_normal.any()
    assert red_dev.sum() > 100
    for shape in [(540, 960, 3), (720, 1280, 3), (180, 320, 3)]:
        img = np.zeros(shape, np.uint8)
        state = HudState(dev_lines=[("test", (255, 255, 255))] * 10)
        draw_developer_panel(img, decoded, calib, state)


class Model:
    active_provider = policy_provider = "test"

    def step(self, img, big_img=None, desire=None, *, timestamp=None):
        vision = np.zeros(1576, np.float32)
        vision[107] = 1.2
        plan = road_prediction().plan
        policy = np.zeros(1000, np.float32)
        policy[: IDX_N * PLAN_WIDTH] = plan.ravel()
        return vision, policy


def test_runtime_h_key_and_tuner_switch_share_state(monkeypatch, tmp_path):
    import simsteer.runtime.loop as loop
    from simsteer.core.control import ControllerConfig
    from simsteer.ui.tuner import Tuner

    clip = tmp_path / "clip"
    writer = ClipWriter(clip, game="ets2")
    for i in range(4):
        writer.write_frame(i / 20, np.zeros((180, 320, 3), np.uint8))
        writer.write_telemetry(
            i / 20, {"speed_mps": 20, "yaw_rate_rad_s": 0, "wheel_angle_rad": 0, "game_steer": 0}
        )
    writer.close("ui-test")
    monkeypatch.setattr(loop, "_setup_window_or_exit", lambda *args: None)
    monkeypatch.setattr(loop.cv2, "setWindowTitle", lambda *args: None)
    monkeypatch.setattr(loop.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(loop, "show_scaled", lambda *args: None)
    monkeypatch.setattr(loop, "draw_developer_panel", lambda *args: None)
    modes = []
    original = loop.HudRenderer.draw

    def draw(self, img, state, mode):
        modes.append(mode)
        # The flow's instruction already explains its engagement gate.
        assert state.calibration is not None
        assert state.warnings == ["Frame history warming up"]
        original(self, img, state, mode)

    monkeypatch.setattr(loop.HudRenderer, "draw", draw)
    settings = Settings(no_gamepad=True, no_tuner=True)
    tuner = Tuner(
        Calibration(), ControllerConfig(), ManualInputs(), settings=settings, start_thread=False
    )
    tuner._set_developer(True)

    class Keys:
        def __init__(self):
            self.frames = iter([[], [ord("h")], [], [ord("q")]])

        def poll(self):
            return next(self.frames)

    clock = ReplayClock()
    assert (
        loop.run(
            settings,
            "ets2",
            source=ReplaySource(clip, clock),
            clock=clock,
            make_telemetry=lambda: ReplayTelemetry(clip, clock),
            model=Model(),
            keys=Keys(),
        )
        == 0
    )
    assert modes == ["dev", "user", "user"]
    assert settings.hud_mode == Settings.load().hud_mode == "user"
