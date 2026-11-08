"""Model-free replay geometry checks, not ONNX accuracy measurements.

Two coloured road markers of known separation stand in for learned
scale cues. Their expansion over one second gives a forward-speed
observation through the real warp, independently of the solver's FOV
equation. Mount height changes their vertical position in the images.
"""
from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache

import cv2
import numpy as np
import pytest

from simsteer.calib.flow import Step
from simsteer.calib.fov_source import config_reader
from simsteer.core.calibration import Calibration
from simsteer.core.preprocess import crop_for_model
from simsteer.core.warp import MEDMODEL_FL
from simsteer.runtime.fov import FovSolver
from simsteer.runtime.replay import ClipWriter, ReplayClock, ReplaySource, ReplayTelemetry
from test_calib_flow import new_flow
from tools.make_synthetic_clip import CameraModel, main, render_calib

TRUE_FOV = 75.0
SPEED = 22.0
MARKER_X = 1.8
FOV_TOL_DEG = 1.0
RESOLUTIONS = [(1280, 720), (1920, 1080), (2560, 1440)]
HEIGHTS = [1.2, 1.6, 2.0]


def _marker_centres(image):
    b, g, r = cv2.split(image)
    mask = ((b > 180) & (r > 180) & (g < 100)).astype(np.uint8)
    count, _, _, centres = cv2.connectedComponentsWithStats(mask)
    assert count == 3, "both road markers must survive recording and warping"
    return centres[1:][np.argsort(centres[1:, 0])]


@pytest.fixture(scope="module", params=HEIGHTS, ids=lambda h: f"height-{h}")
def mount_clip(request, tmp_path_factory, resolution):
    tmp_path = tmp_path_factory.mktemp("fov-clip")
    width, height = resolution
    cam = CameraModel(TRUE_FOV, 0, 0, request.param, width, height)
    writer = ClipWriter(tmp_path / "clip", game="ets2", width=width)
    for t in (0.0, 1.0):
        image = render_calib(SPEED * t, cam)
        points = cam.project(np.array([-MARKER_X, MARKER_X]),
                              np.full(2, 40.0 - SPEED * t))
        for point in points:
            cv2.circle(image, tuple(np.rint(point * 256).astype(int)),
                        max(3, round(width / 320)) * 256, (255, 0, 255),
                        -1, cv2.LINE_AA, shift=8)
        writer.write_frame(t, image)
        writer.write_telemetry(t, {"speed_mps": SPEED, "yaw_rate_rad_s": 0.0})
    meta = writer.close("synthetic-fov-test")
    assert (meta["width"], meta["height"]) == resolution
    clock = ReplayClock()
    telemetry = ReplayTelemetry(tmp_path / "clip", clock)
    frames = []
    with ReplaySource(tmp_path / "clip", clock) as source:
        while (frame := source.grab()) is not None:
            assert frame.shape[:2] == (height, width)
            assert telemetry.speed_mps() == SPEED
            assert telemetry.yaw_rate_rad_s() == 0.0
            frames.append(frame)
    assert len(frames) == 2 and clock.now() == 1.0
    # Independent vertical projection: changing the mount moves the
    # markers, rather than just attaching a height label to identical clips.
    row = float(_marker_centres(frames[0])[:, 1].mean())
    expected = height / 2 + cam.f * request.param / 40.0
    assert row == pytest.approx(expected, abs=1.0)
    return frames, request.param


@pytest.fixture(scope="module", params=RESOLUTIONS, ids=lambda wh: f"{wh[0]}x{wh[1]}")
def resolution(request):
    return request.param


@pytest.mark.parametrize("simple", [True, False], ids=["simple", "perspective"])
def test_fov_replay_across_mounts_and_resolutions(mount_clip, resolution, simple, capsys):
    frames, height_m = mount_clip
    cal = Calibration(image_w=resolution[0], image_h=resolution[1],
                      fov_h_deg=95.0, height_m=height_m, simple_warp=simple,
                      crop_top_pct=.1, crop_bottom_pct=.1)

    @lru_cache(maxsize=None)
    def measured_speed(fov):
        distances = []
        for frame in frames:
            narrow, _ = crop_for_model(frame, replace(cal, fov_h_deg=fov))
            centres = _marker_centres(narrow)
            span = float(centres[1, 0] - centres[0, 0])
            distances.append(2 * MARKER_X * MEDMODEL_FL / span)
        return distances[0] - distances[1]

    # The stand-in derives speed from the replayed pixels at each warp,
    # rather than returning tan(true/2)/tan(set/2) by construction.
    def views(frame, candidate):
        speed = measured_speed(candidate.fov_h_deg)
        return np.full((6, 1, 1), speed), np.zeros((6, 1, 1))

    solver = FovSolver(lambda img, big: (float(img[0, 6, 0, 0]), .1), views)
    flow = new_flow(cal=cal, make_fov_solver=lambda: solver, persist=False)
    for _ in range(250):
        flow.feed_fov(frames[-1], measured_speed(95.0), SPEED, 0.0)
        assert cal.fov_h_deg == 95.0  # proposals never silently change the warp
        if flow.fov_proposal is not None:
            break
    assert solver.stage == "bracket" and solver.done
    assert solver.result == pytest.approx(TRUE_FOV, abs=FOV_TOL_DEG)
    assert flow.accept_fov()
    assert flow.step is Step.CAMERA and flow.fov_source == "measured"
    error = cal.fov_h_deg - TRUE_FOV
    assert abs(error) <= FOV_TOL_DEG
    capsys.readouterr()
    print(f"FOV height={height_m:.1f} resolution={resolution[0]}x{resolution[1]} "
          f"warp={'simple' if simple else 'perspective'} error={error:+.1f} deg")


@pytest.mark.parametrize("axis", ["horizontal", "vertical"])
def test_config_fov_uses_replayed_capture_aspect(mount_clip, resolution, tmp_path, axis):
    frames, height_m = mount_clip
    value = TRUE_FOV
    if axis == "vertical":
        value = math.degrees(2 * math.atan(math.tan(math.radians(TRUE_FOV) / 2)
                                          * resolution[1] / resolution[0]))
    config = tmp_path / "camera.cfg"
    config.write_text(f"camera_fov = {value:.12f}\n")
    cal = Calibration(fov_h_deg=95, height_m=height_m)
    flow = new_flow(cal=cal, fov_config=config_reader(str(config), "camera_fov", axis),
                    persist=False)
    flow.feed_fov(frames[0], None, None, None)
    assert flow.step is Step.CAMERA and flow.fov_source == "config"
    assert cal.fov_h_deg == pytest.approx(TRUE_FOV, abs=.1)


def test_generator_preserves_requested_resolution_and_height(tmp_path):
    import json

    out = tmp_path / "generated"
    assert main(["--calib", "--out", str(out), "--minutes", "0.001",
                 "--width", "1280", "--height", "720", "--height-m", "2.0"]) == 0
    meta = json.loads((out / "clip.json").read_text())
    assert (meta["width"], meta["height"]) == (1280, 720)
    assert meta["truth"]["height_m"] == 2.0
    with ReplaySource(out, ReplayClock()) as source:
        assert source.grab().shape[:2] == (720, 1280)
