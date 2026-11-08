from __future__ import annotations

import math

import numpy as np
import pytest

from simsteer.core import preprocess, warp
from simsteer.core.calibration import Calibration


@pytest.mark.parametrize("width", [1280, 1664, 1920, 2496, 2560])
@pytest.mark.parametrize("fov", [60.0, 75.0, 90.0, 100.0, 110.0])
def test_sampling_matches_intrinsic_warp(width, fov):
    ratio = warp.source_sampling_ratio(width, fov)
    matrix = warp.get_warp_matrix((0, 0, 0), warp.camera_intrinsics(width, 720, fov))
    assert matrix[0, 0] == pytest.approx(ratio)
    assert matrix[1, 1] == pytest.approx(ratio)
    assert warp.source_sampling_ratio(width, fov, bigmodel=True) == pytest.approx(2 * ratio)
    assert bool(preprocess.sampling_warning(width, fov)) == (ratio < 1.5)


def test_sampling_uses_luma_pixels_before_yuv_packing():
    # Exactly one source pixel per 512x256 model pixel, not two per
    # packed tensor element. Increasing both dimensions 1.5x gives 1.5:1.
    fov = math.degrees(2 * math.atan(1664 / (2 * warp.MEDMODEL_FL)))
    assert warp.source_sampling_ratio(1664, fov) == pytest.approx(1.0)
    assert warp.source_sampling_ratio(2496, fov) == pytest.approx(1.5)
    assert preprocess.sampling_warning(1664, fov)
    assert preprocess.sampling_warning(2496, fov) is None


def test_sampling_warning_recommends_resolution_and_fov():
    message = preprocess.sampling_warning(1280, 100.0)
    assert "0.59:1" in message
    assert "higher game resolution" in message
    assert "narrower in-game FOV" in message
    assert preprocess.sampling_warning(2560, 60.0) is None


@pytest.mark.parametrize("width, fov", [(0, 90), (-1, 90), (1280, 0),
                                        (1280, 180), (1280, float("nan"))])
def test_sampling_rejects_invalid_intrinsics(width, fov):
    with pytest.raises(ValueError):
        warp.source_sampling_ratio(width, fov)


def test_sampling_matches_simple_crop_scale():
    # A horizontal ramp exposes the actual crop/resize scale, including
    # integer rounding. Vertical cab crops and mount height do not zoom it.
    width, height = 1920, 1080
    image = np.broadcast_to(np.arange(width, dtype=np.float32)[None, :, None],
                             (height, width, 3)).copy()
    cal = Calibration(fov_h_deg=75, height_m=2.0, crop_top_pct=.15,
                      crop_bottom_pct=.15)
    view = warp.warp_to_model(image, cal)
    actual = float(view[100, 400, 0] - view[100, 100, 0]) / 300
    assert actual == pytest.approx(warp.source_sampling_ratio(width, 75), abs=.002)


def test_sampling_banner_is_once_per_run_after_fov_and_nonblocking(tmp_path, monkeypatch, capsys):
    import simsteer.runtime.loop as loop
    from simsteer.app.settings import Settings
    from simsteer.calib.flow import Step
    from simsteer.runtime.replay import ReplayClock, ReplaySource, ReplayTelemetry
    from test_replay import FakeModel, _make_clip

    clip = _make_clip(tmp_path / "clip", frames=20)
    clock = ReplayClock()
    flows = []
    real_flow = loop.CalibFlow

    def make_flow(*args, **kwargs):
        kwargs["fov_config"] = lambda aspect: 75.0
        flow = real_flow(*args, **kwargs)
        flows.append(flow)
        return flow

    monkeypatch.setattr(loop, "CalibFlow", make_flow)
    log = tmp_path / "run.jsonl"
    assert loop.run(Settings(no_tuner=True, no_gamepad=True), "ets2",
                    source=ReplaySource(clip, clock),
                    make_telemetry=lambda: ReplayTelemetry(clip, clock),
                    clock=clock, model=FakeModel(), headless=True,
                    log_path=str(log)) == 0
    output = capsys.readouterr().out
    assert output.count("Low capture detail") == 1
    assert output.index("calib: fov 75.0 (config)") < output.index("Low capture detail")
    assert flows[0].step is Step.CAMERA
    assert len(log.read_text().splitlines()) == 20
