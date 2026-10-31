"""Rendered vanishing points must land at each virtual camera's principal point."""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from simsteer.core.calibration import Calibration
from simsteer.core.warp import (
    MEDMODEL_CX, MEDMODEL_CY, MEDMODEL_FL,
    SBIGMODEL_CX, SBIGMODEL_CY, SBIGMODEL_FL,
    warp_to_model,
)


def vanishing_point_scene(w, h, hfov, pitch):
    # A forward ray at infinity: square-pixel pinhole projection of a
    # camera pitched down. A Gaussian marker keeps subpixel positions
    # measurable after both interpolation and downsampling.
    focal = w / (2 * math.tan(math.radians(hfov) / 2))
    u, v = w / 2, h / 2 - focal * math.tan(math.radians(pitch))
    rows, cols = np.ogrid[:h, :w]
    marker = np.exp(-((cols - u) ** 2 + (rows - v) ** 2) / (2 * 2.0**2))
    return np.repeat(marker.astype(np.float32)[:, :, None], 3, axis=2)


def centroid(view):
    energy = view[:, :, 0].astype(np.float64)
    assert energy.sum() > 0, "vanishing point lost"
    rows, cols = np.indices(energy.shape)
    return np.array([(energy * cols).sum(), (energy * rows).sum()]) / energy.sum()


@pytest.mark.parametrize("bigmodel", [False, True], ids=["narrow", "wide"])
@pytest.mark.parametrize("pitch", [-4.0, 0.0, 2.0, 5.0])
@pytest.mark.parametrize("w,h,hfov,top,bottom", [
    (1920, 1080, 120.0, 0.0, 0.0),
    (1920, 1080, 75.0, 0.18, 0.04),
    (1366, 768, 90.0, 0.03, 0.20),
    (3840, 1600, 110.0, 0.12, 0.12),
    (1600, 1200, 60.0, 0.0, 0.15),
    (1280, 720, 40.0, 0.05, 0.0),
])
def test_simple_vanishing_point_matches_perspective(w, h, hfov, top, bottom,
                                                   pitch, bigmodel):
    scene = vanishing_point_scene(w, h, hfov, pitch)
    calib = Calibration(image_w=w, image_h=h, fov_h_deg=hfov,
                        pitch_deg=pitch, crop_top_pct=top, crop_bottom_pct=bottom)
    expected = np.array([SBIGMODEL_CX, SBIGMODEL_CY] if bigmodel
                        else [MEDMODEL_CX, MEDMODEL_CY])
    simple = centroid(warp_to_model(scene, calib, bigmodel))
    perspective = centroid(warp_to_model(scene, replace(calib, simple_warp=False),
                                         bigmodel))
    # The crop uses integer boundaries. Allow one source pixel after
    # scaling; the perspective reference has no crop-rounding error.
    focal = w / (2 * math.tan(math.radians(hfov) / 2))
    model_fl = SBIGMODEL_FL if bigmodel else MEDMODEL_FL
    tolerance = 1.0 + model_fl / focal
    assert perspective == pytest.approx(expected, abs=0.3)
    assert simple == pytest.approx(expected, abs=tolerance)
    assert simple == pytest.approx(perspective, abs=tolerance)


def test_simple_warp_explicitly_ignores_yaw():
    scene = vanishing_point_scene(1280, 720, 75.0, 2.0)
    calib = Calibration(fov_h_deg=75.0, pitch_deg=2.0)
    assert np.array_equal(warp_to_model(scene, calib),
                          warp_to_model(scene, replace(calib, yaw_deg=3.0)))
    # The full warp does correct yaw; the mode distinction is intentional.
    perspective = replace(calib, simple_warp=False)
    assert not np.array_equal(warp_to_model(scene, perspective),
                              warp_to_model(scene, replace(perspective, yaw_deg=3.0)))


def test_capture_dimensions_override_stale_saved_dimensions():
    scene = vanishing_point_scene(1920, 1080, 75.0, 2.0)
    calib = Calibration(image_w=2560, image_h=1600, fov_h_deg=75.0, pitch_deg=2.0)
    assert np.array_equal(warp_to_model(scene, calib),
                          warp_to_model(scene, replace(calib, image_w=1920, image_h=1080)))
