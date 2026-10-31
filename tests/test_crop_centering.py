"""A white band at the exact center of the capture must land at the
center of both model views, or the model gets a lateral bias."""
from __future__ import annotations

import numpy as np
import pytest

from simsteer.core.calibration import Calibration
from simsteer.core.constants import SRC_H, SRC_W
from simsteer.core.preprocess import crop_for_model

CAPTURES = [(3840, 1600), (2560, 1600), (3840, 2160), (2560, 1080),
            (1920, 1080), (1920, 1200), (1366, 768)]

# A 1 px column aliases away in the ~4x downscale of a 4K wide view, so
# probe with a 5 px band and compare its centroid. A 1 px bias still fails.
BAND_HALF_WIDTH = 2
TOL_PX = 0.75


@pytest.mark.parametrize("w,h", CAPTURES, ids=[f"{w}x{h}" for w, h in CAPTURES])
def test_center_band_stays_centered(w, h):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, w // 2 - BAND_HALF_WIDTH:w // 2 + BAND_HALF_WIDTH + 1, :] = 255

    narrow, wide = crop_for_model(frame, Calibration(image_w=w, image_h=h))

    for name, view in (("narrow", narrow), ("wide", wide)):
        assert view.shape[:2] == (SRC_H, SRC_W)
        energy = view.sum(axis=(0, 2)).astype(np.float64)
        assert energy.sum() > 0, f"{name}: center band lost"
        centroid = float((energy * np.arange(SRC_W)).sum() / energy.sum())
        assert centroid == pytest.approx(SRC_W / 2, abs=TOL_PX), name
