"""Independent projection and real-model acceptance contracts."""

import numpy as np
import pytest

from simsteer.core.warp import SBIGMODEL_CY, SBIGMODEL_INTRINSICS, get_warp_matrix
from tools.diagnose_fov_render import projection_audit
from tools.measure_time_base import fov_target_met
from tools.make_synthetic_clip import dash_intervals


def test_wide_camera_principal_point_matches_pinned_upstream():
    # common/transformations/model.py:26-34 at the pinned revision:
    # https://github.com/commaai/openpilot/blob/0f17a9879364c6f410df7ddf990380ad623a4c92/common/transformations/model.py#L26-L34
    expected = np.array([[455., 0., 256.], [0., 455., 151.8], [0., 0., 1.]])
    assert SBIGMODEL_CY == 151.8
    np.testing.assert_array_equal(SBIGMODEL_INTRINSICS, expected)
    # Test the inverse used by the actual warp, not just the public matrix.
    source = np.array([[480., 0., 480.], [0., 480., 270.], [0., 0., 1.]])
    point = get_warp_matrix((0., 0., 0.), source, bigmodel=True) @ [256., 151.8, 1.]
    np.testing.assert_allclose(point[:2] / point[2], [480., 270.], atol=2e-5)


@pytest.mark.parametrize("fov", [75., 90.])
def test_renderer_fov_and_motion_match_independent_projection(fov):
    audit = projection_audit(fov)
    assert audit["recovered_hfov"] == pytest.approx(fov, abs=1e-9)
    assert audit["pixel_aspect"] == pytest.approx(1., abs=1e-12)
    assert audit["recovered_speed_mps"] == pytest.approx(22., abs=1e-9)
    assert audit["reprojection_max_px"] < 1e-9


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), 86.11, 91.50001])
def test_exact_flow_result_cannot_pass_a_failed_real_model_solve(bad):
    rows = [{"hz": hz, "real_model_fov_deg": 90., "synthetic_fov_deg": 90.}
            for hz in (10, 15, 18, 20)]
    rows[0]["real_model_fov_deg"] = bad
    assert not fov_target_met(rows, 90.)


def test_fov_acceptance_includes_all_rates_and_the_exact_boundary():
    rows = [{"hz": hz, "real_model_fov_deg": fov}
            for hz, fov in zip((10, 15, 18, 20), (88.5, 90., 91.5, 90.))]
    assert fov_target_met(rows, 90.)
    assert not fov_target_met([], 90.)
    assert not fov_target_met(rows[:-1], 90.)
    assert not fov_target_met([rows[0]] * 4, 90.)


def test_diagnostic_dash_boundaries_move_with_the_world():
    a = next(start for start, _ in dash_intervals(.7, 2., 160.)
             if abs(start + .7 - 36) < 1e-8)
    b = next(start for start, _ in dash_intervals(5.1, 2., 160.)
             if abs(start + 5.1 - 36) < 1e-8)
    assert (a - b) / .2 == pytest.approx(22., abs=1e-9)
