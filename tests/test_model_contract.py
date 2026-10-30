from __future__ import annotations

import pytest

from simsteer.paths import model_path

VISION = model_path("driving_vision.onnx")
POLICY = model_path("driving_policy.onnx")


@pytest.mark.skipif(not (VISION.exists() and POLICY.exists()),
                    reason="models not fetched (python tools/fetch_model.py)")
def test_models_match_constants():
    from simsteer.core.model import check_contract
    assert check_contract(VISION, POLICY) == []
