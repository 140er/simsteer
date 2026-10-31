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


def test_model_replacement_changes_certification_identity(tmp_path):
    from simsteer.core.model_identity import actual_model_identity
    vision, policy = tmp_path / "vision.onnx", tmp_path / "policy.onnx"
    vision.write_bytes(b"first vision")
    policy.write_bytes(b"policy")
    original = actual_model_identity(vision, policy)
    vision.write_bytes(b"replacement vision")
    changed = actual_model_identity(vision, policy)
    assert original["vision_sha256"] != changed["vision_sha256"]
    assert original["policy_sha256"] == changed["policy_sha256"]
