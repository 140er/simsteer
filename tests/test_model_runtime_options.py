"""Session options and traffic side for both model runners.

The live runners must cap ORT's CPU threads (its default of one per
core starves the game) and default to right-side traffic, matching
openpilot's `traffic_convention[int(is_rhd)] = 1`:

https://github.com/commaai/openpilot/blob/0f17a9879364c6f410df7ddf990380ad623a4c92/selfdrive/modeld/modeld.py#L254-L266
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

RUNNERS = ["simsteer.core.model", "pilot.model"]


class _Session:
    def __init__(self, width: int) -> None:
        self.width = width
        self.inputs = None

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, outputs, inputs):
        self.inputs = {key: value.copy() for key, value in inputs.items()}
        return [np.zeros((1, self.width), dtype=np.float16)]


@pytest.mark.parametrize("runner", RUNNERS)
def test_driving_model_caps_ort_intra_op_threads(runner, monkeypatch, tmp_path):
    module = importlib.import_module(runner)
    vision, policy = tmp_path / "vision.onnx", tmp_path / "policy.onnx"
    vision.touch()
    policy.touch()
    monkeypatch.setattr(module, "VISION_PATH", vision)
    monkeypatch.setattr(module, "POLICY_PATH", policy)
    if hasattr(module, "actual_model_identity"):
        monkeypatch.setattr(module, "actual_model_identity", lambda *paths: None)
    options = {}

    def session(path, sess_options=None, providers=None):
        options[path] = sess_options
        return _Session(1576 if path == str(vision) else 1000)

    monkeypatch.setattr(module.ort, "InferenceSession", session)
    module.DrivingModel()

    assert set(options) == {str(vision), str(policy)}
    for path, opts in options.items():
        assert opts is not None, f"{path} built without SessionOptions"
        assert opts.intra_op_num_threads == module.ORT_INTRA_OP_THREADS == 3


@pytest.mark.parametrize("runner", RUNNERS)
def test_default_traffic_convention_is_right_side_traffic(runner, monkeypatch):
    module = importlib.import_module(runner)
    vision, policy = _Session(1576), _Session(1000)
    monkeypatch.setattr(module, "_make_session",
                        lambda path, providers: vision if path == module.VISION_PATH else policy)
    if hasattr(module, "actual_model_identity"):
        monkeypatch.setattr(module, "actual_model_identity", lambda *paths: None)

    module.DrivingModel().step(np.zeros((1, 12, 128, 256), dtype=np.uint8))

    np.testing.assert_array_equal(policy.inputs["traffic_convention"], [[1, 0]])
    assert policy.inputs["traffic_convention"].dtype == np.float16
