from __future__ import annotations

import numpy as np
import pytest

from simsteer.core.model import DrivingModel
from simsteer.core.postprocess import decode
from simsteer.paths import model_path

VISION = model_path("driving_vision.onnx")
POLICY = model_path("driving_policy.onnx")


@pytest.mark.skipif(not (VISION.exists() and POLICY.exists()),
                    reason="models not fetched (python tools/fetch_model.py)")
def test_models_match_constants():
    from simsteer.core.model import check_contract
    assert check_contract(VISION, POLICY) == []


def _upstream_decode(vision, policy):
    """Reference for the pinned non-MHP heads, transcribed from openpilot:

    https://github.com/commaai/openpilot/blob/0f17a9879364c6f410df7ddf990380ad623a4c92/selfdrive/modeld/parse_model_outputs.py#L44-L118
    https://github.com/commaai/openpilot/blob/0f17a9879364c6f410df7ddf990380ad623a4c92/selfdrive/modeld/fill_model_msg.py#L101-L107

    Keep literal offsets/shapes independent of the adapter's slice tables.
    Both pinned heads have exactly twice their prediction width, so the
    upstream parser takes its non-MHP branch with in_N=out_N=0.
    """
    def mdn(raw, shape):
        means, log_stds = np.split(raw, 2)
        # Upstream safe_exp caps log standard deviations at 11.
        return (means.reshape(shape),
                np.exp(np.clip(log_stds, -np.inf, 11)).reshape(shape))

    def probability(logits):
        return 1 / (1 + np.exp(np.clip(-logits, -np.inf, 11)))

    plan, plan_std = mdn(policy[0:990], (33, 15))
    leads, leads_std = mdn(vision[917:1061], (3, 6, 4))
    pose, pose_std = mdn(vision[87:99], (6,))
    road_transform, road_transform_std = mdn(vision[105:117], (6,))
    wide, _ = mdn(vision[99:105], (3,))
    lanes, _ = mdn(vision[117:645], (4, 33, 2))
    edges, _ = mdn(vision[653:917], (2, 33, 2))
    desire = np.exp(policy[990:998] - policy[990:998].max())
    return {
        "plan": plan, "plan_std": plan_std,
        "lane_lines": lanes,
        "lane_lines_prob": probability(vision[645:653])[1::2],
        "road_edges": edges, "pose": pose, "pose_std": pose_std,
        "road_transform": road_transform,
        "road_transform_std": road_transform_std,
        "wide_from_device_euler": wide,
        "desire_state": desire / desire.sum(),
        "lead_prob": probability(vision[1061:1064]),
        "leads": leads, "leads_std": leads_std,
    }


@pytest.fixture
def golden_outputs():
    # Every entry, including unused regions and padding, is distinct.
    # Permute each tensor so an incorrect offset cannot hide in a ramp.
    rng = np.random.default_rng(217)
    values = np.arange(2576, dtype=np.float32) / 256 - 4
    return rng.permutation(values[:1576]), rng.permutation(values[1576:])


@pytest.mark.parametrize("field", [
    "plan", "plan_std", "lane_lines", "lane_lines_prob", "road_edges",
    "pose", "pose_std", "road_transform", "road_transform_std",
    "wide_from_device_euler", "desire_state", "lead_prob", "leads", "leads_std",
])
def test_decode_matches_pinned_upstream(golden_outputs, field):
    vision, policy = golden_outputs
    expected = _upstream_decode(vision, policy)
    actual = decode(vision, policy)
    np.testing.assert_allclose(getattr(actual, field), expected[field], rtol=1e-6)


@pytest.mark.parametrize("log_std", [-15.0, -2.0, 0.0, 2.0, 11.0, 12.0])
def test_plan_and_lead_stds_are_linear_and_bounded(golden_outputs, log_std):
    vision, policy = golden_outputs
    vision[989:1061] = log_std
    policy[495:990] = log_std
    expected = _upstream_decode(vision, policy)
    actual = decode(vision, policy)
    np.testing.assert_allclose(actual.plan_std, expected["plan_std"], rtol=1e-6)
    np.testing.assert_allclose(actual.leads_std, expected["leads_std"], rtol=1e-6)


@pytest.fixture
def model_inputs(monkeypatch):
    class Session:
        def __init__(self, width):
            self.width = width
            self.inputs = None

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, outputs, inputs):
            self.inputs = {key: value.copy() for key, value in inputs.items()}
            return [np.zeros((1, self.width), dtype=np.float16)]

    vision, policy = Session(1576), Session(1000)
    monkeypatch.setattr("simsteer.core.model._make_session",
                        lambda path, providers: vision if path == VISION else policy)
    image = np.zeros((1, 12, 128, 256), dtype=np.uint8)
    return DrivingModel(), policy, image


def test_default_traffic_convention_is_left_hand_drive(model_inputs):
    """is_rhd=False sets index zero, for driving on the right:

    https://github.com/commaai/openpilot/blob/0f17a9879364c6f410df7ddf990380ad623a4c92/selfdrive/modeld/modeld.py#L254-L266
    """
    model, policy, image = model_inputs
    model.step(image)
    np.testing.assert_array_equal(policy.inputs["traffic_convention"], [[1, 0]])
    assert policy.inputs["traffic_convention"].dtype == np.float16


@pytest.mark.parametrize("is_rhd", [False, True])
def test_explicit_traffic_convention_reaches_policy(model_inputs, is_rhd):
    model, policy, image = model_inputs
    convention = np.zeros(2)
    convention[int(is_rhd)] = 1
    model.step(image, traffic_convention=tuple(convention))
    np.testing.assert_array_equal(policy.inputs["traffic_convention"], convention[None])
    assert policy.inputs["traffic_convention"].dtype == np.float16


def test_desire_history_matches_pinned_upstream(model_inputs):
    """Rising-edge extraction, idle suppression and temporal max pooling:

    https://github.com/commaai/openpilot/blob/0f17a9879364c6f410df7ddf990380ad623a4c92/selfdrive/modeld/modeld.py#L89-L118
    https://github.com/commaai/openpilot/blob/0f17a9879364c6f410df7ddf990380ad623a4c92/selfdrive/modeld/compile_modeld.py#L116-L151

    The pinned model consumes 25 windows of four frames. Check every
    window through held requests, release/re-arm, switches, history expiry
    and fractional inputs that do not meet the upstream >.99 delta gate.
    """
    model, policy, image = model_inputs
    idle, left, right = np.eye(8, dtype=np.float32)[[0, 3, 4]]
    half_left = left * 0.5
    requests = ([None, idle, left] + [left] * 105
                + [None, left, left, right, right, left, None, half_left, left]
                + [None] * 105)
    previous = np.zeros(8, dtype=np.float32)
    history = np.zeros((100, 8), dtype=np.float32)
    for frame, request in enumerate(requests):
        current = np.zeros(8, dtype=np.float32) if request is None else request.copy()
        current[0] = 0
        edge = np.where(current - previous > .99, current, 0)
        previous[:] = current
        history[:-1] = history[1:]
        history[-1] = edge
        expected = history.reshape(25, 4, 8).max(axis=1)[None].astype(np.float16)
        supplied = None if request is None else request.copy()
        model.step(image, desire=supplied)
        np.testing.assert_array_equal(policy.inputs["desire_pulse"], expected,
                                      err_msg=f"desire history at frame {frame}")
        assert policy.inputs["desire_pulse"].dtype == np.float16
        if request is not None:
            np.testing.assert_array_equal(supplied, request)


def test_held_desire_expires_from_history(model_inputs):
    model, policy, image = model_inputs
    request = np.eye(8, dtype=np.float32)[3]
    model.step(image, desire=request)
    assert policy.inputs["desire_pulse"][0, -1, 3] == 1
    for _ in range(100):
        model.step(image, desire=request)
    np.testing.assert_array_equal(policy.inputs["desire_pulse"], 0)
