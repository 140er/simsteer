"""Two-model driving inference: vision -> policy.

Each step:
  1. Run vision on (img, big_img) — both (1, 12, 128, 256) uint8.
  2. Slice `hidden_state` from the vision output; push it onto the
     features ringbuffer; sample 25 timesteps as `features_buffer`.
  3. Push the desire pulse onto the desire ringbuffer; max-pool over
     each FRAME_SKIP window to get 25 timesteps as `desire_pulse`.
  4. Run policy on (desire_pulse, traffic_convention, features_buffer).
  5. Return the flat outputs from both models (as float32 numpy arrays);
     postprocess.py decodes them.
"""

from __future__ import annotations

import base64
import io
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort

from simsteer.core.constants import (
    DESIRE_BUFFER_LEN,
    DESIRE_LEN,
    FEATURE_BUFFER_LEN,
    FEATURE_LEN,
    FRAME_SKIP,
    MODEL_H,
    MODEL_W,
    POLICY_SLICES,
    VISION_SLICES,
)
from simsteer.paths import model_path

VISION_PATH = model_path("driving_vision.onnx")
POLICY_PATH = model_path("driving_policy.onnx")


def _make_session(path: Path, providers: list[str]) -> ort.InferenceSession:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `python tools/fetch_model.py`")
    available = set(ort.get_available_providers())
    providers = [p for p in providers if p in available] or ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), providers=providers)


class DrivingModel:
    """Stateful: holds the feature/desire ringbuffers between steps."""

    FEAT_BUF_LEN = FRAME_SKIP * (FEATURE_BUFFER_LEN - 1) + 1   # 97
    DESIRE_BUF_LEN = FRAME_SKIP * DESIRE_BUFFER_LEN            # 100

    def __init__(self, providers: list[str] | None = None,
                 policy_providers: list[str] | None = None) -> None:
        # DirectML accelerates the vision encoder (the big one). The policy
        # head currently fails to initialize on DML (opset-20 op the DML EP
        # rejects with E_INVALIDARG), and at ~3 ms on CPU it isn't the
        # bottleneck anyway. Caller can override either list.
        if providers is None:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        if policy_providers is None:
            policy_providers = ["CPUExecutionProvider"]
        self.vision = _make_session(VISION_PATH, providers)
        self.policy = _make_session(POLICY_PATH, policy_providers)
        self.active_provider = self.vision.get_providers()[0]
        self.policy_provider = self.policy.get_providers()[0]

        # Ringbuffers (float32 internally; cast to float16 only when feeding policy).
        self._feat_q: deque[np.ndarray] = deque(
            [np.zeros(FEATURE_LEN, dtype=np.float32) for _ in range(self.FEAT_BUF_LEN)],
            maxlen=self.FEAT_BUF_LEN,
        )
        self._desire_q: deque[np.ndarray] = deque(
            [np.zeros(DESIRE_LEN, dtype=np.float32) for _ in range(self.DESIRE_BUF_LEN)],
            maxlen=self.DESIRE_BUF_LEN,
        )

    def vision_only(self, img: np.ndarray, big_img: np.ndarray) -> np.ndarray:
        """Run the vision model alone and return its flat output. Vision is
        stateless given the stacked frame pair, so this leaves the policy's
        feature buffer untouched."""
        out = self.vision.run(None, {"img": img.astype(np.uint8, copy=False),
                                     "big_img": big_img.astype(np.uint8, copy=False)})
        return out[0].astype(np.float32).reshape(-1)

    def step(self, img: np.ndarray, big_img: np.ndarray | None = None,
             desire: np.ndarray | None = None,
             traffic_convention: tuple[float, float] = (0.0, 1.0),
             ) -> tuple[np.ndarray, np.ndarray]:
        # traffic_convention: openpilot's modeld sets index `int(is_rhd)`
        # to 1.0. So:
        #   (0, 1)  =  is_rhd False  =  drive on the RIGHT  (US, EU, AC stock tracks)
        #   (1, 0)  =  is_rhd True   =  drive on the LEFT   (UK, Japan, AU)
        # We had (1, 0) hardcoded — telling the model "we drive on the
        # left" which biases its lane targeting. Default flipped to (0, 1).
        """Run one inference step. Returns (vision_out_flat, policy_out_flat)
        as float32 numpy arrays of shapes (1576,) and (1000,)."""
        if big_img is None:
            big_img = img

        # --- Vision ---
        vision_outputs = self.vision.run(
            None, {"img": img.astype(np.uint8, copy=False),
                   "big_img": big_img.astype(np.uint8, copy=False)})
        vision_flat = vision_outputs[0].astype(np.float32).reshape(-1)

        # Latch hidden_state into the feature ringbuffer.
        h_start, h_end = VISION_SLICES["hidden_state"]
        hidden = vision_flat[h_start:h_end]
        self._feat_q.append(hidden)

        # Push desire pulse into its ringbuffer.
        if desire is None:
            desire = np.zeros(DESIRE_LEN, dtype=np.float32)
        self._desire_q.append(desire.astype(np.float32, copy=False))

        # Build policy inputs by sampling the queues.
        feat_arr = np.stack(list(self._feat_q), axis=0)        # (97, 512)
        feat_sampled = feat_arr[::FRAME_SKIP]                  # (25, 512)
        features_buffer = feat_sampled[np.newaxis, ...].astype(np.float16)

        des_arr = np.stack(list(self._desire_q), axis=0)        # (100, 8)
        des_pooled = des_arr.reshape(DESIRE_BUFFER_LEN, FRAME_SKIP, DESIRE_LEN).max(axis=1)
        desire_pulse = des_pooled[np.newaxis, ...].astype(np.float16)  # (1, 25, 8)

        tc = np.asarray([traffic_convention], dtype=np.float16)  # (1, 2)

        # --- Policy ---
        policy_outputs = self.policy.run(None, {
            "desire_pulse": desire_pulse,
            "traffic_convention": tc,
            "features_buffer": features_buffer,
        })
        policy_flat = policy_outputs[0].astype(np.float32).reshape(-1)

        return vision_flat, policy_flat


class _SliceUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) == ("builtins", "slice"):
            return slice
        raise pickle.UnpicklingError(f"refusing {module}.{name} in model metadata")


def _output_slices(sess: ort.InferenceSession, total: int
                   ) -> dict[str, tuple[int, int]] | None:
    raw = sess.get_modelmeta().custom_metadata_map.get("output_slices")
    if raw is None:
        return None
    slices = _SliceUnpickler(io.BytesIO(base64.b64decode(raw))).load()
    return {k: (v.start or 0, total if v.stop is None else v.stop)
            for k, v in slices.items()}


def check_contract(vision_path: Path, policy_path: Path) -> list[str]:
    """Check both models against the layout in `constants.py`: input names,
    shapes and dtypes, flat output width, the embedded output slices, and a
    zero-input step through `decode`. Returns a list of problems."""
    from simsteer.core.postprocess import decode

    img = [1, 2 * 6, MODEL_H, MODEL_W]
    want_inputs = {
        "vision": {"img": (img, "tensor(uint8)"),
                   "big_img": (img, "tensor(uint8)")},
        "policy": {"desire_pulse": ([1, DESIRE_BUFFER_LEN, DESIRE_LEN], "tensor(float16)"),
                   "traffic_convention": ([1, 2], "tensor(float16)"),
                   "features_buffer": ([1, FEATURE_BUFFER_LEN, FEATURE_LEN],
                                       "tensor(float16)")},
    }
    tables = {"vision": VISION_SLICES, "policy": POLICY_SLICES}
    # The flat outputs end in padding that the slice tables leave out.
    widths = {"vision": 1576, "policy": 1000}
    problems: list[str] = []
    sessions = {}
    for kind, path in (("vision", vision_path), ("policy", policy_path)):
        sess = _make_session(path, ["CPUExecutionProvider"])
        sessions[kind] = sess
        got = {i.name: (list(i.shape), i.type) for i in sess.get_inputs()}
        if got != want_inputs[kind]:
            problems.append(f"{kind} inputs {got} != {want_inputs[kind]}")
        outs = sess.get_outputs()
        width = widths[kind]
        if len(outs) != 1 or outs[0].shape[-1] != width:
            problems.append(f"{kind} outputs {[(o.name, o.shape) for o in outs]}, "
                            f"want one output of width {width}")
            continue
        slices = _output_slices(sess, width)
        if slices is not None:
            slices.pop("pad", None)
        if slices is not None and slices != tables[kind]:
            problems.append(f"{kind} output_slices {slices} != {tables[kind]}")
    if problems:
        return problems

    vision = sessions["vision"].run(None, {
        "img": np.zeros(img, np.uint8), "big_img": np.zeros(img, np.uint8)})[0]
    policy = sessions["policy"].run(None, {
        k: np.zeros(shape, np.float16)
        for k, (shape, _) in want_inputs["policy"].items()})[0]
    d = decode(vision.astype(np.float32).reshape(-1),
               policy.astype(np.float32).reshape(-1))
    for name, arr, shape in (("plan", d.plan, (33, 15)),
                             ("lane_lines", d.lane_lines, (4, 33, 2)),
                             ("road_edges", d.road_edges, (2, 33, 2)),
                             ("pose", d.pose, (6,)),
                             ("road_transform", d.road_transform, (6,)),
                             ("desire_state", d.desire_state, (DESIRE_LEN,)),
                             ("leads", d.leads, (3, 6, 4))):
        if arr.shape != shape:
            problems.append(f"decoded {name} shape {arr.shape} != {shape}")
        elif not np.all(np.isfinite(arr)):
            problems.append(f"decoded {name} is not finite on a zero input")
    return problems
