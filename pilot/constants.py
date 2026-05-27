"""Mirror of openpilot's selfdrive/modeld/constants.py + the output-slice
indices we extracted from the ONNX metadata. Pinned to the model versions
fetched on 2026-05-13. If `tools/inspect_model.py` reports different
numbers, update this file.
"""

from __future__ import annotations


def _idx_function(idx: int, max_val: float, max_idx: int = 32) -> float:
    return max_val * (idx / max_idx) ** 2


# Time and distance grids (33 quadratically spaced points 0..max).
IDX_N = 33
T_IDXS = [_idx_function(i, max_val=10.0) for i in range(IDX_N)]   # seconds 0..10
X_IDXS = [_idx_function(i, max_val=192.0) for i in range(IDX_N)]  # meters  0..192

# Frame timing.
MODEL_RUN_FREQ = 20      # Hz, how often we step the model
MODEL_CONTEXT_FREQ = 5   # Hz, temporal context sampling
FRAME_SKIP = MODEL_RUN_FREQ // MODEL_CONTEXT_FREQ  # 4
N_FRAMES = 2             # vision input stacks current + previous

# Buffer lengths.
FEATURE_LEN = 512        # hidden_state width per timestep
FEATURE_BUFFER_LEN = 25  # 5 s * 5 Hz
DESIRE_LEN = 8           # one-hot width
DESIRE_BUFFER_LEN = 25

# Vision input dims.
MODEL_W = 256
MODEL_H = 128
# Source frame size (before YUV420 + 2x2 stride collapse).
SRC_W = MODEL_W * 2      # 512
SRC_H = MODEL_H * 2      # 256

# Policy plan structure.
PLAN_WIDTH = 15          # per timestep: pos(3), vel(3), accel(3), euler(3), orient_rate(3)
PLAN_TIMESTEPS = IDX_N

# Lane / road edges.
NUM_LANE_LINES = 4
NUM_ROAD_EDGES = 2
LANE_LINES_WIDTH = 2     # (x_offset, height) per polynomial point

# Output slices baked into the ONNX metadata. Verified via inspect_model.py.
VISION_SLICES: dict[str, tuple[int, int]] = {
    "meta":                   (0,    55),
    "desire_pred":            (55,   87),
    "pose":                   (87,   99),
    "wide_from_device_euler": (99,  105),
    "road_transform":         (105, 117),
    "lane_lines":             (117, 645),
    "lane_lines_prob":        (645, 653),
    "road_edges":             (653, 917),
    "lead":                   (917, 1061),
    "lead_prob":              (1061, 1064),
    "hidden_state":           (1064, 1576),
}

POLICY_SLICES: dict[str, tuple[int, int]] = {
    "plan":         (0,   990),
    "desire_state": (990, 998),
}


# Plan field offsets within the 15-wide plan vector.
class PlanField:
    POSITION = slice(0, 3)           # x, y, z (m)
    VELOCITY = slice(3, 6)           # vx, vy, vz (m/s)
    ACCELERATION = slice(6, 9)       # ax, ay, az (m/s^2)
    EULER = slice(9, 12)             # roll, pitch, yaw (rad)
    ORIENTATION_RATE = slice(12, 15) # rates (rad/s)
