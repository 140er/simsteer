"""Finite display data for ticks where the driving pipeline is unavailable."""

import numpy as np

from simsteer.core.constants import IDX_N, PLAN_WIDTH
from simsteer.core.postprocess import Decoded


def neutral_decoded() -> Decoded:
    return Decoded(
        plan=np.zeros((IDX_N, PLAN_WIDTH)), plan_std=np.zeros((IDX_N, PLAN_WIDTH)),
        lane_lines=np.zeros((4, IDX_N, 2)), lane_lines_prob=np.zeros(4),
        road_edges=np.zeros((2, IDX_N, 2)),
        pose=None, pose_std=None,
        road_transform=np.zeros(6), road_transform_std=np.zeros(6),
        wide_from_device_euler=np.zeros(3), desire_state=np.zeros(8),
        lead_prob=np.zeros(3), leads=np.zeros((3, 6, 4)), leads_std=np.zeros((3, 6, 4)))
