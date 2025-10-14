"""
Helpers for the `align` PPO-Cus task.

This task augments the policy state with marker pose information expressed in the
base frame. Each marker contributes a 9D vector:
    - translation (tx, ty, tz)
    - rotation represented as the first two columns of the rotation matrix (6D)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import numpy as np

from .jointhold_marker_check import (
    DEFAULT_TARGET_JOINT_POS as _DEFAULT_TARGET_JOINT_POS,
)


def _rotation_matrix_to_6d(rotation: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to the 6D representation (Zhou et al., 2019).
    """
    if rotation.shape != (3, 3):
        raise ValueError(f"Rotation matrix must be 3x3, got {rotation.shape}")

    x_axis = rotation[:, 0]
    y_axis = rotation[:, 1]

    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    y_axis = y_axis - np.dot(x_axis, y_axis) * x_axis
    y_length = np.linalg.norm(y_axis) + 1e-8
    if y_length < 1e-6:
        # Fallback: use original y-axis if projection is degenerate
        y_axis = rotation[:, 1]
        y_length = np.linalg.norm(y_axis) + 1e-8
    y_axis = y_axis / y_length

    return np.concatenate([x_axis, y_axis])


@dataclass
class AlignTask:
    rollout: "RolloutPpoCus"
    params: Mapping[str, object] = field(default_factory=dict)
    _target_joint_pos: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        custom_target = self.params.get("target_joint_pos")
        if custom_target is not None:
            self._target_joint_pos = np.asarray(custom_target, dtype=np.float32).reshape(-1)
        else:
            self._target_joint_pos = _DEFAULT_TARGET_JOINT_POS.copy()

        marker_ids = self.rollout.required_marker_ids
        if not marker_ids:
            raise ValueError(
                "[AlignTask] No marker IDs specified in meta info. "
                "Please populate 'ppo_task.markers'."
            )
        if len(marker_ids) != 1:
            raise ValueError(
                "[AlignTask] Expected exactly one marker ID for align task, "
                f"got {marker_ids}"
            )
        self.marker_id = marker_ids[0]

    def on_reset(self) -> None:  # pragma: no cover - optional hook
        return

    def get_extra_state(self) -> Dict[str, np.ndarray]:
        extra: Dict[str, np.ndarray] = {
            "target_joint_pos": self._target_joint_pos.astype(np.float32)
        }

        if self.marker_id not in self.rollout.marker_transform_cache:
            raise RuntimeError(
                f"[AlignTask] Marker id {self.marker_id} not available in cache."
            )

        T = self.rollout.marker_transform_cache[self.marker_id]
        if T.shape != (4, 4):
            raise ValueError(
                f"[AlignTask] Expected 4x4 transform matrix, got shape {T.shape}"
            )

        translation = T[:3, 3].astype(np.float32)
        rotation = T[:3, :3]
        rotation6d = _rotation_matrix_to_6d(rotation).astype(np.float32)

        z_axis = rotation[:, 2]
        extra_components = z_axis[:2]  # take first two elements of the third column

        marker_pose = np.concatenate(
            [translation, rotation6d, extra_components]
        ).astype(np.float32)
        extra["marker_pose_base"] = marker_pose

        return extra


def build_ppo_task(
    rollout: "RolloutPpoCus", params: Optional[Mapping[str, object]] = None
):
    if params is None:
        params = {}
    return AlignTask(rollout=rollout, params=params)
