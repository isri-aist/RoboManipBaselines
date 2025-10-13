"""
Helpers for the `jointhold_marker_check` PPO-Cus task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import numpy as np


DEFAULT_TARGET_JOINT_POS = np.array(
    [
        0.0,
        -0.477,
        0.0,
        0.8571976,
        0.0,
        1.2771976,
        -1.5707964,
        40.0,
    ],
    dtype=np.float32,
)


@dataclass
class JointholdMarkerCheckTask:
    rollout: "RolloutPpoCus"
    params: Mapping[str, object] = field(default_factory=dict)
    _target_joint_pos: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        custom_target = self.params.get("target_joint_pos")
        if custom_target is not None:
            self._target_joint_pos = np.asarray(custom_target, dtype=np.float32).reshape(
                -1
            )
        else:
            self._target_joint_pos = DEFAULT_TARGET_JOINT_POS.copy()

    # The interface expected by RolloutPpoCus ------------------------------
    def on_reset(self) -> None:  # pragma: no cover - no-op hook for now
        """Reset any task-specific state (none required here)."""
        return

    def get_extra_state(self) -> Dict[str, np.ndarray]:
        return {"target_joint_pos": self._target_joint_pos.copy()}


def build_ppo_task(
    rollout: "RolloutPpoCus", params: Optional[Mapping[str, object]] = None
):
    """
    Entry point required by RolloutPpoCus. Returns a task handler instance.
    """

    if params is None:
        params = {}

    return JointholdMarkerCheckTask(rollout=rollout, params=params)
