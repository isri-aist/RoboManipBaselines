from os import path

import numpy as np

from .MujocoUR10eEnvBase import MujocoUR10eEnvBase


class MujocoUR10eCableEnv(MujocoUR10eEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoUR10eEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/ur10e/env_ur10e_cable.xml",
            ),
            np.array(
                [
                    np.pi,
                    -np.pi / 2,
                    -0.75 * np.pi,
                    -0.25 * np.pi,
                    np.pi / 2,
                    np.pi / 2,
                    *np.zeros(8),
                ]
            ),
            **kwargs,
        )

        self.original_pole_pos = self.model.body("poles").pos.copy()
        self.pole_pos_offsets = np.array(
            [
                [-0.03, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.03, 0.0, 0.0],
                [0.06, 0.0, 0.0],
                [0.09, 0.0, 0.0],
                [0.12, 0.0, 0.0],
            ]
        )  # [m]

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.pole_pos_offsets)
        self.model.body("poles").pos = (
            self.original_pole_pos + self.pole_pos_offsets[world_idx]
        )
        return world_idx
