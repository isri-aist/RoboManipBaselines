from os import path

import numpy as np
import pybulletX as px
from scipy.spatial.transform import Rotation as R

from .TactoSawyerEnvBase import TactoSawyerEnvBase


class TactoSawyerInsertEnv(TactoSawyerEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        self.board_pos_offsets = np.array(
            [
                [-0.04, 0.0, -0.009],
                [-0.015, 0.0, 0.005],
                [0.0, 0.0, 0.01],
                [0.015, 0.0, 0.005],
                [0.04, 0.0, -0.009],
            ]
        )  # [m]
        self.cube_pos_offsets = np.array(
            [
                [-0.096, 0.0, -0.05],
                [-0.045, 0.0, -0.05],
                [-0.001, 0.0, 0.0],
                [0.016, 0.0, 0.0],
                [0.031, 0.0, 0.0],
            ]
        )  # [m]
        self.cube2_pos_offsets = np.array(
            [
                [-0.031, 0.0, 0.0],
                [-0.016, 0.0, 0.0],
                [0.001, 0.0, 0.0],
                [0.045, 0.0, -0.05],
                [0.096, 0.0, -0.05],
            ]
        )  # [m]
        self.rotation_offsets = np.array(
            [
                [0.0, 40.0, 0.0],
                [0.0, 20.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, -20.0, 0.0],
                [0.0, -40.0, 0.0],
            ]
        )

        TactoSawyerEnvBase.__init__(
            self,
            np.array(
                [
                    0,
                    -1.24,
                    -0.58,
                    2.18,
                    0.27,
                    0.7,
                    1,
                    -0.02,
                    0.02,
                ]
            ),
        )

    def setup_task_specific_object(self):
        self.obj = px.Body(
            urdf_path=path.join(
                path.dirname(__file__), "../assets/tacto/objects/insert/board.urdf"
            ),
            base_position=[0.505, -0.03, 0.077],
            global_scaling=0.6,
        )
        self.obj2 = px.Body(
            urdf_path=path.join(
                path.dirname(__file__), "../assets/tacto/objects/insert/cube.urdf"
            ),
            base_position=[0.493, -0.03, 0.03],
            global_scaling=0.68,
            use_fixed_base=True,
        )
        self.obj3 = px.Body(
            urdf_path=path.join(
                path.dirname(__file__), "../assets/tacto/objects/insert/cube.urdf"
            ),
            base_position=[0.517, -0.03, 0.03],
            global_scaling=0.68,
            use_fixed_base=True,
        )
        self.obj4 = px.Body(
            urdf_path=path.join(
                path.dirname(__file__), "../assets/tacto/objects/insert/box.urdf"
            ),
            base_position=[0.505, 0.2, 0.05],
            global_scaling=0.6,
            use_fixed_base=True,
        )

        self.rgb_tactiles.add_body(self.obj)

    def reset_task_specific_object(self):
        pass

    def _get_reward(self):
        (x, y, z), _ = self.obj.get_base_pose()
        (xt, yt, zt), _ = self.obj4.get_base_pose()

        if abs(x - xt) < 0.01 and abs(y - yt) < 0.07 and z < 0.1:
            return 1.0
        else:
            return 0.0

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        if world_idx is None:
            world_idx = 1

        pos = self.obj.init_base_position.copy() + self.board_pos_offsets[world_idx]
        pos2 = self.obj2.init_base_position.copy() + self.cube_pos_offsets[world_idx]
        pos3 = self.obj3.init_base_position.copy() + self.cube2_pos_offsets[world_idx]
        rot = R.from_euler("xyz", self.rotation_offsets[world_idx], degrees=True)
        if self.world_random_scale is not None:
            pos += np.random.uniform(
                low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
            )
        self.obj.set_base_pose(pos, rot.as_quat())
        self.obj2.set_base_pose(pos2)
        self.obj3.set_base_pose(pos3)

        return world_idx
