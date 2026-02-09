from os import path

import numpy as np
import pybullet as p
import pybulletX as px

from .TactoSawyerEnvBase import Camera, TactoSawyerEnvBase


class TactoSawyerSpoonEnv(TactoSawyerEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        TactoSawyerEnvBase.__init__(
            self,
            np.array(
                [
                    0.1,
                    -1.37,
                    -0.68,
                    1.7,
                    0.2,
                    1.3,
                    1.1,
                    -0.02,
                    0.02,
                ]
            ),
        )

        self.cameras["front"] = Camera(
            camera_pos=[0.8, 0, 0.05],
            camera_distance=1.0,
            yaw=90,
            pitch=-30.0,
            roll=0,
        )

        self.spoon_positions = np.array([[0.775, -0.034, 0.1], [0.755, -0.051, 0.25]])

        self.spoon_orientations_rpy = np.array(
            [
                [np.deg2rad(92), np.deg2rad(0), np.deg2rad(0)],
                [np.deg2rad(-88), np.deg2rad(-5), np.deg2rad(0)],
            ]
        )

    def setup_task_specific_object(self):
        self.box_scaling = 0.4
        self.spoon = px.Body(
            urdf_path=path.join(
                path.dirname(__file__), "../assets/tacto/objects/spoon/spoon.urdf"
            ),
            base_position=[0.75, 0.1, 0.2],
            base_orientation=[-1.0, -0.4, -0, 1],
            global_scaling=0.06,
        )
        self.start_box = px.Body(
            urdf_path=path.join(
                path.dirname(__file__), "../assets/tacto/objects/spoon/blue_box.urdf"
            ),
            base_position=[0.8, 0, 0.1],
            global_scaling=self.box_scaling,
            use_fixed_base=True,
        )
        self.green_box = px.Body(
            urdf_path=path.join(
                path.dirname(__file__), "../assets/tacto/objects/spoon/green_box.urdf"
            ),
            base_position=[0.8, -0.3, 0.1],
            global_scaling=self.box_scaling,
            use_fixed_base=True,
        )
        self.red_box = px.Body(
            urdf_path=path.join(
                path.dirname(__file__), "../assets/tacto/objects/spoon/red_box.urdf"
            ),
            base_position=[0.8, 0.3, 0.1],
            global_scaling=self.box_scaling,
            use_fixed_base=True,
        )

        self.goal_boxes = [self.red_box, self.green_box]
        self.all_task_obj = [self.spoon, self.start_box, self.green_box, self.red_box]
        for obj in self.all_task_obj:
            self.rgb_tactiles.add_body(obj)

    def reset_task_specific_object(self):
        for obj in self.goal_boxes:
            obj.reset()

    def _get_reward(self):
        (x, y, z), _ = self.spoon.get_base_pose()
        goal_height = 0.75 * self.box_scaling
        if self.world_idx == 0:
            box = self.goal_boxes[0]
            (bx, by, _), _ = box.get_base_pose()
            bx_min = bx - (0.3 * self.box_scaling)
            bx_max = bx + (0.3 * self.box_scaling)
            by_min = by - (0.3 * self.box_scaling)
            by_max = by + (0.3 * self.box_scaling)
            if (
                z < goal_height - 0.1
                and (bx_min < x and x < bx_max)
                and (by_min < y and y < by_max)
            ):
                return 1.0
        if self.world_idx == 1:
            box = self.goal_boxes[1]
            (bx, by, _), _ = box.get_base_pose()
            bx_min = bx - (0.3 * self.box_scaling)
            bx_max = bx + (0.3 * self.box_scaling)
            by_min = by - (0.3 * self.box_scaling)
            by_max = by + (0.3 * self.box_scaling)
            if (
                z < goal_height + 0.1
                and (bx_min < x and x < bx_max)
                and (by_min < y and y < by_max)
            ):
                return 1.0
        return 0.0

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        if world_idx is None:
            world_idx = cumulative_idx % len(self.spoon_orientations_rpy)

        spoon_pos = self.spoon_positions[world_idx].copy()
        spoon_ori = p.getQuaternionFromEuler(self.spoon_orientations_rpy[world_idx])
        self.spoon.set_base_pose(spoon_pos, spoon_ori)
        self.start_box.set_base_pose(
            self.start_box.init_base_position, self.start_box.init_base_orientation
        )

        if self.world_random_scale is not None:
            rand_offset = np.random.uniform(
                low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
            )

            spoon_pos += rand_offset
            self.spoon.set_base_pose(spoon_pos, spoon_ori)
            box_pos = self.start_box.init_base_position.copy()
            box_pos += rand_offset
            self.start_box.set_base_pose(box_pos, self.start_box.init_base_orientation)

        return world_idx
