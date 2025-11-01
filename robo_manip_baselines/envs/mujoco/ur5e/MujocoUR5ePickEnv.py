from os import path

import mujoco
import numpy as np

from .MujocoUR5eEnvBase import MujocoUR5eEnvBase


class MujocoUR5ePickEnv(MujocoUR5eEnvBase):
    sim_timestep = 0.002
    frame_skip = 16
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": int(1 / (sim_timestep * frame_skip)),
    }

    def __init__(
        self,
        **kwargs,
    ):
        MujocoUR5eEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/ur5e/env_ur5e_pick.xml",
            ),
            np.array(
                [
                    np.pi,
                    -np.pi / 2,
                    -0.55 * np.pi,
                    -0.45 * np.pi,
                    np.pi / 2,
                    np.pi,
                    *np.zeros(8),
                ]
            ),
            **kwargs,
        )

        self.obj_name_list = [f"obj{i}" for i in range(1, 10)]
        self.original_obj_pos_list = [
            self.model.body(obj_name).pos.copy() for obj_name in self.obj_name_list
        ]
        self.reversed_obj_pos_list = [
            obj_pos + np.array([0.0, 0.3, 0.0])
            for obj_pos in self.original_obj_pos_list
        ]
        self.basket_name_list = [f"basket{i}" for i in range(1, 3)]
        self.original_basket_pos_list = [
            self.model.body(basket_name).pos.copy()
            for basket_name in self.basket_name_list
        ]
        self.reversed_basket_pos_list = [
            basket_pos + np.array([0.0, -0.6, 0.0])
            for basket_pos in self.original_basket_pos_list
        ]
        self.basket_pos_offsets = np.array(
            [
                [-0.09, 0.0, 0.0],
                [-0.06, 0.0, 0.0],
                [-0.03, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.03, 0.0, 0.0],
                [0.06, 0.0, 0.0],
            ]
        )  # [m]

        # Either None or "<obj_name>_<basket_name>"
        # (where <obj_name> is "obj1"-"obj9" and <basket_name> is "basket1"-"basket2")
        self.target_task = None
        self.world_random_factors = []

    def _get_reward(self):
        obj_pos_list = {
            obj_name: self.data.body(obj_name).xpos.copy()
            for obj_name in self.obj_name_list
        }
        basket_pos_list = {
            basket_name: self.data.body(basket_name).xpos.copy()
            for basket_name in self.basket_name_list
        }
        basket_half_extents = np.array([0.12, 0.24, 0.2])  # [m]

        reward = 0.0
        for obj_name, obj_pos in obj_pos_list.items():
            for basket_name, basket_pos in basket_pos_list.items():
                if (
                    self.target_task is not None
                    and self.target_task != f"{obj_name}_{basket_name}"
                ):
                    continue
                if np.all(np.abs(obj_pos - basket_pos) <= basket_half_extents):
                    reward += 1.0

        return reward

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.basket_pos_offsets)

        if self.world_random_factors is not None:
            supported_world_random_factors = {
                "shuffle_side_of_baskets",
                "shuffle_object_pos",
                "shuffle_basket_pos",
            }
            invalid_world_random_factors = (
                set(self.world_random_factors) - supported_world_random_factors
            )
            if invalid_world_random_factors:
                raise ValueError(
                    f"[{self.__class__.__name__}] Invalid world_random_factors: {invalid_world_random_factors}."
                )

        if "shuffle_side_of_baskets" in self.world_random_factors:
            all_obj_pos_lists = [self.original_obj_pos_list, self.reversed_obj_pos_list]
            all_basket_pos_lists = [
                self.original_basket_pos_list,
                self.reversed_basket_pos_list,
            ]
            current_pos_list_idx = np.random.randint(0, 2)
            current_obj_pos_list = all_obj_pos_lists[current_pos_list_idx]
            current_basket_pos_list = all_basket_pos_lists[current_pos_list_idx]
        else:
            current_obj_pos_list = self.original_obj_pos_list
            current_basket_pos_list = self.original_basket_pos_list
        if "shuffle_object_pos" in self.world_random_factors:
            current_obj_pos_list = self.shuffle_object_pos(current_obj_pos_list)
        if "shuffle_basket_pos" in self.world_random_factors:
            current_basket_pos_list = self.shuffle_object_pos(current_basket_pos_list)

        basket_pos_offset = self.basket_pos_offsets[world_idx]
        for basket_name, basket_pos in zip(
            self.basket_name_list, current_basket_pos_list
        ):
            self.model.body(basket_name).pos = basket_pos + basket_pos_offset

        if (self.world_random_factors is not None) or (
            self.world_random_scale is not None
        ):
            for obj_name, obj_pos in zip(self.obj_name_list, current_obj_pos_list):
                if self.world_random_scale is None:
                    obj_pos_with_offset = obj_pos
                else:
                    obj_pos_offset = np.random.uniform(
                        low=-1.0 * self.world_random_scale,
                        high=self.world_random_scale,
                        size=3,
                    )
                    obj_pos_with_offset = obj_pos + obj_pos_offset

                obj_joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{obj_name}_freejoint"
                )
                obj_qpos_addr = self.model.jnt_qposadr[obj_joint_id]
                self.init_qpos[obj_qpos_addr : obj_qpos_addr + 3] = obj_pos_with_offset

        return world_idx

    def shuffle_object_pos(self, obj_pos_list):
        swapped_obj_pos_list = np.array(obj_pos_list)
        np.random.shuffle(swapped_obj_pos_list)
        return swapped_obj_pos_list
