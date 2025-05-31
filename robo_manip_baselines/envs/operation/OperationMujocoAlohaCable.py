import gymnasium as gym
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import (
    ReachPhaseBase,
    get_pose_from_se3,
)


def get_target_se3(op, pos_z, rot_y):
    left_target_pos = op.env.unwrapped.get_body_pose("cable_B0")[0:3]
    left_target_pos[1] += 0.05  # [m]
    left_target_pos[2] = pos_z
    left_target_pose = get_pose_from_se3(
        pin.SE3(pin.rpy.rpyToMatrix(0.0, rot_y, -np.pi / 2), left_target_pos)
    )

    right_target_pose = get_pose_from_se3(
        op.motion_manager.body_manager_list[1].current_se3
    )

    return np.concatenate([left_target_pose, right_target_pose])


class ReachPhase1(ReachPhaseBase):
    def set_target(self):
        self.target_se3 = get_target_se3(
            self.op,
            pos_z=0.3,  # [m]
            rot_y=np.deg2rad(30),  # [rad]
        )
        self.duration = 0.7  # [s]


class ReachPhase2(ReachPhaseBase):
    def set_target(self):
        self.target_se3 = get_target_se3(
            self.op,
            pos_z=0.2,  # [m]
            rot_y=np.deg2rad(60),  # [rad]
        )
        self.duration = 0.3  # [s]


class OperationMujocoAlohaCable:
    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/MujocoAlohaCableEnv-v0", render_mode=render_mode
        )

    def get_pre_motion_phases(self):
        return [
            ReachPhase1(self),
            ReachPhase2(self),
        ]
