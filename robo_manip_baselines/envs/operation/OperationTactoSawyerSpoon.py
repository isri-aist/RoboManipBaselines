import gymnasium as gym
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import GraspPhaseBase, ReachPhaseBase


def get_target_se3(op, pos_z):
    target_pos = np.array([-0.52, 0.054, 0.5])
    target_pos[2] = pos_z
    return pin.SE3(np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]), target_pos)


class ReachPhase1(ReachPhaseBase):
    def set_target(self):
        self.target_se3 = get_target_se3(self.op, pos_z=0.7)

        self.duration = 0.4


class GraspPhase1(GraspPhaseBase):
    def set_target(self):
        self.gripper_joint_pos = np.array([0.04])
        self.duration = 0.2


class ReachPhase2(ReachPhaseBase):
    def set_target(self):
        self.target_se3 = get_target_se3(self.op, pos_z=0.65)

        self.duration = 0.4


class GraspPhase2(GraspPhaseBase):
    def set_target(self):
        self.gripper_joint_pos = np.array([0.01])
        self.duration = 0.2


class OperationTactoSawyerSpoon:
    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/TactoSawyerSpoonEnv-v0",
            render_mode=render_mode,
        )

    def get_pre_motion_phases(self):
        return [
            ReachPhase1(self),
            GraspPhase1(self),
            ReachPhase2(self),
            GraspPhase2(self),
        ]
