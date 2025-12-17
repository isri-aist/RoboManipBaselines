import gymnasium as gym
import numpy as np

from robo_manip_baselines.common import GraspPhaseBase


class GraspPhase(GraspPhaseBase):
    def set_target(self):
        self.gripper_joint_pos = np.array([0])
        self.duration = 0.5  # [s]

class OperationRealUR10eDemo:
    def __init__(self, robot_ip, camera_ids, gelsight_ids=None):
        self.robot_ip = robot_ip
        self.camera_ids = camera_ids
        self.gelsight_ids = gelsight_ids
        super().__init__()

    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/RealUR10eDemoEnv-v0",
            robot_ip=self.robot_ip,
            camera_ids=self.camera_ids,
            gelsight_ids=self.gelsight_ids,
        )

    def get_pre_motion_phases(self):
        return [GraspPhase(self)]
