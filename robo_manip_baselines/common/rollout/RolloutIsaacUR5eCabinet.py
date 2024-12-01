import numpy as np
import gymnasium as gym
from robo_manip_baselines.common import MotionStatus
from .RolloutBase import RolloutBase


class RolloutIsaacUR5eCabinet(RolloutBase):
    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/IsaacUR5eCabinetEnv-v0", render_mode="human"
        )

    def set_arm_command(self):
        if self.data_manager.status in (MotionStatus.PRE_REACH, MotionStatus.REACH):
            target_pos = self.env.unwrapped.get_link_pose("ur5e", "base_link")[0:3]
            if self.data_manager.status == MotionStatus.PRE_REACH:
                target_pos += np.array([0.33, 0.0, 0.3])  # [m]
            elif self.data_manager.status == MotionStatus.REACH:
                target_pos += np.array([0.38, 0.0, 0.3])  # [m]
            self.motion_manager.target_se3.translation = target_pos
            self.motion_manager.inverse_kinematics()
        else:
            super().set_arm_command()