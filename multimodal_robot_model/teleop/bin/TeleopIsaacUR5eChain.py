import numpy as np
import gymnasium as gym
import pinocchio as pin
import multimodal_robot_model
from multimodal_robot_model.teleop import TeleopBase
from multimodal_robot_model.common import RecordStatus

class TeleopIsaacUR5eChain(TeleopBase):
    def setupEnv(self):
        self.env = gym.make(
            "multimodal_robot_model/IsaacUR5eChainEnv-v0",
            render_mode="human"
        )
        self.demo_name = "IsaacUR5eChain"

    def setArmCommand(self):
        if self.record_manager.status in (RecordStatus.PRE_REACH, RecordStatus.REACH):
            target_pos = self.env.unwrapped.get_link_pose("chain_end", "box")[0:3]
            if self.record_manager.status == RecordStatus.PRE_REACH:
                target_pos[2] += 0.22 # [m]
            elif self.record_manager.status == RecordStatus.REACH:
                target_pos[2] += 0.14 # [m]
            self.motion_manager.target_se3 = pin.SE3(np.diag([-1.0, 1.0, -1.0]), target_pos)
        else:
            super().setArmCommand()

    def setGripperCommand(self):
        if self.record_manager.status == RecordStatus.GRASP:
            self.motion_manager.gripper_pos = 150.0
        else:
            super().setGripperCommand()

if __name__ == "__main__":
    teleop = TeleopIsaacUR5eChain()
    teleop.run()