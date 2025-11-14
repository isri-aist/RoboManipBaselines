from os import path
import mujoco
import numpy as np
import math

# Assuming MujocoUR5eEnvBase is available in the environment
from .MujocoUR5eEnvBase import MujocoUR5eEnvBase


class MujocoUR5eSimplePickEnv(MujocoUR5eEnvBase):
    """
    A simplified environment for the UR5e robot to pick up a single box.
    The reward function encourages both reaching for the box and lifting it.
    """
    def __init__(
        self,
        **kwargs,
    ):
        # 1. Load the new XML environment file defining the single box
        MujocoUR5eEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                # This path assumes you saved the modified XML with this name
                "../../assets/mujoco/envs/ur5e/env_ur5e_simple_pick.xml",
            ),
            # Initial joint configuration
            np.array(
                [
                    np.pi,        # Joint 1: Shoulder Pan
                    -np.pi / 2,   # Joint 2: Shoulder Lift
                    -np.pi / 2,    # Joint 3: Elbow
                    -np.pi / 2,   # Joint 4: Wrist 1
                    np.pi / 2,    # Joint 5: Wrist 2
                    np.pi,        # Joint 6: Wrist 3
                    *np.zeros(8),

                ]

            ),
            **kwargs,
        )

        # 2. Get info for the 'target_box' free joint
        self.box_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_box_freejoint"
        )
        self.box_qpos_addr = self.model.jnt_qposadr[self.box_joint_id]
        
        # Store original position and orientation from init_qpos
        self.original_box_pos = self.init_qpos[self.box_qpos_addr : self.box_qpos_addr + 3].copy()
        
        # 3. Define randomization parameters
        self.pos_randomization_radius = 0.09 #0.09 # 9 cm radius for position
        self.orientation_randomization_range = np.pi / 2 #np.pi / 2 # +/- 90 degrees for yaw

    def get_input_device_kwargs(self, input_device_name):
        if input_device_name == "spacemouse":
            return {"rpy_scale": 1e-2}
        else:
            return super().get_input_device_kwargs(input_device_name)

    def _get_reward(self, full=True):
        """
        Calculates a reward that encourages both reaching for the box and lifting it.
        The total reward is a sum of a reaching component and a lifting component.
        """
        box_pos = self.data.body("target_box").xpos.copy()
        if full:
            # --- Reaching Reward Component ---
            
            # Get gripper and box positions.
            # Based on the provided files, we use the position of the 'wrist_3_link' body
            # as a robust proxy for the end-effector's position.
            gripper_pos = self.data.body("wrist_3_link").xpos.copy()

            # Calculate the distance between the gripper and the box.
            distance = np.linalg.norm(gripper_pos - box_pos)
            
            # The reaching reward is high when the distance is small.
            # We use `1 - tanh(x)` which maps a distance of 0 to a reward of 1,
            # and larger distances to a reward approaching 0.
            # The scaling factor of 10.0 makes the reward drop off quickly.
            reach_reward = 1 - np.tanh(2.0 * distance)

        # --- Lifting Reward Component ---

        # This part is similar to the original reward calculation.
        INITIAL_BOX_HEIGHT = 0.83978449 # The box's approximate starting Z position
        PICK_HEIGHT = 0.9                # Target Z position for a successful pick

        # Calculate a normalized lifting reward.
        lift_reward = (box_pos[2] - INITIAL_BOX_HEIGHT) / (PICK_HEIGHT - INITIAL_BOX_HEIGHT)
        
        # We only care about positive lifting, so we clip the reward at 0.
        # We also cap it at 1 for successfully reaching the pick height.
        lift_reward = np.clip(lift_reward, 0.0, 1.0)
        if full:
            if lift_reward == 1.0:
                return 1.0
            # --- Combined Reward ---
            
            # The final reward is a weighted sum of the two components.
            # This encourages the agent to first get close to the box (to get reach_reward)
            # and then lift it (to get lift_reward).
            # We give equal importance to both reaching and lifting.
            reward = (0.2 * reach_reward) + (0.8 * lift_reward)
            # print('reward: ', reward)
        else:
            reward = lift_reward
        return reward

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """
        Randomizes the position and orientation of the target box.
        """
        # Position randomization
        # Generate a random angle and radius
        rand_angle = np.random.uniform(0, 2 * np.pi)
        rand_radius = np.random.uniform(0, self.pos_randomization_radius)
        
        # Calculate dx and dy
        dx = rand_radius * np.cos(rand_angle)
        dy = rand_radius * np.sin(rand_angle)
        
        # Create new position, keeping z the same
        new_box_pos = self.original_box_pos.copy()
        new_box_pos[0] += dx
        new_box_pos[1] += dy
        
        # Orientation randomization (yaw)
        random_yaw = np.random.uniform(
            -self.orientation_randomization_range, self.orientation_randomization_range
        )
        # Create a quaternion for yaw rotation
        yaw_quat = np.array([np.cos(random_yaw / 2), 0, 0, np.sin(random_yaw / 2)])
        
        new_box_quat = yaw_quat
        
        # Update the initial state (qpos) for the box's free joint
        self.init_qpos[self.box_qpos_addr : self.box_qpos_addr + 3] = new_box_pos
        self.init_qpos[self.box_qpos_addr + 3 : self.box_qpos_addr + 7] = new_box_quat

        if world_idx is None and cumulative_idx is not None:
             world_idx = cumulative_idx % 6
        elif world_idx is None:
            world_idx = 0
            
        return world_idx

