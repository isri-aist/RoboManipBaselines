import gymnasium as gym
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import GraspPhaseBase, ReachPhaseBase


# Define the maximum bounds for the random offsets in meters
MAX_XY_RANDOMNESS = 0.10  # +/- 10 cm in X and Y
MAX_Z_RANDOMNESS = 0.05   # +/- 5 cm in Z
# Define the maximum bounds for random orientation offsets in radians (Default: 0)
MAX_ROLL_RANDOMNESS = 0.0   # +/- 0 radians
MAX_PITCH_RANDOMNESS = 0.0  # +/- 0 radians
MAX_YAW_RANDOMNESS = 0.0    # +/- 0 radians


def get_target_se3(op, delta_pos_z, roll_r=np.pi, pitch_r=0.0, yaw_r=np.pi / 2):
    """
    Calculates the target SE3 (pose) for the end-effector based on the
    target box's position, applying a vertical offset and specified RPY angles.
    """
    # target_pos = op.env.unwrapped.get_geom_pose("target_box")[0:3]
    target_pos = op.env.unwrapped.data.body("target_box").xpos.copy()
    target_pos[2] += delta_pos_z
    
    # Calculate the rotation matrix from the provided Roll, Pitch, and Yaw
    R = pin.rpy.rpyToMatrix(roll_r, pitch_r, yaw_r)
    return pin.SE3(R, target_pos)


class ReachPhase1(ReachPhaseBase):
    def set_target(self):
        # Nominal RPY angles for end-effector pointing down/straight
        nominal_roll = np.pi
        nominal_pitch = 0.0
        nominal_yaw = np.pi / 2
        
        # 1. Generate random offsets for translation
        # np.random.uniform generates a random float in the range [low, high)
        random_dx = np.random.uniform(-MAX_XY_RANDOMNESS, MAX_XY_RANDOMNESS)
        random_dy = np.random.uniform(-MAX_XY_RANDOMNESS, MAX_XY_RANDOMNESS)
        random_dz = np.random.uniform(-MAX_Z_RANDOMNESS, MAX_Z_RANDOMNESS)

        # 2. Generate random offsets for orientation (currently zero range, as requested)
        random_roll = np.random.uniform(-MAX_ROLL_RANDOMNESS, MAX_ROLL_RANDOMNESS)
        random_pitch = np.random.uniform(-MAX_PITCH_RANDOMNESS, MAX_PITCH_RANDOMNESS)
        random_yaw = np.random.uniform(-MAX_YAW_RANDOMNESS, MAX_YAW_RANDOMNESS)
        
        # 3. Calculate the target SE3 with randomized orientation
        self.target_se3 = get_target_se3(
            self.op,
            delta_pos_z=0.3,  # [m] Nominal height above the cube
            roll_r=nominal_roll + random_roll,
            pitch_r=nominal_pitch + random_pitch,
            yaw_r=nominal_yaw + random_yaw,
        )

        # 4. Apply the random translational offsets
        # This makes the initial approach randomized for better generalization
        self.target_se3.translation[0] += random_dx
        self.target_se3.translation[1] += random_dy
        self.target_se3.translation[2] += random_dz

        self.duration = 0.7  # [s]


class ReachPhase2(ReachPhaseBase):
    def set_target(self):
        self.target_se3 = get_target_se3(
            self.op,
            delta_pos_z=0.15,  # [m] just above enough for the gripper to grasp
        )
        self.duration = 0.3  # [s]


class GraspPhase(GraspPhaseBase):
    def set_target(self):
        # The original request had both set_target_close() and set_target_open() commented out
        # I am defaulting to open (as in the original GraspPhaseBase intention)
        self.set_target_open() 


class OperationMujocoUR5eSimplePick:
    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/MujocoUR5eSimplePickEnv-v0", render_mode=render_mode
        )

    def get_pre_motion_phases(self):
        return [
            ReachPhase1(self), # go randomly above the cube
            # ReachPhase2(self), # go to the cube (currently commented out)
            GraspPhase(self), # open/close the gripper (defaulting to open target)
        ]
