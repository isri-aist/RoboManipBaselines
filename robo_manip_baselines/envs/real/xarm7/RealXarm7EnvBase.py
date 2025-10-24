import time
from os import path

import numpy as np
from gymnasium.spaces import Box, Dict
from xarm.wrapper import XArmAPI

from robo_manip_baselines.common import ArmConfig
from robo_manip_baselines.teleop import (
    GelloInputDevice,
    KeyboardInputDevice,
    SpacemouseInputDevice,
)

from ..RealEnvBase import RealEnvBase


class RealXarm7EnvBase(RealEnvBase):
    action_space = Box(
        low=np.deg2rad(
            [
                -2 * np.pi,
                np.deg2rad(-118),
                -2 * np.pi,
                np.deg2rad(-11),
                -2 * np.pi,
                np.deg2rad(-97),
                -2 * np.pi,
                0.0,
            ],
            dtype=np.float32,
        ),
        high=np.array(
            [
                2 * np.pi,
                np.deg2rad(120),
                2 * np.pi,
                np.deg2rad(225),
                2 * np.pi,
                np.pi,
                2 * np.pi,
                840.0,
            ],
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    observation_space = Dict(
        {
            "joint_pos": Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64),
            "joint_vel": Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64),
            "wrench": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
        }
    )

    def __init__(
        self,
        robot_ip,
        camera_ids,
        gelsight_ids,
        init_qpos,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Setup robot
        self.init_qpos = init_qpos
        self.joint_vel_limit = np.deg2rad(180)  # [rad/s]
        self.body_config_list = [
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__),
                    "../../assets/common/robots/xarm7/xarm7.urdf",
                ),
                arm_root_pose=None,
                ik_eef_joint_id=7,
                arm_joint_idxes=np.arange(7),
                gripper_joint_idxes=np.array([7]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([0]),
                eef_idx=0,
                init_arm_joint_pos=self.init_qpos[0:7],
                init_gripper_joint_pos=np.zeros(1),
            )
        ]

        # Connect to xArm7
        print(f"[{self.__class__.__name__}] Start connecting the xArm7.")
        self.robot_ip = robot_ip
        self.xarm_api = XArmAPI(self.robot_ip)
        self.xarm_api.connect()
        self.xarm_api.motion_enable(enable=True)
        self.xarm_api.ft_sensor_enable(1)
        time.sleep(0.2)
        self.xarm_api.ft_sensor_set_zero()
        time.sleep(0.2)
        self.xarm_api.clean_error()
        self.xarm_api.set_mode(6)
        self.xarm_api.set_state(0)
        self.xarm_api.set_collision_sensitivity(1)
        self.xarm_api.clean_gripper_error()
        self.xarm_api.set_gripper_mode(0)
        self.xarm_api.set_gripper_enable(True)
        time.sleep(0.2)
        xarm_code, joint_states = self.xarm_api.get_joint_states(is_radian=True)
        if xarm_code != 0:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Invalid xArm API code: {xarm_code}"
            )
        self.arm_joint_pos_actual = joint_states[0]
        print(f"[{self.__class__.__name__}] Finish connecting the xArm7.")

        # Connect to RealSense
        self.setup_realsense(camera_ids)
        self.setup_gelsight(gelsight_ids)

    def close(self):
        self.xarm_api.disconnect()

    def setup_input_device(self, input_device_name, motion_manager, overwrite_kwargs):
        if input_device_name == "spacemouse":
            InputDeviceClass = SpacemouseInputDevice
        elif input_device_name == "gello":
            InputDeviceClass = GelloInputDevice
        elif input_device_name == "keyboard":
            InputDeviceClass = KeyboardInputDevice
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid input device key: {input_device_name}"
            )

        default_kwargs = self.get_input_device_kwargs(input_device_name)

        return [
            InputDeviceClass(
                motion_manager.body_manager_list[0],
                **{**default_kwargs, **overwrite_kwargs},
            )
        ]

    def get_input_device_kwargs(self, input_device_name):
        return {}

    def _reset_robot(self):
        print(
            f"[{self.__class__.__name__}] Start moving the robot to the reset position."
        )
        self._set_action(
            self.init_qpos, duration=None, joint_vel_limit_scale=0.1, wait=True
        )
        print(
            f"[{self.__class__.__name__}] Finish moving the robot to the reset position."
        )

    def _set_action(self, action, duration=None, joint_vel_limit_scale=0.5, wait=False):
        start_time = time.time()

        # Overwrite duration or joint_pos for safety
        action, duration = self.overwrite_command_for_safety(
            action, duration, joint_vel_limit_scale
        )

        # Send command to xArm7
        arm_joint_pos_command = action[self.body_config_list[0].arm_joint_idxes]
        scaled_joint_vel_limit = (
            np.clip(joint_vel_limit_scale, 0.01, 10.0) * self.joint_vel_limit
        )

        if True:
            xarm_code = self.xarm_api.set_servo_angle_j(
                arm_joint_pos_command,
                speed=scaled_joint_vel_limit,  # set_servo_angle_j expects deg/s
                is_radian=True,
            )
        else:
            xarm_code = self.xarm_api.set_servo_angle(
                angle=arm_joint_pos_command,
                speed=scaled_joint_vel_limit,
                mvtime=duration,
                is_radian=True,
                wait=False,
            )
        if xarm_code != 0:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Invalid xArm API code: {xarm_code}"
            )

        # Send command to xArm gripper
        gripper_pos = action[self.body_config_list[0].gripper_joint_idxes][0]
        xarm_code = self.xarm_api.set_gripper_position(gripper_pos, wait=False)
        if xarm_code != 0:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Invalid xArm API code: {xarm_code}"
            )

        # Wait
        elapsed_duration = time.time() - start_time
        if wait and elapsed_duration < duration:
            time.sleep(duration - elapsed_duration)

    def _get_obs(self):
        # Get state from xArm7
        xarm_code, joint_states = self.xarm_api.get_joint_states(is_radian=True)
        if xarm_code != 0:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Invalid xArm API code: {xarm_code}"
            )
        arm_joint_pos = joint_states[0]
        arm_joint_vel = joint_states[1]
        self.arm_joint_pos_actual = arm_joint_pos.copy()

        # Get state from Robotiq gripper
        xarm_code, gripper_pos = self.xarm_api.get_gripper_position()
        if xarm_code != 0:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Invalid xArm API code: {xarm_code}"
            )
        gripper_joint_pos = np.array([gripper_pos], dtype=np.float64)
        gripper_joint_vel = np.zeros(1)

        # Get wrench from force sensor
        wrench = np.array(self.xarm_api.get_ft_sensor_data()[1], dtype=np.float64)
        force = wrench[0:3]
        torque = wrench[3:6]

        return {
            "joint_pos": np.concatenate(
                (arm_joint_pos, gripper_joint_pos), dtype=np.float64
            ),
            "joint_vel": np.concatenate(
                (arm_joint_vel, gripper_joint_vel), dtype=np.float64
            ),
            "wrench": np.concatenate((force, torque), dtype=np.float64),
        }
