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


class RealXarm7DualEnvBase(RealEnvBase):
    action_space = Box(
        low=np.array(
            [
                -2 * np.pi,
                np.deg2rad(-118),
                -2 * np.pi,
                np.deg2rad(-11),
                -2 * np.pi,
                np.deg2rad(-97),
                -2 * np.pi,
                0.0,
            ]
            * 2,
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
            ]
            * 2,
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    observation_space = Dict(
        {
            "joint_pos": Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float64),
            "joint_vel": Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float64),
            "wrench": Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64),
        }
    )

    def __init__(
        self,
        robot_ip_left,
        robot_ip_right,
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
            ),
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__),
                    "../../assets/common/robots/xarm7/xarm7.urdf",
                ),
                arm_root_pose=None,
                ik_eef_joint_id=7,
                arm_joint_idxes=np.arange(8, 15),
                gripper_joint_idxes=np.array([15]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([1]),
                eef_idx=1,
                init_arm_joint_pos=self.init_qpos[8:15],
                init_gripper_joint_pos=np.zeros(1),
            ),
        ]

        # Connect to xArm7Dual
        print(f"[{self.__class__.__name__}] Start connecting the xArm7Dual.")

        self.robot_ip_left = robot_ip_left
        self.xarm_api_left = XArmAPI(self.robot_ip_left)
        self.xarm_api_left.connect()
        self.xarm_api_left.motion_enable(enable=True)
        self.xarm_api_left.set_ft_sensor_enable(1)
        time.sleep(0.2)
        self.xarm_api_left.set_ft_sensor_zero()
        time.sleep(0.2)
        self.xarm_api_left.clean_error()
        self.xarm_api_left.set_mode(6)
        self.xarm_api_left.set_state(0)
        self.xarm_api_left.set_collision_sensitivity(1)
        self.xarm_api_left.clean_gripper_error()
        self.xarm_api_left.set_gripper_mode(0)
        self.xarm_api_left.set_gripper_enable(True)

        time.sleep(0.2)
        xarm_code, left_joint_states = self.xarm_api_left.get_joint_states(
            is_radian=True
        )
        if xarm_code != 0:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Invalid xArm API code: {xarm_code}"
            )

        self.robot_ip_right = robot_ip_right
        self.xarm_api_right = XArmAPI(self.robot_ip_right)
        self.xarm_api_right.connect()
        self.xarm_api_right.motion_enable(enable=True)
        self.xarm_api_right.set_ft_sensor_enable(1)
        time.sleep(0.2)
        self.xarm_api_right.set_ft_sensor_zero()
        time.sleep(0.2)
        self.xarm_api_right.clean_error()
        self.xarm_api_right.set_mode(6)
        self.xarm_api_right.set_state(0)
        self.xarm_api_right.set_collision_sensitivity(1)
        self.xarm_api_right.clean_gripper_error()
        self.xarm_api_right.set_gripper_mode(0)
        self.xarm_api_right.set_gripper_enable(True)
        time.sleep(0.2)
        xarm_code, right_joint_states = self.xarm_api_right.get_joint_states(
            is_radian=True
        )
        if xarm_code != 0:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Invalid xArm API code: {xarm_code}"
            )

        self.arm_joint_pos_actual = np.concatenate(
            [left_joint_states[0], right_joint_states[0]]
        )

        print(f"[{self.__class__.__name__}] Finish connecting the xArm7Dual.")

        # Connect to RealSense
        self.setup_realsense(camera_ids)
        self.setup_gelsight(gelsight_ids)

    def close(self):
        self.xarm_api_left.disconnect()
        self.xarm_api_right.disconnect()

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
                body_manager,
                **{
                    **default_kwargs.get(device_idx, {}),
                    **overwrite_kwargs.get(device_idx, {}),
                },
            )
            for device_idx, body_manager in enumerate(motion_manager.body_manager_list)
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

        # Send command to Xarm7Dual
        left_arm_joint_pos_command = action[self.body_config_list[0].arm_joint_idxes]
        right_arm_joint_pos_command = action[self.body_config_list[1].arm_joint_idxes]
        scaled_joint_vel_limit = (
            np.clip(joint_vel_limit_scale, 0.01, 10.0) * self.joint_vel_limit
        )

        # send command to the left arm
        left_xarm_code = self.xarm_api_left.set_servo_angle(
            angle=left_arm_joint_pos_command,
            speed=scaled_joint_vel_limit,
            mvtime=duration,
            is_radian=True,
            wait=False,
        )
        # send command to the right arm
        right_xarm_code = self.xarm_api_right.set_servo_angle(
            angle=right_arm_joint_pos_command,
            speed=scaled_joint_vel_limit,
            mvtime=duration,
            is_radian=True,
            wait=False,
        )
        if left_xarm_code != 0:
            left_err = self._format_err_warn(self.xarm_api_left, "left")
            raise RuntimeError(
                f"[{self.__class__.__name__}] Invalid xArm API code: {left_xarm_code} ({left_err})"
            )

        if right_xarm_code != 0:
            right_err = self._format_err_warn(self.xarm_api_right, "right")
            raise RuntimeError(
                f"[{self.__class__.__name__}] Invalid xArm API code: {right_xarm_code} ({right_err})"
            )

        # Send command to xArm gripper
        left_gripper_pos = action[self.body_config_list[0].gripper_joint_idxes][0]
        right_gripper_pos = action[self.body_config_list[1].gripper_joint_idxes][0]

        xarm_code = self.xarm_api_left.set_gripper_position(
            left_gripper_pos, wait=False
        )
        if xarm_code != 0:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Invalid xArm API code: {xarm_code}"
            )

        xarm_code = self.xarm_api_right.set_gripper_position(
            right_gripper_pos, wait=False
        )
        if xarm_code != 0:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Invalid xArm API code: {xarm_code}"
            )

        # Wait
        elapsed_duration = time.time() - start_time
        if wait and elapsed_duration < duration:
            time.sleep(duration - elapsed_duration)

    def _get_obs(self):
        # Get state from xArm7Dual
        left_code, left_joint_states = self.xarm_api_left.get_joint_states(
            is_radian=True
        )
        if left_code != 0:
            left_err = self._format_err_warn(self.xarm_api_left, "left")
            raise RuntimeError(
                f"[{self.__class__.__name__}] Invalid xArm API code: {left_code} ({left_err})"
            )
        right_code, right_joint_states = self.xarm_api_right.get_joint_states(
            is_radian=True
        )
        if right_code != 0:
            right_err = self._format_err_warn(self.xarm_api_right, "right")
            raise RuntimeError(
                f"[{self.__class__.__name__}] Invalid xArm API code: {right_code} ({right_err})"
            )
        left_arm_joint_pos = left_joint_states[0]
        left_arm_joint_vel = left_joint_states[1]
        right_arm_joint_pos = right_joint_states[0]
        right_arm_joint_vel = right_joint_states[1]

        self.arm_joint_pos_actual = np.concatenate(
            [left_arm_joint_pos, right_arm_joint_pos], dtype=np.float64
        )

        # Get state from UFactory gripper
        xarm_code, left_gripper_pos = self.xarm_api_left.get_gripper_position()
        if xarm_code != 0:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Invalid xArm API code: {xarm_code}"
            )
        left_gripper_joint_pos = np.array([left_gripper_pos], dtype=np.float64)
        left_gripper_joint_vel = np.zeros(1)

        xarm_code, right_gripper_pos = self.xarm_api_right.get_gripper_position()
        if xarm_code != 0:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Invalid xArm API code: {xarm_code}"
            )
        right_gripper_joint_pos = np.array([right_gripper_pos], dtype=np.float64)
        right_gripper_joint_vel = np.zeros(1)

        # Get wrench from force sensor
        wrench_left = np.array(
            self.xarm_api_left.get_ft_sensor_data()[1], dtype=np.float64
        )
        wrench_right = np.array(
            self.xarm_api_right.get_ft_sensor_data()[1], dtype=np.float64
        )
        force = np.concatenate((wrench_left[0:3], wrench_right[0:3]), dtype=np.float64)
        torque = np.concatenate((wrench_left[3:6], wrench_right[3:6]), dtype=np.float64)

        return {
            "joint_pos": np.concatenate(
                (
                    left_arm_joint_pos,
                    left_gripper_joint_pos,
                    right_arm_joint_pos,
                    right_gripper_joint_pos,
                ),
                dtype=np.float64,
            ),
            "joint_vel": np.concatenate(
                (
                    left_arm_joint_vel,
                    left_gripper_joint_vel,
                    right_arm_joint_vel,
                    right_gripper_joint_vel,
                ),
                dtype=np.float64,
            ),
            "wrench": np.concatenate((force, torque), dtype=np.float64),
        }

    def _format_err_warn(self, xarm_api, arm_label):
        get_code, err_warn = xarm_api.get_err_warn_code()
        # err_warn is [err, warn] if get_code == 0, otherwise cached values
        err_code, warn_code = (
            err_warn
            if isinstance(err_warn, (list, tuple)) and len(err_warn) >= 2
            else (None, None)
        )
        mode = getattr(xarm_api, "mode", None)
        state = getattr(xarm_api, "state", None)
        return f"{arm_label}: err={err_code}, warn={warn_code}, get_err_ret={get_code}, mode={mode}, state={state}"
