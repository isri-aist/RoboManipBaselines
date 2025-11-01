import time
from os import path

import numpy as np
import rtde_control
import rtde_receive
from gello.robots.robotiq_gripper import RobotiqGripper
from gymnasium.spaces import Box, Dict

from robo_manip_baselines.common import ArmConfig
from robo_manip_baselines.teleop import (
    GelloInputDevice,
    KeyboardInputDevice,
    SpacemouseInputDevice,
)

from ..RealEnvBase import RealEnvBase


class RealUR5eEnvBase(RealEnvBase):
    action_space = Box(
        low=np.array(
            [
                -2 * np.pi,
                -2 * np.pi,
                -1 * np.pi,
                -2 * np.pi,
                -2 * np.pi,
                -2 * np.pi,
                0.0,
            ],
            dtype=np.float32,
        ),
        high=np.array(
            [
                2 * np.pi,
                2 * np.pi,
                1 * np.pi,
                2 * np.pi,
                2 * np.pi,
                2 * np.pi,
                255.0,
            ],
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    observation_space = Dict(
        {
            "joint_pos": Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
            "joint_vel": Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
            "wrench": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
        }
    )

    def __init__(
        self,
        robot_ip,
        camera_ids,
        pointcloud_camera_ids,
        gelsight_ids,
        sanwa_keyboard_ids,
        init_qpos,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Setup robot
        self.init_qpos = init_qpos
        self.joint_vel_limit = np.deg2rad(191)  # [rad/s]
        self.body_config_list = [
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__), "../../assets/common/robots/ur5e/ur5e.urdf"
                ),
                arm_root_pose=None,
                ik_eef_joint_id=6,
                arm_joint_idxes=np.arange(6),
                gripper_joint_idxes=np.array([6]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([0]),
                eef_idx=0,
                init_arm_joint_pos=self.init_qpos[0:6],
                init_gripper_joint_pos=np.zeros(1),
            )
        ]

        # Connect to UR5e
        print(f"[{self.__class__.__name__}] Start connecting the UR5e.")
        self.robot_ip = robot_ip
        self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        self.rtde_c.endFreedriveMode()
        self.arm_joint_pos_actual = np.array(self.rtde_r.getActualQ())
        print(f"[{self.__class__.__name__}] Finish connecting the UR5e.")

        # Connect to Robotiq gripper
        print(f"[{self.__class__.__name__}] Start connecting the Robotiq gripper.")
        self.gripper_port = 63352
        self.gripper = RobotiqGripper()
        self.gripper.connect(hostname=self.robot_ip, port=self.gripper_port)
        self._gripper_activated = False
        print(f"[{self.__class__.__name__}] Finish connecting the Robotiq gripper.")

        # Connect to RealSense
        self.setup_realsense(camera_ids)
        self.setup_femtobolt(pointcloud_camera_ids)
        self.setup_gelsight(gelsight_ids)
        self.setup_sanwa_keyboard(sanwa_keyboard_ids)

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
        if input_device_name == "spacemouse":
            return {"pos_scale": 1.5e-2, "rpy_scale": 1e-2, "gripper_scale": 10.0}
        else:
            return super().get_input_device_kwargs(input_device_name)

    def _reset_robot(self):
        print(
            f"[{self.__class__.__name__}] Start moving the robot to the reset position."
        )
        self._set_action(
            self.init_qpos, duration=None, joint_vel_limit_scale=0.3, wait=True
        )
        print(
            f"[{self.__class__.__name__}] Finish moving the robot to the reset position."
        )

        if not self._gripper_activated:
            self._gripper_activated = True
            print(f"[{self.__class__.__name__}] Start activating the Robotiq gripper.")
            self.gripper.activate()
            print(f"[{self.__class__.__name__}] Finish activating the Robotiq gripper.")

        # Calibrate force sensor
        time.sleep(0.2)
        self.rtde_c.zeroFtSensor()
        time.sleep(0.2)

    def _set_action(self, action, duration=None, joint_vel_limit_scale=0.5, wait=False):
        start_time = time.time()

        # Overwrite duration or joint_pos for safety
        action, duration = self.overwrite_command_for_safety(
            action, duration, joint_vel_limit_scale
        )

        # Send command to UR5e
        arm_joint_pos_command = action[self.body_config_list[0].arm_joint_idxes]
        velocity = 0.5
        acceleration = 0.5
        lookahead_time = 0.2  # [s]
        gain = 100
        period = self.rtde_c.initPeriod()
        self.rtde_c.servoJ(
            arm_joint_pos_command,
            velocity,
            acceleration,
            duration,
            lookahead_time,
            gain,
        )
        self.rtde_c.waitPeriod(period)

        # Send command to Robotiq gripper
        gripper_pos = action[self.body_config_list[0].gripper_joint_idxes][0]
        speed = 50
        force = 10
        self.gripper.move(int(gripper_pos), speed, force)

        # Wait
        elapsed_duration = time.time() - start_time
        if wait and elapsed_duration < duration:
            time.sleep(duration - elapsed_duration)

    def _get_obs(self):
        # Get state from UR5e
        arm_joint_pos = np.array(self.rtde_r.getActualQ())
        arm_joint_vel = np.array(self.rtde_r.getActualQd())
        self.arm_joint_pos_actual = arm_joint_pos.copy()

        # Get state from Robotiq gripper
        gripper_joint_pos = np.array(
            [self.gripper.get_current_position()], dtype=np.float64
        )
        gripper_joint_vel = np.zeros(1)

        # Get wrench from force sensor
        wrench = np.array(self.rtde_r.getActualTCPForce(), dtype=np.float64)

        return {
            "joint_pos": np.concatenate(
                (arm_joint_pos, gripper_joint_pos), dtype=np.float64
            ),
            "joint_vel": np.concatenate(
                (arm_joint_vel, gripper_joint_vel), dtype=np.float64
            ),
            "wrench": wrench,
        }
