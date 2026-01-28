import time
from os import path

import numpy as np
from franky import (
    Duration,
    Gripper,
    JointMotion,
    JointState,
    JointWaypoint,
    JointWaypointMotion,
    Robot,
)
from gymnasium.spaces import Box, Dict

from robo_manip_baselines.common import ArmConfig
from robo_manip_baselines.teleop import (
    GelloInputDevice,
    KeyboardInputDevice,
    SpacemouseInputDevice,
)

from ..RealEnvBase import RealEnvBase


class RealFR3EnvBase(RealEnvBase):
    action_space = Box(
        low=np.array(
            [
                -2.3093,
                -1.5133,
                -2.4937,
                -2.7478,
                -2.4800,
                0.8521,
                -2.6895,
                0.0,
            ],
            dtype=np.float32,
        ),
        high=np.array(
            [
                2.3093,
                1.5133,
                2.4937,
                -0.4461,
                2.4800,
                4.2094,
                2.6895,
                0.08,
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
        pointcloud_camera_ids,
        gelsight_ids,
        sanwa_keyboard_ids,
        init_qpos,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Setup robot
        self.init_qpos = init_qpos
        self.joint_vel_limit = 1.0  # [rad/s]
        self.body_config_list = [
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__),
                    "../../assets/common/robots/panda/panda.urdf",
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

        # Connect to FR3
        print(f"[{self.__class__.__name__}] Start connecting the FR3.")
        self.robot_ip = robot_ip
        self.robot = Robot(self.robot_ip)
        self.robot.relative_dynamics_factor = 0.3
        self.arm_joint_pos_actual = np.array(self.robot.state.q)
        self.arm_joint_pos_command = np.array(self.robot.state.q_d)
        print(f"[{self.__class__.__name__}] Finish connecting the FR3.")

        # Connect to Robotiq gripper
        print(f"[{self.__class__.__name__}] Start connecting the Robotiq gripper.")
        self.gripper = Gripper(self.robot_ip)
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

    def _set_action(self, action, duration=None, joint_vel_limit_scale=0.5, wait=False):
        start_time = time.time()

        # Since franky does not have an API to specify duration,
        # duration is ignored and only joint_vel_limit_scale is considered.
        joint_vel_limit_scale = np.clip(joint_vel_limit_scale, 1e-3, 0.3)
        self.robot.relative_dynamics_factor = joint_vel_limit_scale

        # Overwrite duration or joint_pos for safety
        action, duration = self.overwrite_command_for_safety(
            action, duration, joint_vel_limit_scale
        )

        # Send command to FR3
        arm_joint_pos_command = action[self.body_config_list[0].arm_joint_idxes]
        if duration > self.dt:
            self.robot.move(JointMotion(arm_joint_pos_command), asynchronous=True)
        else:
            arm_joint_vel_command = (
                arm_joint_pos_command - self.arm_joint_pos_command
            ) / duration
            self.robot.move(
                JointWaypointMotion(
                    [
                        JointWaypoint(
                            JointState(arm_joint_pos_command, arm_joint_vel_command),
                            minimum_time=Duration(int(1e3 * duration)),
                        ),
                        JointWaypoint(
                            JointState(
                                arm_joint_pos_command
                                + duration * arm_joint_vel_command,
                                arm_joint_vel_command,
                            )
                        ),
                    ]
                ),
                asynchronous=True,
            )
        self.arm_joint_pos_command = arm_joint_pos_command.copy()

        # Send command to Robotiq gripper
        gripper_pos = action[self.body_config_list[0].gripper_joint_idxes][0]
        gripper_speed = 0.1  # [m/s]
        self.gripper.move_async(gripper_pos, gripper_speed)

        # Wait
        elapsed_duration = time.time() - start_time
        if wait and duration is not None and elapsed_duration < duration:
            time.sleep(duration - elapsed_duration)

    def _get_obs(self):
        # Get state from FR3
        arm_joint_pos = np.array(self.robot.state.q)
        arm_joint_vel = np.array(self.robot.state.dq)
        self.arm_joint_pos_actual = arm_joint_pos.copy()

        # Get state from Robotiq gripper
        gripper_joint_pos = np.array([self.gripper.width], dtype=np.float64)
        gripper_joint_vel = np.zeros(1)

        # Get wrench from force sensor
        wrench = np.array(self.robot.state.K_F_ext_hat_K, dtype=np.float64)

        return {
            "joint_pos": np.concatenate(
                (arm_joint_pos, gripper_joint_pos), dtype=np.float64
            ),
            "joint_vel": np.concatenate(
                (arm_joint_vel, gripper_joint_vel), dtype=np.float64
            ),
            "wrench": wrench,
        }
