from abc import ABC, abstractmethod
from os import path

import cv2
import gymnasium as gym
import numpy as np
import pybullet as p
import pybulletX as px
from gymnasium.spaces import Box, Dict
from tacto import Sensor

from robo_manip_baselines.common import ArmConfig, DataKey, EnvDataMixin
from robo_manip_baselines.teleop import (
    GelloInputDevice,
    KeyboardInputDevice,
    SpacemouseInputDevice,
)


class Camera:
    def __init__(
        self,
        camera_pos,
        camera_distance,
        yaw,
        pitch,
        roll,
        width=640,
        height=480,
        fov=60,
        near_plane=0.01,
        far_plane=100.0,
        up_axis_index=2,
    ):
        self.width = width
        self.height = height

        camTargetPos = camera_pos
        camDistance = camera_distance
        upAxisIndex = up_axis_index

        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll

        self.fov = fov
        self.nearPlane = near_plane
        self.farPlane = far_plane

        self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(
            camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex
        )

        aspect = width / height

        self.projectionMatrix = p.computeProjectionMatrixFOV(
            fov, aspect, self.nearPlane, self.farPlane
        )

    def get_image(self):
        img_arr = p.getCameraImage(
            self.width,
            self.height,
            self.viewMatrix,
            self.projectionMatrix,
            shadow=1,
            lightDirection=[1, 1, 1],
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        img_np = np.array(img_arr[2], dtype=np.uint8).reshape(
            (self.height, self.width, 4)
        )
        rgb = img_np[:, :, :3]  # color image RGB H x W x 3 (uint8)
        dep = np.array(img_arr[3], dtype=np.float32).reshape(
            (self.height, self.width)
        )  # depth image H x W (float32)

        near = self.nearPlane
        far = self.farPlane
        dep = (2.0 * near * far) / (far + near - (2.0 * dep - 1.0) * (far - near))

        return rgb, dep


class TactoSawyerEnvBase(EnvDataMixin, gym.Env, ABC):
    metadata = {
        "render_modes": [
            "human",
        ],
    }

    def __init__(
        self,
        init_qpos,
        **kwargs,
    ):
        self.init_qpos = init_qpos
        self.render_mode = kwargs.get("render_mode")

        # Setup tacto
        px.init(mode=p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.2,
            cameraYaw=90,
            cameraPitch=-30,
            cameraTargetPosition=[0.2, 0.0, 0.1],
        )

        self.setup_robot(init_qpos)
        self.setup_rgb_tactile_sensor()
        self.setup_camera()
        self.setup_task_specific_object()

        self.world_random_scale = None
        self.reset()

        # Setup environment parameters
        self.dt = 0.03  # [s]
        self.sim_dt = 0.005  # [s]

        p.setPhysicsEngineParameter(
            numSolverIterations=200, solverResidualThreshold=1e-8, enableConeFriction=1
        )
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.sim_dt)

        self.action_space = Box(
            low=np.concatenate(
                (
                    self.robot.action_space["joint_position"].low[:7],
                    self.robot.action_space["joint_position"].low[-1:],
                )
            ),
            high=np.concatenate(
                (
                    self.robot.action_space["joint_position"].high[:7],
                    self.robot.action_space["joint_position"].high[-1:],
                )
            ),
            shape=(8,),
            dtype=np.float64,
        )
        self.observation_space = Dict(
            {
                "joint_pos": Box(
                    low=np.concatenate(
                        (
                            self.robot.state_space["joint_position"].low[:7],
                            self.robot.state_space["joint_position"].low[-1:],
                        )
                    ),
                    high=np.concatenate(
                        (
                            self.robot.state_space["joint_position"].high[:7],
                            self.robot.state_space["joint_position"].high[-1:],
                        )
                    ),
                    shape=(8,),
                    dtype=np.float64,
                ),
                "joint_vel": Box(
                    low=np.concatenate(
                        (
                            self.robot.state_space["joint_velocity"].low[:7],
                            self.robot.state_space["joint_velocity"].low[-1:],
                        )
                    ),
                    high=np.concatenate(
                        (
                            self.robot.state_space["joint_velocity"].high[:7],
                            self.robot.state_space["joint_velocity"].high[-1:],
                        )
                    ),
                    shape=(8,),
                    dtype=np.float64,
                ),
                "wrench": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
            }
        )

    def setup_robot(self, init_qpos):
        self.robot = px.Robot(
            urdf_path=path.join(
                path.dirname(__file__),
                "../assets/tacto/robots/sawyer/sawyer_wsg50.urdf",
            ),
            base_position=[0.0, 0, 0.1],
            use_fixed_base=True,
        )
        self.robot.zero_pose = init_qpos
        self.body_config_list = [
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__), "../assets/common/robots/sawyer/sawyer.urdf"
                ),
                arm_root_pose=self.get_link_pose("right_j0"),
                ik_eef_joint_id=7,
                arm_joint_idxes=np.arange(7),
                gripper_joint_idxes=np.array([7]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([0]),
                eef_idx=0,
                init_arm_joint_pos=self.init_qpos[0:7],
                init_gripper_joint_pos=np.array([0.02]),
            )
        ]
        p.enableJointForceTorqueSensor(
            self.robot.id, self.robot.get_joint_index_by_name("right_hand"), True
        )
        self.robot.configure_state_space(applied_joint_motor_torque=False)
        self.tactile_joint_names = ["joint_finger_tip_left", "joint_finger_tip_right"]
        for name in self.tactile_joint_names:
            p.changeDynamics(
                self.robot.id,
                self.robot.get_joint_index_by_name(name),
                lateralFriction=1.2,
                spinningFriction=0.03,
                rollingFriction=0.03,
                contactStiffness=300,
                contactDamping=30,
            )

    def setup_rgb_tactile_sensor(self):
        self.rgb_tactiles = Sensor(width=480, height=640, visualize_gui=False)
        tactile_link_ids = [
            self.robot.get_joint_index_by_name(name)
            for name in self.tactile_joint_names
        ]
        self.rgb_tactiles.add_camera(self.robot.id, tactile_link_ids)

    def setup_camera(self):
        self.cameras = {}
        self.cameras["front"] = Camera(
            camera_pos=[0.5, 0, 0.05],
            camera_distance=0.4,
            yaw=90,
            pitch=-30.0,
            roll=0,
        )

    @abstractmethod
    def setup_task_specific_object(self):
        pass

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.sim_time = 0.0

        self.robot.reset()

        self.reset_task_specific_object()

        self.obs = self._get_obs()
        self.info = self._get_info()

        return self.obs, self.info

    @abstractmethod
    def reset_task_specific_object(self):
        pass

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
        return {"gripper_scale": 0.002}

    def step(self, action):
        for _ in range(int(self.dt / self.sim_dt)):
            self._set_actions(action)
            p.stepSimulation()
        self.sim_time += self.dt

        self.obs = self._get_obs()
        reward = self._get_reward()
        terminated = False
        self.info = self._get_info()

        # Update viewer
        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return (self.obs, reward, terminated, False, self.info)

    def _set_actions(self, action):
        gripper_joint_pos = action[-1]
        tacto_action = np.concatenate(
            (action[:7], np.array([-1 * gripper_joint_pos, gripper_joint_pos]))
        )
        max_forces = np.full(len(tacto_action), 200.0)
        max_forces[-2:] = 20.0
        self.robot.set_joint_position(
            tacto_action, max_forces, use_joint_effort_limits=False
        )

    def _get_obs(self):
        arm_joint_name_list = [
            "right_j0",
            "right_j1",
            "right_j2",
            "right_j3",
            "right_j4",
            "right_j5",
            "right_j6",
        ]
        gripper_joint_name_list = [
            "base_joint_gripper_left",
            "base_joint_gripper_right",
        ]

        arm_joint_idx = [
            self.robot.get_joint_index_by_name(joint_name)
            for joint_name in arm_joint_name_list
        ]
        arm_joint_state = self.robot.get_joint_states(joint_indices=arm_joint_idx)
        arm_joint_pos = arm_joint_state["joint_position"]
        arm_joint_vel = arm_joint_state["joint_velocity"]

        gripper_joint_idx = [
            self.robot.get_joint_index_by_name(joint_name)
            for joint_name in gripper_joint_name_list
        ]
        gripper_joint_state = self.robot.get_joint_states(
            joint_indices=gripper_joint_idx
        )
        gripper_joint_pos = np.abs(gripper_joint_state["joint_position"]).mean(
            keepdims=True
        )
        gripper_joint_vel = np.zeros(1)

        eef_joint_state = self.robot.get_joint_state_by_name("right_hand")
        wrench = np.array(eef_joint_state["joint_reaction_forces"], dtype=np.float64)

        return {
            "joint_pos": np.concatenate(
                (arm_joint_pos, gripper_joint_pos),
                dtype=np.float64,
            ),
            "joint_vel": np.concatenate(
                (arm_joint_vel, gripper_joint_vel),
                dtype=np.float64,
            ),
            "wrench": wrench,
        }

    def _get_info(self):
        info = {}

        if len(self.camera_names) + len(self.rgb_tactile_names) == 0:
            return info

        info["rgb_images"] = {}
        info["depth_images"] = {}

        for camera_name, camera in self.cameras.items():
            rgb_image, depth_image = camera.get_image()
            info["rgb_images"][camera_name] = rgb_image
            info["depth_images"][camera_name] = depth_image

        rgb_tactiles, depth_tactiles = self.rgb_tactiles.render()
        for tactile_name, rgb_tactile, depth_tactile in zip(
            self.rgb_tactile_names, rgb_tactiles, depth_tactiles
        ):
            info["rgb_images"][tactile_name] = cv2.rotate(
                rgb_tactile, cv2.ROTATE_90_CLOCKWISE
            )
            info["depth_images"][tactile_name] = cv2.rotate(
                depth_tactile, cv2.ROTATE_90_CLOCKWISE
            )

        return info

    def _get_reward(self):
        return 0.0

    def close(self):
        pass

    def get_joint_pos_from_obs(self, obs):
        """Get joint position from observation."""
        return obs["joint_pos"]

    def get_joint_vel_from_obs(self, obs):
        """Get joint velocity from observation."""
        return obs["joint_vel"]

    def get_gripper_joint_pos_from_obs(self, obs):
        """Get gripper joint position from observation."""
        joint_pos = self.get_joint_pos_from_obs(obs)
        gripper_joint_pos = np.zeros(
            DataKey.get_dim(DataKey.COMMAND_GRIPPER_JOINT_POS, self)
        )

        for body_config in self.body_config_list:
            if not isinstance(body_config, ArmConfig):
                continue

            gripper_joint_pos[body_config.gripper_joint_idxes_in_gripper_joint_pos] = (
                joint_pos[body_config.gripper_joint_idxes]
            )

        return gripper_joint_pos

    def get_eef_wrench_from_obs(self, obs):
        """Get end-effector wrench (fx, fy, fz, nx, ny, nz) from observation."""
        return obs["wrench"]

    def get_time(self):
        """Get simulation time. [s]"""
        return self.sim_time

    def get_link_pose(self, link_name):
        """Get link pose (tx, ty, tz, qw, qx, qy, qz)."""
        link_state = self.robot.get_link_state_by_name(link_name)
        return np.concatenate(
            (link_state.link_world_position, link_state.link_world_orientation)
        )

    @property
    def camera_names(self):
        """Get camera names."""
        return list(self.cameras.keys())

    @property
    def pointcloud_camera_names(self):
        """Get pointcloud camera names."""
        return []

    @property
    def rgb_tactile_names(self):
        """Get names of tactile sensors with RGB output."""
        return ["tactile_left", "tactile_right"]

    @property
    def intensity_tactile_names(self):
        """Get names of tactile sensors with intensity output."""
        return list(self.intensity_tactiles.keys())

    def get_camera_fovy(self, camera_name):
        """Get vertical field-of-view of the camera."""
        return self.cameras[camera_name].fov

    @abstractmethod
    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        pass

    def draw_box_marker(self, pos, mat, size, rgba):
        """Draw box marker."""
        pass
