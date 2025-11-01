from abc import ABC, abstractmethod
from os import path

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict
from isaacgym import (
    gymapi,  # noqa: F401
    gymtorch,  # noqa: F401
    gymutil,  # noqa: F401
)

from robo_manip_baselines.common import ArmConfig, DataKey, EnvDataMixin
from robo_manip_baselines.teleop import (
    GelloInputDevice,
    KeyboardInputDevice,
    SpacemouseInputDevice,
)


class IsaacUR5eEnvBase(EnvDataMixin, gym.Env, ABC):
    metadata = {
        "render_modes": [
            "human",
        ],
    }

    def __init__(
        self,
        init_qpos,
        num_envs=1,
        **kwargs,
    ):
        self.init_qpos = init_qpos
        self.render_mode = kwargs.get("render_mode")

        # Setup Isaac Gym
        self.arm_joint_idxes = np.arange(6)
        self.gripper_joint_idxes = np.array([6])
        self.setup_sim(num_envs)

        # Setup robot
        self.body_config_list = [
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__), "../assets/common/robots/ur5e/ur5e.urdf"
                ),
                arm_root_pose=self.get_link_pose("ur5e", "base_link"),
                ik_eef_joint_id=6,
                arm_joint_idxes=self.arm_joint_idxes,
                gripper_joint_idxes=self.gripper_joint_idxes,
                gripper_joint_idxes_in_gripper_joint_pos=np.array([0]),
                eef_idx=0,
                init_arm_joint_pos=self.init_qpos[0:6],
                init_gripper_joint_pos=np.zeros(1),
            )
        ]

        # Setup environment parameters
        self.skip_sim = 2
        self.dt = self.skip_sim * self.gym.get_sim_params(self.sim).dt
        self.world_random_scale = None

        robot_dof_props = self.gym.get_actor_dof_properties(
            self.env_list[self.rep_env_idx], self.robot_handle_list[self.rep_env_idx]
        )
        self.action_space = Box(
            low=np.concatenate(
                (robot_dof_props["lower"][0:6], np.array([0.0], dtype=np.float32))
            ),
            high=np.concatenate(
                (robot_dof_props["upper"][0:6], np.array([255.0], dtype=np.float32))
            ),
            dtype=np.float32,
        )
        self.observation_space = Dict(
            {
                "joint_pos": Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
                ),
                "joint_vel": Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
                ),
                "wrench": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
            }
        )

        self.action_list = None
        self.obs_list = None
        self.info_list = None

        self.action_fluctuation_scale = np.array(
            [np.deg2rad(0.1)] * 6 + [0.0], dtype=np.float32
        )
        self.action_fluctuation_list = [
            np.zeros(self.action_space.shape, dtype=np.float32)
            for env_idx in range(self.num_envs)
        ]

        # Setup internal variables
        self.quit_flag = False
        self.pause_flag = False

    def setup_sim(self, num_envs):
        # For visualization reasons, the last of the Isaac Gym parallel environments is considered representative
        self.rep_env_idx = num_envs - 1

        # Setup sim
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, self.get_sim_params())

        # Setup robot asset
        robot_asset_root = path.join(
            path.dirname(__file__), "../assets/isaac/robots/ur5e"
        )
        robot_asset_file = "ur5e_integrated.urdf"
        robot_asset_options = gymapi.AssetOptions()
        robot_asset_options.armature = 0.01
        robot_asset_options.fix_base_link = True
        robot_asset_options.flip_visual_attachments = True
        self.robot_asset = self.gym.load_asset(
            self.sim, robot_asset_root, robot_asset_file, robot_asset_options
        )

        # Setup force sensor
        force_sensor_body_name = "wrist_3_link"
        force_sensor_body_idx = self.gym.find_asset_rigid_body_index(
            self.robot_asset, force_sensor_body_name
        )
        force_sensor_pose_local = gymapi.Transform(
            r=gymapi.Quat.from_euler_zyx(0, np.pi, 0)
        )
        force_sensor_idx = self.gym.create_asset_force_sensor(
            self.robot_asset, force_sensor_body_idx, force_sensor_pose_local
        )

        # Setup ground
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        # Setup task-specific assets
        self.setup_task_specific_assets()

        # Setup variables
        self.env_list = []
        self.robot_handle_list = []
        self.force_sensor_list = []
        self.camera_handles_list = []
        self.camera_properties_list = []
        self.setup_task_specific_variables()

        # Setup env
        spacing = 0.5
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        for env_idx in range(num_envs):
            env = self.gym.create_env(
                self.sim, env_lower, env_upper, int(num_envs**0.5)
            )
            self.env_list.append(env)

            # Setup viewer
            if (env_idx == self.rep_env_idx) and (self.render_mode == "human"):
                self.viewer = self.gym.create_viewer(
                    self.sim, gymapi.CameraProperties()
                )
                camera_origin_pos = gymapi.Vec3(1.0, 0.5, 1.0)
                camera_lookat_pos = gymapi.Vec3(0.3, 0.0, 0.3)
                self.gym.viewer_camera_look_at(
                    self.viewer, env, camera_origin_pos, camera_lookat_pos
                )
                self.gym.subscribe_viewer_keyboard_event(
                    self.viewer, gymapi.KEY_ESCAPE, "quit"
                )
                self.gym.subscribe_viewer_keyboard_event(
                    self.viewer, gymapi.KEY_Q, "quit"
                )
                self.gym.subscribe_viewer_keyboard_event(
                    self.viewer, gymapi.KEY_SPACE, "pause_resume"
                )

            # Setup robot actor
            robot_pose = gymapi.Transform(p=gymapi.Vec3(0, 0, 0))
            robot_handle = self.gym.create_actor(
                env, self.robot_asset, robot_pose, "ur5e", env_idx, 0
            )
            self.robot_handle_list.append(robot_handle)
            force_sensor = self.gym.get_actor_force_sensor(
                env, robot_handle, force_sensor_idx
            )
            self.force_sensor_list.append(force_sensor)

            # Setup gripper mimic joints
            if env_idx == 0:
                gripper_mimic_multiplier_map = {
                    "robotiq_85_left_knuckle_joint": 1.0,
                    "robotiq_85_right_knuckle_joint": -1.0,
                    "robotiq_85_left_inner_knuckle_joint": 1.0,
                    "robotiq_85_right_inner_knuckle_joint": -1.0,
                    "robotiq_85_left_finger_tip_joint": -1.0,
                    "robotiq_85_right_finger_tip_joint": 1.0,
                }
                self.gripper_mimic_multiplier_list = np.zeros(
                    len(gripper_mimic_multiplier_map), dtype=np.float32
                )
                for (
                    joint_name,
                    mimic_multiplier,
                ) in gripper_mimic_multiplier_map.items():
                    dof_idx = self.gym.find_actor_dof_index(
                        env, robot_handle, joint_name, gymapi.DOMAIN_ACTOR
                    )
                    self.gripper_mimic_multiplier_list[dof_idx - 6] = mimic_multiplier

            # Setup robot joint control mode
            robot_dof_props = self.gym.get_asset_dof_properties(self.robot_asset)
            robot_dof_props["driveMode"][:] = gymapi.DOF_MODE_POS
            robot_dof_props["armature"] = np.array(
                [0.1] * 6 + [0.001] * 6, dtype=np.float32
            )
            robot_dof_props["stiffness"] = np.array(
                [2000, 2000, 2000, 500, 500, 500] + [400] * 6, dtype=np.int32
            )
            robot_dof_props["damping"] = np.array(
                [400, 400, 400, 100, 100, 100] + [80] * 6, dtype=np.int32
            )
            self.gym.set_actor_dof_properties(env, robot_handle, robot_dof_props)

            # Setup gripper command scale
            if env_idx == 0:
                gripper_dof_idx = self.gym.find_actor_dof_index(
                    env,
                    robot_handle,
                    "robotiq_85_left_knuckle_joint",
                    gymapi.DOMAIN_ACTOR,
                )
                original_gripper_range = (
                    robot_dof_props["upper"][gripper_dof_idx]
                    - robot_dof_props["lower"][gripper_dof_idx]
                )
                new_gripper_range = 255.0
                self.gripper_command_scale = original_gripper_range / new_gripper_range

            # Setup robot joint control command
            if env_idx == 0:
                robot_num_dofs = self.gym.get_asset_dof_count(self.robot_asset)
                self.init_robot_dof_state = np.zeros(
                    robot_num_dofs, gymapi.DofState.dtype
                )
                self.init_robot_dof_state["pos"] = (
                    self.get_robot_dof_pos_from_joint_pos(self.init_qpos)
                )
            self.gym.set_actor_dof_states(
                env, robot_handle, self.init_robot_dof_state, gymapi.STATE_ALL
            )
            self.gym.set_actor_dof_position_targets(
                env, robot_handle, self.init_robot_dof_state["pos"]
            )

            # Setup task-specific actors
            self.setup_task_specific_actors(env_idx)

            # Setup task-specific cameras
            camera_handles = {}
            camera_properties = {}
            self.camera_handles_list.append(camera_handles)
            self.camera_properties_list.append(camera_properties)
            self.setup_task_specific_cameras(env_idx)

            # Setup common cameras
            single_camera_properties = gymapi.CameraProperties()
            single_camera_properties.width = 640
            single_camera_properties.height = 480
            camera_handle = self.gym.create_camera_sensor(env, single_camera_properties)
            body_handle = self.gym.find_actor_rigid_body_handle(
                env, robot_handle, "camera_link"
            )
            camera_pose_local = gymapi.Transform(p=gymapi.Vec3(0, -0.02, 0.005))
            self.gym.attach_camera_to_body(
                camera_handle,
                env,
                body_handle,
                camera_pose_local,
                gymapi.FOLLOW_TRANSFORM,
            )
            camera_handles["hand"] = camera_handle
            camera_properties["hand"] = single_camera_properties

        # Store state
        self.init_state = np.copy(
            self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_ALL)
        )
        self.original_init_state = np.copy(self.init_state)

    def get_sim_params(self):
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.80665)  # [m/s^2]
        sim_params.dt = 1.0 / 60.0  # [s]
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = False
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.rest_offset = 0.001
        sim_params.physx.contact_offset = 0.02
        sim_params.physx.friction_offset_threshold = 0.04
        sim_params.physx.friction_correlation_distance = 0.025
        sim_params.physx.num_threads = 0
        sim_params.physx.use_gpu = True  # False
        return sim_params

    @abstractmethod
    def setup_task_specific_variables(self):
        pass

    @abstractmethod
    def setup_task_specific_assets(self):
        pass

    @abstractmethod
    def setup_task_specific_actors(self, env_idx):
        pass

    @abstractmethod
    def setup_task_specific_cameras(self, env_idx):
        pass

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.gym.set_sim_rigid_body_states(self.sim, self.init_state, gymapi.STATE_ALL)
        for env_idx, (env, robot_handle) in enumerate(
            zip(self.env_list, self.robot_handle_list)
        ):
            self.gym.set_actor_dof_states(
                env, robot_handle, self.init_robot_dof_state, gymapi.STATE_ALL
            )
            self.gym.set_actor_dof_position_targets(
                env, robot_handle, self.init_robot_dof_state["pos"]
            )
            self.reset_task_specific_actors(env_idx)

        self.obs_list = self._get_obs_list()
        self.info_list = self._get_info_list()

        # Return only the results of the representative environment to comply with the Gym API
        return self.obs_list[self.rep_env_idx], self.info_list[self.rep_env_idx]

    def reset_task_specific_actors(self, env_idx):
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
        return {}

    def step(self, action):
        # Check key input
        if self.render_mode == "human":
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "quit" and evt.value > 0:
                    self.quit_flag = True
                elif evt.action == "pause_resume" and evt.value > 0:
                    self.pause_flag = not self.pause_flag
        if self.quit_flag:
            self.close()
            return

        # Set robot joint command
        for env_idx, (env, robot_handle) in enumerate(
            zip(self.env_list, self.robot_handle_list)
        ):
            if self.action_list is None:
                if env_idx == 0:
                    robot_dof_pos = self.get_robot_dof_pos_from_joint_pos(action)
            else:
                robot_dof_pos = self.get_robot_dof_pos_from_joint_pos(
                    self.action_list[env_idx]
                )
            self.gym.set_actor_dof_position_targets(env, robot_handle, robot_dof_pos)
        self.action_list = None

        # Update simulation
        if not self.pause_flag:
            for _ in range(self.skip_sim):
                self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # Update viewer
        if self.render_mode == "human":
            self.render()

        self.obs_list = self._get_obs_list()
        reward = 0.0
        terminated = False
        self.info_list = self._get_info_list()
        self.success_list = self._get_success_list()

        # self.gym.sync_frame_time(self.sim)

        # Return only the results of the representative environment to comply with the Gym API
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return (
            self.obs_list[self.rep_env_idx],
            reward,
            terminated,
            False,
            self.info_list[self.rep_env_idx],
        )

    def _get_obs_list(self):
        obs_list = []

        for env, robot_handle, force_sensor in zip(
            self.env_list, self.robot_handle_list, self.force_sensor_list
        ):
            robot_dof_state = self.gym.get_actor_dof_states(
                env, robot_handle, gymapi.STATE_ALL
            )

            arm_joint_pos = robot_dof_state["pos"][0:6]
            arm_joint_vel = robot_dof_state["vel"][0:6]
            gripper_joint_pos = self.get_gripper_joint_pos_from_dof_pos(
                robot_dof_state["pos"][6:12]
            )
            gripper_joint_vel = np.zeros(1)
            wrench = force_sensor.get_forces()
            force = np.array([wrench.force.x, wrench.force.y, wrench.force.z])
            torque = np.array([wrench.torque.x, wrench.torque.y, wrench.torque.z])

            obs = {
                "joint_pos": np.concatenate(
                    (arm_joint_pos, gripper_joint_pos), dtype=np.float64
                ),
                "joint_vel": np.concatenate(
                    (arm_joint_vel, gripper_joint_vel), dtype=np.float64
                ),
                "wrench": np.concatenate((force, torque), dtype=np.float64),
            }
            obs_list.append(obs)

        return obs_list

    def _get_info_list(self):
        # Update camera image
        self.gym.clear_lines(self.viewer)
        self.gym.render_all_camera_sensors(self.sim)

        # Get camera images
        info_list = []
        for env, camera_handles in zip(self.env_list, self.camera_handles_list):
            info = {"rgb_images": {}, "depth_images": {}}
            for camera_name, camera_handle in camera_handles.items():
                rgb_image = self.gym.get_camera_image(
                    self.sim, env, camera_handle, gymapi.IMAGE_COLOR
                )
                rgb_image = rgb_image.reshape(rgb_image.shape[0], -1, 4)[:, :, :3]
                depth_image = -1 * self.gym.get_camera_image(
                    self.sim, env, camera_handle, gymapi.IMAGE_DEPTH
                )
                depth_image[np.isinf(depth_image)] = 0.0
                info["rgb_images"][camera_name] = rgb_image
                info["depth_images"][camera_name] = depth_image
            info_list.append(info)

        return info_list

    def _get_success_list(self):
        # Intended to be overridden in derived classes
        return [False] * self.num_envs

    def close(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def render(self):
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)

    @property
    def num_envs(self):
        """Get the number of the Isaac Gym parallel environments."""
        return len(self.env_list)

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
        return self.gym.get_sim_time(self.sim)

    def get_link_pose(self, actor_name, link_name=None, link_idx=0):
        """Get link pose (tx, ty, tz, qw, qx, qy, qz)."""
        env = self.env_list[self.rep_env_idx]
        actor_idx = self.gym.find_actor_index(env, actor_name, gymapi.DOMAIN_ENV)
        if link_name is not None:
            link_idx = self.gym.find_actor_rigid_body_index(
                env, actor_idx, link_name, gymapi.DOMAIN_ACTOR
            )
        link_pose = gymapi.Transform.from_buffer(
            self.gym.get_actor_rigid_body_states(env, actor_idx, gymapi.STATE_POS)[
                "pose"
            ][link_idx]
        )
        return np.array(
            [
                link_pose.p.x,
                link_pose.p.y,
                link_pose.p.z,
                link_pose.r.w,
                link_pose.r.x,
                link_pose.r.y,
                link_pose.r.z,
            ]
        )

    @property
    def camera_names(self):
        """Get camera names."""
        return list(self.camera_handles_list[self.rep_env_idx].keys())

    @property
    def pointcloud_camera_names(self):
        """Get pointcloud camera names."""
        return []

    @property
    def rgb_tactile_names(self):
        """Get names of tactile sensors with RGB output."""
        return []

    @property
    def intensity_tactile_names(self):
        """Get names of tactile sensors with intensity output."""
        return list(self.intensity_tactiles.keys())

    def get_camera_fovy(self, camera_name):
        """Get vertical field-of-view of the camera."""
        single_camera_properties = self.camera_properties_list[self.rep_env_idx][
            camera_name
        ]
        camera_fovy = (
            single_camera_properties.height
            / single_camera_properties.width
            * single_camera_properties.horizontal_fov
        )
        return camera_fovy

    @abstractmethod
    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        pass

    def draw_box_marker(self, pos, mat, size, rgba):
        """Draw box marker."""
        # TODO: Implement a method
        pass

    def get_robot_dof_pos_from_joint_pos(self, robot_joint_pos):
        robot_num_dofs = self.gym.get_asset_dof_count(self.robot_asset)
        robot_dof_pos = np.zeros(robot_num_dofs, dtype=np.float32)
        robot_dof_pos[0:6] = robot_joint_pos[self.arm_joint_idxes]
        robot_dof_pos[6:12] = self.get_gripper_dof_pos_from_joint_pos(
            robot_joint_pos[self.gripper_joint_idxes]
        )
        return robot_dof_pos

    def get_gripper_dof_pos_from_joint_pos(self, gripper_joint_pos):
        return (
            gripper_joint_pos[0]
            * self.gripper_command_scale
            * self.gripper_mimic_multiplier_list
        )

    def get_gripper_joint_pos_from_dof_pos(self, gripper_dof_pos):
        return (
            gripper_dof_pos
            / (self.gripper_command_scale * self.gripper_mimic_multiplier_list)
        ).mean(keepdims=True)

    def get_fluctuated_action_list(self, action, update_fluctuation=True):
        action_list = []
        for env_idx in range(self.num_envs):
            if update_fluctuation and (env_idx != self.rep_env_idx):
                # Set action fluctuation by random walk
                self.action_fluctuation_list[env_idx] += np.random.normal(
                    scale=self.action_fluctuation_scale
                )
            action_list.append(action + self.action_fluctuation_list[env_idx])
        return action_list
