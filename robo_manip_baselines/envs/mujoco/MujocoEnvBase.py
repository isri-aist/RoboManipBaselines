from abc import ABC, abstractmethod

import mujoco
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer

from robo_manip_baselines.common import ArmConfig, DataKey, EnvDataMixin


class MujocoEnvBase(EnvDataMixin, MujocoEnv, ABC):
    sim_timestep = 0.004
    frame_skip = 8
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": int(1 / (sim_timestep * frame_skip)),
    }

    def __init__(
        self,
        xml_file,
        init_qpos,
        **kwargs,
    ):
        MujocoEnv.__init__(
            self,
            model_path=xml_file,
            frame_skip=self.frame_skip,
            observation_space=self.observation_space,
            width=640,
            height=480,
            default_camera_config=self.default_camera_config,
            **kwargs,
        )
        self.mujoco_renderer.width = None
        self.mujoco_renderer.height = None

        self.world_random_scale = None

        self.setup_robot(init_qpos)
        self.setup_camera()
        self.setup_intensity_tactile()

    @abstractmethod
    def setup_robot(self, init_qpos):
        pass

    def setup_camera(self):
        self.cameras = {}
        for camera_id in range(self.model.ncam):
            camera = {}
            camera_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id
            )
            camera["name"] = camera_name
            camera["id"] = camera_id
            camera["viewer"] = OffScreenViewer(
                self.model, self.data, width=640, height=480
            )
            # Because "/" are not allowed in HDF5 keys, replace "/" with "_" in dictionary keys
            self.cameras[camera_name.replace("/", "_")] = camera

        # This is required to automatically switch context to free camera in render()
        # https://github.com/Farama-Foundation/Gymnasium/blob/81b87efb9f011e975f3b646bab6b7871c522e15e/gymnasium/envs/mujoco/mujoco_rendering.py#L695-L697
        self.mujoco_renderer._viewers["dummy"] = None

        self._first_render = True

    def setup_intensity_tactile(self):
        self.intensity_tactile_names = []
        for sensor_id in range(self.model.nsensor):
            sensor = self.model.sensor(sensor_id)
            if ("tactile" in sensor.name) and (
                sensor.type == mujoco.mjtSensor.mjSENS_PLUGIN
            ):
                self.intensity_tactile_names.append(sensor.name)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = False
        info = self._get_info()

        if self.render_mode == "human":
            if self._first_render:
                self._first_render = False
                self.mujoco_renderer.viewer._hide_menu = True
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return obs, reward, terminated, False, info

    @abstractmethod
    def _get_obs(self):
        pass

    def _get_info(self):
        info = {}

        if len(self.camera_names) == 0:
            return info

        # Set camera images
        info["rgb_images"] = {}
        info["depth_images"] = {}
        for camera_name, camera in self.cameras.items():
            camera["viewer"].make_context_current()
            rgb_image = camera["viewer"].render(
                render_mode="rgb_array", camera_id=camera["id"]
            )
            info["rgb_images"][camera_name] = rgb_image
            depth_image = camera["viewer"].render(
                render_mode="depth_array", camera_id=camera["id"]
            )
            # See https://github.com/google-deepmind/mujoco/blob/631b16e7ad192df936195658fe79f2ada85f755c/python/mujoco/renderer.py#L170-L178
            extent = self.model.stat.extent
            near = self.model.vis.map.znear * extent
            far = self.model.vis.map.zfar * extent
            depth_image = near / (1 - depth_image * (1 - near / far))
            info["depth_images"][camera_name] = depth_image

        # Set tactile intensity measurements
        if len(self.intensity_tactile_names) > 0:
            info["intensity_tactile"] = {}
        for intensity_tactile_name in self.intensity_tactile_names:
            # Get tactile data
            tactile_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, intensity_tactile_name
            )
            tactile_adr = self.model.sensor_adr[tactile_id]
            tactile_dim = int(self.model.sensor_dim[tactile_id] / 7)
            tactile_data = self.data.sensordata[tactile_adr : tactile_adr + tactile_dim]

            # Get tactile shape
            plugin_id = self.model.sensor_plugin[tactile_id]
            if plugin_id < 0:
                raise ValueError(
                    f"[{self.__class__.__name__}] Tactile sensor {intensity_tactile_name} is not plugin-based."
                )
            plugin_attr = self.model.plugin_attr[
                self.model.plugin_attradr[plugin_id] :
            ].tobytes()
            tactile_shape = tuple(map(int, plugin_attr.split(b"\x00", 1)[0].split()))

            info["intensity_tactile"][intensity_tactile_name] = tactile_data.reshape(
                tactile_shape
            )

        return info

    def _get_reset_info(self):
        return self._get_info()

    def _get_reward(self):
        return 0.0

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def close(self):
        for camera in self.cameras.values():
            camera["viewer"].close()
        MujocoEnv.close(self)

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
        return self.data.time

    def get_body_pose(self, body_name):
        """Get body pose (tx, ty, tz, qw, qx, qy, qz)."""
        body = self.data.body(body_name)
        return np.concatenate((body.xpos, body.xquat))

    def get_geom_pose(self, geom_name):
        """Get geom pose (tx, ty, tz, qw, qx, qy, qz)."""
        geom = self.data.geom(geom_name)
        xquat = np.zeros(4)
        mujoco.mju_mat2Quat(xquat, geom.xmat.flatten())
        return np.concatenate((geom.xpos, xquat))

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
        return []

    def get_camera_fovy(self, camera_name):
        """Get vertical field-of-view of the camera."""
        return self.model.cam(self.cameras[camera_name]["name"]).fovy[0]

    @abstractmethod
    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        pass

    def draw_box_marker(self, pos, mat, size, rgba):
        """Draw box marker."""
        self.mujoco_renderer.viewer.add_marker(
            pos=pos,
            mat=mat,
            label="",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            rgba=rgba,
        )
