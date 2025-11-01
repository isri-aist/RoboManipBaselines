import concurrent.futures
import os
import re
import sys
import time
from abc import ABC, abstractmethod
from queue import Queue

import cv2
import gymnasium as gym
import numpy as np
from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids

from robo_manip_baselines.common import ArmConfig, DataKey, EnvDataMixin


class RealEnvBase(EnvDataMixin, gym.Env, ABC):
    metadata = {
        "render_modes": [],
    }

    def __init__(
        self,
        **kwargs,
    ):
        # Setup environment parameters
        self.init_time = time.time()
        self.dt = 0.02  # [s]
        self.world_random_scale = None

        # Setup device variables
        self.cameras = {}
        self.pointcloud_cameras = {}
        self.rgb_tactiles = {}
        self.intensity_tactiles = {}

    def setup_realsense(self, camera_ids):
        if camera_ids is None:
            return

        detected_camera_ids = get_device_ids()
        for camera_name, camera_id in camera_ids.items():
            if camera_id not in detected_camera_ids:
                raise RuntimeError(
                    f"[{self.__class__.__name__}] Specified camera (name: {camera_name}, ID: {camera_id}) not detected. Detected camera IDs: {detected_camera_ids}"
                )

            camera = RealSenseCamera(device_id=camera_id, flip=False)
            frames = camera._pipeline.wait_for_frames()
            color_intrinsics = (
                frames.get_color_frame().profile.as_video_stream_profile().intrinsics
            )
            camera.color_fovy = np.rad2deg(
                2 * np.arctan(color_intrinsics.height / (2 * color_intrinsics.fy))
            )
            depth_intrinsics = (
                frames.get_depth_frame().profile.as_video_stream_profile().intrinsics
            )
            camera.depth_fovy = np.rad2deg(
                2 * np.arctan(depth_intrinsics.height / (2 * depth_intrinsics.fy))
            )

            self.cameras[camera_name] = camera

    def setup_femtobolt(self, pointcloud_camera_ids):
        if pointcloud_camera_ids is None:
            return

        sys.path.append(
            os.path.join(os.path.dirname(__file__), "../../../third_party/pyorbbecsdk")
        )
        from pyorbbecsdk import (
            AlignFilter,
            Config,
            Context,
            OBSensorType,
            OBStreamType,
            Pipeline,
            PointCloudFilter,
        )

        ctx = Context()
        device_list = ctx.query_devices()
        curr_device_cnt = device_list.get_count()
        for (
            pointcloud_camera_name,
            pointcloud_camera_id,
        ) in pointcloud_camera_ids.items():
            if pointcloud_camera_id > curr_device_cnt:
                raise RuntimeError(
                    f"[{self.__class__.__name__}] Specified camera (name: {pointcloud_camera_name}, ID: {pointcloud_camera_id}) not detected. Max camera ID: {curr_device_cnt}"
                )
            pointcloud_camera = {}
            queue = Queue()
            device = device_list.get_device_by_index(pointcloud_camera_id)
            pipeline = Pipeline(device)
            config = Config()
            depth_profile_list = pipeline.get_stream_profile_list(
                OBSensorType.DEPTH_SENSOR
            )
            if depth_profile_list is None:
                raise RuntimeError(
                    f"[{self.__class__.__name__}] No proper depth profile, can not generate point cloud."
                )
            depth_profile = depth_profile_list.get_default_video_stream_profile()
            config.enable_stream(depth_profile)
            has_color_sensor = False
            color_profile_list = pipeline.get_stream_profile_list(
                OBSensorType.COLOR_SENSOR
            )
            if color_profile_list is not None:
                has_color_sensor = True
                color_profile = color_profile_list.get_default_video_stream_profile()
                config.enable_stream(color_profile)
            pipeline.enable_frame_sync()
            pipeline.start(
                config,
                lambda frame_set,
                pointcloud_camera_name=pointcloud_camera_name: self.femtobolt_callback(
                    frame_set, pointcloud_camera_name
                ),
            )
            camera_param = pipeline.get_camera_param()
            align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
            point_cloud_filter = PointCloudFilter()
            point_cloud_filter.set_camera_param(camera_param)

            pointcloud_camera["queue"] = queue
            pointcloud_camera["has_color_sensor"] = has_color_sensor
            pointcloud_camera["pipeline"] = pipeline
            pointcloud_camera["align_filter"] = align_filter
            pointcloud_camera["point_cloud_filter"] = point_cloud_filter

            self.pointcloud_cameras[pointcloud_camera_name] = pointcloud_camera

    def femtobolt_callback(self, frames, pointcloud_camera_name):
        if frames is None:
            return

        pointcloud_camera = self.pointcloud_cameras[pointcloud_camera_name]
        queue = pointcloud_camera["queue"]
        if queue.qsize() >= 5:
            queue.get()
        queue.put(frames)

    def setup_gelsight(self, gelsight_ids):
        if gelsight_ids is None:
            return

        for rgb_tactile_name, gelsight_id in gelsight_ids.items():
            for device_name in os.listdir("/sys/class/video4linux"):
                real_device_name = os.path.realpath(
                    "/sys/class/video4linux/" + device_name + "/name"
                )
                with (
                    open(real_device_name, "rt") as device_name_file
                ):  # "rt": read-text mode ("t" is default, so "r" alone is the same)
                    detected_gelsight_id = device_name_file.read().rstrip()
                if gelsight_id in detected_gelsight_id:
                    tactile_num = int(re.search("\d+$", device_name).group(0))
                    print(
                        f"[{self.__class__.__name__}] Found GelSight sensor. ID: {detected_gelsight_id}, device: {device_name}, num: {tactile_num}"
                    )

                    rgb_tactile = cv2.VideoCapture(tactile_num)
                    if rgb_tactile is None or not rgb_tactile.isOpened():
                        print(
                            f"[{self.__class__.__name__}] Unable to open video source of GelSight sensor."
                        )
                        continue

                    self.rgb_tactiles[rgb_tactile_name] = rgb_tactile
                    break

            if rgb_tactile_name not in self.rgb_tactiles:
                raise RuntimeError(
                    f"[{self.__class__.__name__}] Specified GelSight (name: {rgb_tactile_name}, ID: {gelsight_id}) not detected."
                )

    def setup_sanwa_keyboard(self, sanwa_keyboard_ids):
        if sanwa_keyboard_ids is None:
            return

        import hid

        for intensity_tactile_name, device_path in sanwa_keyboard_ids.items():
            if not os.path.exists(device_path):
                raise RuntimeError(
                    f"[{self.__class__.__name__}] Specified keyboard (path: {device_path}) not detected."
                )

            intensity_tactile_device = hid.Device(path=device_path.encode())
            if intensity_tactile_device is None:
                print(
                    f"[{self.__class__.__name__}] Unable to open keyboard (path: {device_path})."
                )
                continue
            intensity_tactile_buf = np.zeros(shape=(2, 3), dtype=np.uint8)
            intensity_tactile = {
                "device": intensity_tactile_device,
                "buf": intensity_tactile_buf,
            }

            self.intensity_tactiles[intensity_tactile_name] = intensity_tactile

    def get_input_device_kwargs(self, input_device_name):
        return {}

    def reset(self, *, seed=None, options=None):
        self.init_time = time.time()

        super().reset(seed=seed)

        self._reset_robot()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        self._set_action(action, duration=self.dt, joint_vel_limit_scale=10.0, wait=True)

        observation = self._get_obs()
        reward = 0.0
        terminated = False
        info = self._get_info()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def close(self):
        pass

    @abstractmethod
    def _reset_robot(self):
        pass

    @abstractmethod
    def _set_action(self):
        pass

    def overwrite_command_for_safety(self, action, duration, joint_vel_limit_scale):
        arm_joint_idxes = np.concatenate(
            [
                body_config.arm_joint_idxes
                for body_config in self.body_config_list
                if isinstance(body_config, ArmConfig)
            ]
        )
        arm_joint_pos_command = action[arm_joint_idxes]
        scaled_joint_vel_limit = (
            np.clip(joint_vel_limit_scale, 0.01, 10.0) * self.joint_vel_limit
        )

        if duration is None:
            duration_min, duration_max = 0.1, 10.0  # [s]
            duration = np.clip(
                np.max(
                    np.abs(arm_joint_pos_command - self.arm_joint_pos_actual)
                    / scaled_joint_vel_limit
                ),
                duration_min,
                duration_max,
            )
        else:
            arm_joint_pos_error_max = np.max(
                np.abs(arm_joint_pos_command - self.arm_joint_pos_actual)
            )
            arm_joint_pos_error_thre = np.deg2rad(90)
            duration_thre = 0.1  # [s]
            if (
                arm_joint_pos_error_max > arm_joint_pos_error_thre
                and duration < duration_thre
            ):
                raise RuntimeError(
                    f"[{self.__class__.__name__}] Large joint movements are commanded in short duration ({duration} s).\n  command: {arm_joint_pos_command}\n  actual: {self.arm_joint_pos_actual}"
                )

            arm_joint_pos_command_overwritten = self.arm_joint_pos_actual + np.clip(
                arm_joint_pos_command - self.arm_joint_pos_actual,
                -1 * scaled_joint_vel_limit * duration,
                scaled_joint_vel_limit * duration,
            )
            # if np.linalg.norm(arm_joint_pos_command_overwritten - arm_joint_pos_command) > 1e-10:
            #     print(f"[{self.__class__.__name__}] Overwrite joint command for safety.")
            action[arm_joint_idxes] = arm_joint_pos_command_overwritten

        return action, duration

    @abstractmethod
    def _get_obs(self):
        pass

    def _get_info(self):
        info = {}

        if (
            len(self.camera_names)
            + len(self.pointcloud_camera_names)
            + len(self.rgb_tactile_names)
            + len(self.intensity_tactile_names)
            == 0
        ):
            return info

        info["rgb_images"] = {}
        info["depth_images"] = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}

            for camera_name, camera in self.cameras.items():
                futures[executor.submit(self.get_camera_data, camera_name, camera)] = (
                    camera_name
                )

            for (
                pointcloud_camera_name,
                pointcloud_camera,
            ) in self.pointcloud_cameras.items():
                futures[
                    executor.submit(
                        self.get_pointcloud_camera_data,
                        pointcloud_camera_name,
                        pointcloud_camera,
                    )
                ] = pointcloud_camera_name

            for rgb_tactile_name, rgb_tactile in self.rgb_tactiles.items():
                futures[
                    executor.submit(
                        self.get_rgb_tactile_data, rgb_tactile_name, rgb_tactile
                    )
                ] = rgb_tactile_name

            for (
                intensity_tactile_name,
                intensity_tactile,
            ) in self.intensity_tactiles.items():
                futures[
                    executor.submit(
                        self.get_intensity_tactile_data,
                        intensity_tactile_name,
                        intensity_tactile,
                    )
                ] = intensity_tactile_name

            for future in concurrent.futures.as_completed(futures):
                name, result = future.result()
                for key, value in result.items():
                    if value is None:
                        continue
                    if key not in info:
                        info[key] = {}
                    info[key][name] = value

        return info

    def get_camera_data(self, camera_name, camera):
        rgb_image, depth_image = camera.read((640, 480))
        depth_image = (1e-3 * depth_image[:, :, 0]).astype(np.float32)  # [m]
        return camera_name, {"rgb_images": rgb_image, "depth_images": depth_image}

    def get_pointcloud_camera_data(self, pointcloud_camera_name, pointcloud_camera):
        from pyorbbecsdk import OBFormat

        frames = pointcloud_camera["queue"].get()

        rgb_frame = frames.get_color_frame()
        if rgb_frame is None:
            return pointcloud_camera_name, {}
        rgb_width = rgb_frame.get_width()
        rgb_height = rgb_frame.get_height()
        rgb_format = rgb_frame.get_format()
        rgb_data = np.asanyarray(rgb_frame.get_data())
        if rgb_format == OBFormat.RGB:
            rgb_image = np.resize(rgb_data, (rgb_height, rgb_width, 3))
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        elif rgb_format == OBFormat.BGR:
            rgb_image = np.resize(rgb_data, (rgb_height, rgb_width, 3))
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        elif rgb_format == OBFormat.YUYV:
            rgb_image = np.resize(rgb_data, (rgb_height, rgb_width, 2))
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_YUV2BGR_YUYV)
        elif rgb_format == OBFormat.MJPG:
            rgb_image = cv2.imdecode(rgb_data, cv2.IMREAD_COLOR)
        else:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Unsupported rgb format in pointcloud camera: {rgb_format}"
            )
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (640, 480))

        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            return pointcloud_camera_name, {"rgb_images": rgb_image}
        depth_scale = depth_frame.get_depth_scale()
        depth_width = depth_frame.get_width()
        depth_height = depth_frame.get_height()
        depth_image = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_image = depth_image.astype(np.float32) * depth_scale
        depth_image = depth_image.reshape((depth_height, depth_width))
        depth_image = cv2.resize(depth_image, (640, 480))

        frame = pointcloud_camera["align_filter"].process(frames)
        pointcloud_camera["point_cloud_filter"].set_position_data_scaled(depth_scale)
        if pointcloud_camera["has_color_sensor"] and rgb_frame is not None:
            pointcloud_format = OBFormat.RGB_POINT
        else:
            pointcloud_format = OBFormat.POINT
        pointcloud_camera["point_cloud_filter"].set_create_point_format(
            pointcloud_format
        )
        point_cloud_frame = pointcloud_camera["point_cloud_filter"].process(frame)
        pointcloud = np.array(
            pointcloud_camera["point_cloud_filter"].calculate(point_cloud_frame)
        )
        pointcloud = pointcloud[::100]
        pointcloud[:, 3:6] = pointcloud[:, 3:6] / 255.0
        pointcloud[:, :3] = pointcloud[:, :3] / 1e3

        return pointcloud_camera_name, {
            "rgb_images": rgb_image,
            "depth_images": depth_image,
            "pointclouds": pointcloud,
        }

    def get_rgb_tactile_data(self, rgb_tactile_name, rgb_tactile):
        ret, rgb_image = rgb_tactile.read()
        if not ret:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Failed to read tactile image."
            )
        image_size = (640, 480)
        rgb_image = cv2.resize(rgb_image, image_size)
        return rgb_tactile_name, {"rgb_images": rgb_image}

    def get_intensity_tactile_data(self, intensity_tactile_name, intensity_tactile):
        intensity_tactile_device = intensity_tactile["device"]
        intensity_tactile_buf = intensity_tactile["buf"]
        key_binaries = intensity_tactile_device.read(9, timeout=3)

        if len(key_binaries) == 0:
            return intensity_tactile_name, {
                "intensity_tactile": intensity_tactile_buf.copy()
            }

        key_name_map = {
            0x69: "F17",
            0x6A: "F18",
            0x6B: "F19",
            0x6C: "F14",
            0x6D: "F15",
            0x6E: "F16",
        }
        key_idx_map = {
            "F14": 0,
            "F15": 1,
            "F16": 2,
            "F17": 3,
            "F18": 4,
            "F19": 5,
        }

        intensity_tactile_value = np.zeros(shape=(2, 3), dtype=np.uint8)
        for key_binary in key_binaries[3:]:
            if key_binary in key_name_map:
                key_name = key_name_map[key_binary]
                key_idx = key_idx_map[key_name]
                intensity_tactile_value[int(key_idx // 3)][int(key_idx % 3)] = 1

        intensity_tactile_buf[...] = intensity_tactile_value
        return intensity_tactile_name, {"intensity_tactile": intensity_tactile_value}

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
        """Get real-world time. [s]"""
        return time.time() - self.init_time

    @property
    def camera_names(self):
        """Get camera names."""
        return list(self.cameras.keys())

    @property
    def pointcloud_camera_names(self):
        """Get pointcloud camera names."""
        return list(self.pointcloud_cameras.keys())

    @property
    def rgb_tactile_names(self):
        """Get names of tactile sensors with RGB output."""
        return list(self.rgb_tactiles.keys())

    @property
    def intensity_tactile_names(self):
        """Get names of tactile sensors with intensity output."""
        return list(self.intensity_tactiles.keys())

    def get_camera_fovy(self, camera_name):
        """Get vertical field-of-view of the camera."""
        return self.cameras[camera_name].depth_fovy

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        raise NotImplementedError(
            f"[{self.__class__.__name__}] modify_world is not implemented."
        )

    def draw_box_marker(self, pos, mat, size, rgba):
        """Draw box marker."""
        # In a real-world environment, it is not possible to programmatically draw markers
        pass
