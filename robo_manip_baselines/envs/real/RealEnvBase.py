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
from gello.cameras.zed_camera import ZedCamera, get_zed_device_ids as get_zed_ids
from gello.cameras.oak_s2_camera import OakCamera, get_oak_device_ids

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
        self.rgb_tactiles = {}
        self.pointcloud_cameras = {}

    def setup_oakS2(self, camera_ids):
        """
        Initializes and configures OAK-D S2 cameras.

        Args:
            camera_ids (dict): A dictionary mapping a custom camera name to its MxId string.
        """
        detected_camera_ids = get_oak_device_ids()
        for camera_name, camera_id in camera_ids.items():
            # Check if the camera_id from the config corresponds to a detected OAK device.
            # This is necessary because the `camera_ids` dict might contain IDs for
            # other camera types (e.g., RealSense serials are also strings).
            if camera_id not in detected_camera_ids:
                continue

            # Initialize the camera object for the specific device ID.
            camera = OakCamera(device_id=camera_id)

            # Calculate and store the vertical Field of View (fovy) for downstream compatibility.
            # In the OakCamera class, depth is aligned to the color camera (CAM_A),
            # so their FoV and intrinsics are effectively the same for the aligned output.
            intrinsics = camera.get_intrinsics()["color"]
            height = intrinsics["resolution"][1]  # Image height
            fy = intrinsics["fy"]  # Focal length along the y-axis

            # Calculate fovy using the standard formula and convert to degrees.
            fovy_deg = np.rad2deg(2 * np.arctan(height / (2 * fy)))

            # Store fovy on the camera object.
            camera.depth_fovy = fovy_deg
            camera.color_fovy = fovy_deg

            # Add the configured camera to the environment's camera dictionary.
            self.cameras[camera_name] = camera

    def setup_zed(self, camera_ids):
        """
        Initializes and configures ZED cameras.

        Args:
            camera_ids (dict): A dictionary mapping a custom camera name to its serial number.
        """
        detected_camera_ids = get_zed_ids()
        for camera_name, camera_id in camera_ids.items():
            try:
                # ZED IDs are integers; this will fail for non-integer IDs (e.g., RealSense serials).
                cam_id_int = int(camera_id)
            except (ValueError, TypeError):
                # Not a valid ZED ID format, skip it.
                continue

            if cam_id_int not in detected_camera_ids:
                # If the ID is not for a detected ZED device, skip it.
                continue

            camera = ZedCamera(device_id=cam_id_int)

            # Calculate and store fovy for compatibility with downstream tasks
            intrinsics = camera.get_intrinsics()["left"]  # Use left camera intrinsics
            height = intrinsics["resolution"][1]
            fy = intrinsics["fy"]
            camera.depth_fovy = np.rad2deg(2 * np.arctan(height / (2 * fy)))
            # The ZedCamera class doesn't have a native color_fovy, we can assign it if needed
            camera.color_fovy = camera.depth_fovy
            
            self.cameras[camera_name] = camera
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
        self._set_action(action, duration=self.dt, joint_vel_limit_scale=2.0, wait=True)

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
            + len(self.rgb_tactile_names)
            + len(self.pointcloud_camera_names)
            == 0
        ):
            return info

        info["rgb_images"] = {}
        info["depth_images"] = {}
        if len(self.pointcloud_camera_names) > 0:
            info["pointclouds"] = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}

            for camera_name, camera in self.cameras.items():
                futures[executor.submit(self.get_camera_data, camera_name, camera)] = (
                    camera_name
                )

            for rgb_tactile_name, rgb_tactile in self.rgb_tactiles.items():
                futures[
                    executor.submit(
                        self.get_rgb_tactile_data, rgb_tactile_name, rgb_tactile
                    )
                ] = rgb_tactile_name

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

            for future in concurrent.futures.as_completed(futures):
                name, rgb_image, depth_image, pointcloud = future.result()
                info["rgb_images"][name] = rgb_image
                info["depth_images"][name] = depth_image
                if pointcloud is not None:
                    info["pointclouds"][name] = pointcloud

        return info

    def get_camera_data(self, camera_name, camera, img_size=(640, 480), verbose=False):  # Merged
        """
        Reads an image from a camera, handling different camera types.
        """

        if isinstance(camera, RealSenseCamera):
            rgb_image, depth_image = camera.read(img_size)
            depth_image = (1e-3 * depth_image[:, :, 0]).astype(np.float32)  # [m]

        elif isinstance(camera, ZedCamera):
            data = camera.read(img_size=img_size, include_depth=True)
            if data is None:
                raise RuntimeError(f"[{self.__class__.__name__}] Failed to read from ZED camera: {camera_name}")

            # The ZedCamera class uses the 'left_image' key for the BGR image.
            bgr_image = data['left_image']
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            # ZED depth is (H, W), uint16 in mm -> convert to float32 in m
            depth_image = (1e-3 * data['depth_image']).astype(np.float32)

        elif isinstance(camera, OakCamera):
            data = camera.read(img_size=img_size, include_depth=True)
            if data is None:
                raise RuntimeError(f"[{self.__class__.__name__}] Failed to read from OAK camera: {camera_name}")

            # The OakCamera class returns a BGR image in the 'color_image' key.
            bgr_image = data['color_image']
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            # OAK depth is (H, W), uint16 in mm -> convert to float32 in m
            depth_image_raw = data['depth_image']
            depth_image = (1e-3 * depth_image_raw).astype(np.float32)

        else:
            raise TypeError(f"Unsupported camera type: {type(camera)}")

        if verbose:
            # np.mean can be slow, only use for debugging
            average_depth = np.mean(depth_image[np.nonzero(depth_image)])
            print(f"The average depth (z value) for non-zero pixels is: {average_depth:.4f} meters")

        return camera_name, rgb_image, depth_image, None  # Modified return

    def get_rgb_tactile_data(self, rgb_tactile_name, rgb_tactile):
        ret, rgb_image = rgb_tactile.read()
        if not ret:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Failed to read tactile image."
            )
        image_size = (640, 480)
        rgb_image = cv2.resize(rgb_image, image_size)
        return rgb_tactile_name, rgb_image, None, None

    def get_pointcloud_camera_data(self, pointcloud_camera_name, pointcloud_camera):
        from pyorbbecsdk import OBFormat

        frames = pointcloud_camera["queue"].get()

        rgb_frame = frames.get_color_frame()
        if rgb_frame is None:
            return pointcloud_camera_name, None, None, None
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
            return pointcloud_camera_name, rgb_image, None, None
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

        return pointcloud_camera_name, rgb_image, depth_image, pointcloud

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
    def rgb_tactile_names(self):
        """Get names of tactile sensors with RGB output."""
        return list(self.rgb_tactiles.keys())

    @property
    def pointcloud_camera_names(self):
        """Get pointcloud camera names."""
        return list(self.pointcloud_cameras.keys())

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
