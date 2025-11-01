from .DataKey import DataKey


class OperationDataMixin:
    def record_data(self):
        # Add time
        self.data_manager.append_single_data(
            DataKey.TIME, self.phase_manager.phase.get_elapsed_duration()
        )

        # Add reward
        self.data_manager.append_single_data(DataKey.REWARD, self.reward)

        # Add measured data
        for key in self.env.unwrapped.measured_keys_to_save:
            self.data_manager.append_single_data(
                key, self.motion_manager.get_measured_data(key, self.obs)
            )

        # Add command data
        for key in self.env.unwrapped.command_keys_to_save:
            self.data_manager.append_single_data(
                key, self.motion_manager.get_command_data(key)
            )

        # Add relative data
        for key in (
            DataKey.MEASURED_JOINT_POS_REL,
            DataKey.COMMAND_JOINT_POS_REL,
            DataKey.MEASURED_GRIPPER_JOINT_POS_REL,
            DataKey.COMMAND_GRIPPER_JOINT_POS_REL,
            DataKey.MEASURED_EEF_POSE_REL,
            DataKey.COMMAND_EEF_POSE_REL,
        ):
            abs_key = DataKey.get_abs_key(key)
            if abs_key not in (
                *self.env.unwrapped.measured_keys_to_save,
                *self.env.unwrapped.command_keys_to_save,
            ):
                continue

            self.data_manager.append_single_data(
                key, self.data_manager.calc_rel_data(key)
            )

        # Add image
        for camera_name in self.env.unwrapped.camera_names:
            self.data_manager.append_single_data(
                DataKey.get_rgb_image_key(camera_name),
                self.info["rgb_images"][camera_name],
            )
            self.data_manager.append_single_data(
                DataKey.get_depth_image_key(camera_name),
                self.info["depth_images"][camera_name],
            )
        for rgb_tactile_name in self.env.unwrapped.rgb_tactile_names:
            self.data_manager.append_single_data(
                DataKey.get_rgb_image_key(rgb_tactile_name),
                self.info["rgb_images"][rgb_tactile_name],
            )
        for pointcloud_camera_name in self.env.unwrapped.pointcloud_camera_names:
            self.data_manager.append_single_data(
                DataKey.get_rgb_image_key(pointcloud_camera_name),
                self.info["rgb_images"][pointcloud_camera_name],
            )
            self.data_manager.append_single_data(
                DataKey.get_depth_image_key(pointcloud_camera_name),
                self.info["depth_images"][pointcloud_camera_name],
            )

        # Add tactile
        if "intensity_tactile" in self.info:
            for intensity_tactile_name in self.info["intensity_tactile"]:
                self.data_manager.append_single_data(
                    intensity_tactile_name,
                    self.info["intensity_tactile"][intensity_tactile_name].copy(),
                )

        # Add pointcloud
        if "pointclouds" in self.info:
            for pointcloud_camera_name in self.info["pointclouds"]:
                self.data_manager.append_single_data(
                    DataKey.get_pointcloud_key(pointcloud_camera_name) + "_raw",
                    self.info["pointclouds"][pointcloud_camera_name].copy(),
                )
