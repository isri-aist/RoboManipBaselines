from .DataKey import DataKey
from lerobot.teleoperators.utils import TeleopEvents


class TeleopOperationDataMixin:
    def record_data(self):
        """
        Overrides OperationDataMixin.record_data to conditionally save camera feeds
        and add HIL-specific data with verbose logging.
        """
        if self.args.verbose:
            print(f"[Verbose] Recording data for step. info keys: {self.info.keys()}")

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

        # --- MODIFICATION: Enhanced verbose logging for camera data ---
        if self.args.save_camera_feed:
            if self.args.verbose:
                print("[Verbose] save_camera_feed is True. Checking for image data.")

            if "rgb_images" in self.info and self.info["rgb_images"] is not None:
                if self.args.verbose:
                    print(f"[Verbose] Found 'rgb_images' with keys: {self.info['rgb_images'].keys()}")
                for camera_name in self.env.unwrapped.camera_names:
                    if camera_name in self.info["rgb_images"]:
                        self.data_manager.append_single_data(
                            DataKey.get_rgb_image_key(camera_name),
                            self.info["rgb_images"][camera_name],
                        )
                if hasattr(self.env.unwrapped, "rgb_tactile_names"):
                    for rgb_tactile_name in self.env.unwrapped.rgb_tactile_names:
                        if rgb_tactile_name in self.info["rgb_images"]:
                            self.data_manager.append_single_data(
                                DataKey.get_rgb_image_key(rgb_tactile_name),
                                self.info["rgb_images"][rgb_tactile_name],
                            )
            elif self.args.verbose:
                print("[Verbose] 'rgb_images' not found in info dict or is None.")

            if "depth_images" in self.info and self.info["depth_images"] is not None:
                if self.args.verbose:
                    print(f"[Verbose] Found 'depth_images' with keys: {self.info['depth_images'].keys()}")
                for camera_name in self.env.unwrapped.camera_names:
                    if camera_name in self.info["depth_images"]:
                        self.data_manager.append_single_data(
                            DataKey.get_depth_image_key(camera_name),
                            self.info["depth_images"][camera_name],
                        )
            elif self.args.verbose:
                print("[Verbose] 'depth_images' not found in info dict or is None.")
        elif self.args.verbose:
            print("[Verbose] save_camera_feed is False. Skipping image data.")

        # Add tactile
        if "intensity_tactile" in self.info:
            for intensity_tactile_name in self.info["intensity_tactile"]:
                self.data_manager.append_single_data(
                    intensity_tactile_name,
                    self.info["intensity_tactile"][intensity_tactile_name].copy(),
                )

        # HIL-SERL specific data
        if (
            hasattr(self.data_manager, "data_chunk")
            and self.data_manager.data_chunk is not None
        ):
            self.data_manager.data_chunk["is_intervention"][-1] = self.is_intervention
            self.data_manager.data_chunk["success"][-1] = self.teleop_events.get(
                TeleopEvents.SUCCESS, False
            )
