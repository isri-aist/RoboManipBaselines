import concurrent.futures
import os

import h5py
import numpy as np
import videoio

from robo_manip_baselines import __version__

from ..data.DataKey import DataKey
from ..data.RmbData import RmbData
from ..utils.MathUtils import (
    get_rel_pose_from_se3,
    get_se3_from_pose,
    get_se3_from_rel_pose,
)


class DataManager:
    """Data manager."""

    def __init__(self, env, demo_name="", task_desc=""):
        self.env = env

        self.meta_data = {
            "demo_name": demo_name,
            "task_desc": task_desc,
            "version": __version__,
        }
        if self.env is not None:
            self.meta_data["env"] = self.env.spec.name

        self.episode_idx = 0
        self.world_idx = 0

        self.reset()

    def reset(self):
        """Reset."""
        self.all_data_seq = {}

    def append_single_data(self, key, data):
        """Append a single data to the data sequence."""
        if key not in self.all_data_seq:
            self.all_data_seq[key] = []
        self.all_data_seq[key].append(data)

    def get_single_data(self, key, time_idx):
        """Get a single data from the data sequence."""
        return self.all_data_seq[key][time_idx]

    def get_data_seq(self, key):
        """Get data sequence."""
        return self.all_data_seq[key]

    def get_meta_data(self, key):
        """Get meta data."""
        return self.meta_data[key]

    def calc_rel_data(self, key, all_data_seq=None):
        """Calculate relative data."""
        if all_data_seq is None:
            all_data_seq = self.all_data_seq

        abs_key = DataKey.get_abs_key(key)

        if key in (
            DataKey.MEASURED_JOINT_POS_REL,
            DataKey.COMMAND_JOINT_POS_REL,
            DataKey.MEASURED_GRIPPER_JOINT_POS_REL,
            DataKey.COMMAND_GRIPPER_JOINT_POS_REL,
        ):
            if len(all_data_seq[abs_key]) < 2:
                return np.zeros_like(all_data_seq[abs_key][0])
            else:
                current_pos = all_data_seq[abs_key][-1]
                prev_pos = all_data_seq[abs_key][-2]
                return current_pos - prev_pos
        elif key in (DataKey.MEASURED_EEF_POSE_REL, DataKey.COMMAND_EEF_POSE_REL):
            if len(all_data_seq[abs_key]) < 2:
                return np.zeros(6 * (len(all_data_seq[abs_key][0]) // 7))
            else:
                current_pose_list = all_data_seq[abs_key][-1].reshape(-1, 7)
                prev_pose_list = all_data_seq[abs_key][-2].reshape(-1, 7)
                rel_pose_list = []
                for current_pose, prev_pose in zip(current_pose_list, prev_pose_list):
                    current_se3 = get_se3_from_pose(current_pose)
                    prev_se3 = get_se3_from_pose(prev_pose)
                    rel_pose_list.append(
                        get_rel_pose_from_se3(prev_se3.actInv(current_se3))
                    )
                return np.concatenate(rel_pose_list)
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Unsupported key for relative data: {key}"
            )

    def save_data(
        self, filename, all_data_seq=None, meta_data=None, increment_episode_idx=True
    ):
        """Save data."""
        if all_data_seq is None:
            all_data_seq = self.all_data_seq
        if meta_data is None:
            meta_data = self.meta_data

        _, ext = os.path.splitext(filename.rstrip("/"))
        if ext.lower() == ".rmb":
            self.dump_to_rmb(filename, all_data_seq, meta_data)
        elif ext.lower() == ".hdf5":
            self.dump_to_hdf5(filename, all_data_seq, meta_data)
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid file extension '{ext}'. Expected '.hdf5' or '.rmb': {filename}"
            )

        if increment_episode_idx:
            self.episode_idx += 1

    def dump_to_rmb(self, filename, all_data_seq, meta_data):
        os.makedirs(filename)
        hdf5_filename = os.path.join(filename, "main.rmb.hdf5")
        with h5py.File(hdf5_filename, "w") as h5file:
            tasks = []
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for key in all_data_seq.keys():
                    if not isinstance(all_data_seq[key], list):
                        raise ValueError(
                            f"[{self.__class__.__name__}] Unsupported type of data sequence: {type(all_data_seq[key])}"
                        )

                    if DataKey.is_rgb_image_key(key):
                        video_filename = os.path.join(filename, f"{key}.rmb.mp4")
                        images = np.array(all_data_seq[key])
                        tasks.append(
                            executor.submit(self.save_rgb_image, video_filename, images)
                        )
                    elif DataKey.is_depth_image_key(key):
                        video_filename = os.path.join(filename, f"{key}.rmb.mp4")
                        images = (1e3 * np.array(all_data_seq[key])).astype(np.uint16)
                        tasks.append(
                            executor.submit(
                                self.save_depth_image, video_filename, images
                            )
                        )
                    else:
                        h5file.create_dataset(key, data=np.array(all_data_seq[key]))

                concurrent.futures.wait(tasks)
                for task in tasks:
                    err = task.exception()
                    if err is not None:
                        print(
                            f"[{self.__class__.__name__}] Failed to save images: {err}"
                        )

            for key in meta_data.keys():
                h5file.attrs[key] = meta_data[key]
            h5file.attrs["format"] = "RmbData-Compact"

    def dump_to_hdf5(self, filename, all_data_seq, meta_data):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with h5py.File(filename, "w") as h5file:
            for key in all_data_seq.keys():
                if not isinstance(all_data_seq[key], list):
                    raise ValueError(
                        f"[{self.__class__.__name__}] Unsupported type of data sequence: {type(all_data_seq[key])}"
                    )

                h5file.create_dataset(key, data=np.array(all_data_seq[key]))

            for key in meta_data.keys():
                h5file.attrs[key] = meta_data[key]
            h5file.attrs["format"] = "RmbData-SingleHDF5"

    @staticmethod
    def save_rgb_image(video_filename, images):
        videoio.videosave(video_filename, images)

    @staticmethod
    def save_depth_image(video_filename, images):
        videoio.uint16save(video_filename, images)

    def load_data(self, filename, load_keys=None, skip_image=False):
        """Load data."""
        self.all_data_seq = {}
        self.meta_data = {}
        with RmbData(filename, "r") as rmb_data:
            for key in rmb_data.keys():
                if (load_keys is not None) and (key not in load_keys):
                    continue
                elif skip_image and (
                    DataKey.is_rgb_image_key(key) or DataKey.is_depth_image_key(key)
                ):
                    continue
                else:
                    self.all_data_seq[key] = rmb_data[key][:]
            for key in rmb_data.attrs.keys():
                self.meta_data[key] = rmb_data.attrs[key]

    def reverse_data(self, all_data_seq=None):
        """Reverse sequence data."""
        if all_data_seq is None:
            all_data_seq = self.all_data_seq

        for key in all_data_seq.keys():
            if key == DataKey.TIME:
                continue

            if not isinstance(all_data_seq[key], list):
                raise ValueError(
                    f"[{self.__class__.__name__}] Unsupported type of data sequence: {type(all_data_seq[key])}"
                )

            all_data_seq[key].reverse()

            if key in (
                DataKey.MEASURED_JOINT_VEL,
                DataKey.COMMAND_JOINT_VEL,
                DataKey.MEASURED_EEF_VEL,
                DataKey.COMMAND_EEF_VEL,
                DataKey.MEASURED_JOINT_POS_REL,
                DataKey.COMMAND_JOINT_POS_REL,
                DataKey.MEASURED_GRIPPER_JOINT_POS_REL,
                DataKey.COMMAND_GRIPPER_JOINT_POS_REL,
            ):
                all_data_seq[key] = [-1.0 * data for data in all_data_seq[key]]
            elif key in (DataKey.MEASURED_EEF_POSE_REL, DataKey.COMMAND_EEF_POSE_REL):
                all_data_seq[key] = [
                    get_rel_pose_from_se3(get_se3_from_rel_pose(data).inverse())
                    for data in all_data_seq[key]
                ]

    def setup_env_world(self, world_idx=None):
        """Setup the environment world."""
        if world_idx is None:
            kwargs = {"cumulative_idx": self.episode_idx}
        else:
            kwargs = {"world_idx": world_idx}
        self.world_idx = self.env.unwrapped.modify_world(**kwargs)
        self.meta_data["world_idx"] = self.world_idx

    def setup_camera_info(self):
        """Set camera info."""
        self.meta_data["camera_names"] = self.env.unwrapped.camera_names
        self.meta_data["rgb_tactile_names"] = self.env.unwrapped.rgb_tactile_names
        self.meta_data["pointcloud_camera_names"] = (
            self.env.unwrapped.pointcloud_camera_names
        )
        for camera_name in self.env.unwrapped.camera_names:
            depth_key = DataKey.get_depth_image_key(camera_name)
            self.meta_data[depth_key + "_fovy"] = self.env.unwrapped.get_camera_fovy(
                camera_name
            )
