import argparse
import dataclasses
import json
import shutil
from typing import Literal

import numpy as np
import torch
import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

from robo_manip_baselines.common import DataKey, RmbData, find_rmb_files


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


class ConvertRmbDataToLerobot:
    def __init__(self, path, repo_id, robot_type):
        self.rmb_path_list = find_rmb_files(path)
        self.repo_id = repo_id
        self.robot_type = robot_type

    def create_empty_dataset(
        self,
        repo_id: str,
        robot_type: str,
        mode: Literal["video", "image"] = "video",
        *,
        dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    ):
        with RmbData(self.rmb_path_list[0]) as rmb_data:
            num_joints = rmb_data[DataKey.COMMAND_JOINT_POS].shape[1]
            self.joint_name_list = [
                f"joint_{joint_idx}" for joint_idx in range(num_joints)
            ]
            self.camera_name_list = rmb_data.attrs["camera_names"]

            features = {
                "observation.state": {
                    "dtype": "float64",
                    "shape": (len(self.joint_name_list),),
                    "names": [
                        self.joint_name_list,
                    ],
                },
                "action": {
                    "dtype": "float64",
                    "shape": (len(self.joint_name_list),),
                    "names": [
                        self.joint_name_list,
                    ],
                },
            }

            self.has_velocity = DataKey.MEASURED_JOINT_VEL in rmb_data
            if self.has_velocity:
                features["observation.velocity"] = {
                    "dtype": "float64",
                    "shape": (len(self.joint_name_list),),
                    "names": [
                        self.joint_name_list,
                    ],
                }

            self.has_effort = DataKey.MEASURED_JOINT_TORQUE in rmb_data
            if self.has_effort:
                features["observation.effort"] = {
                    "dtype": "float64",
                    "shape": (len(self.joint_name_list),),
                    "names": [
                        self.joint_name_list,
                    ],
                }

            for camera_name in self.camera_name_list:
                rgb_image_key = DataKey.get_rgb_image_key(camera_name)
                rgb_image_shape = rmb_data[rgb_image_key][0].shape
                print(rgb_image_shape, (480, 640, 3))
                features[f"observation.images.{camera_name}_rgb"] = {
                    "dtype": mode,
                    "shape": rgb_image_shape,
                    "names": [
                        "height",
                        "width",
                        "channels",
                    ],
                }

        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=30,
            robot_type=robot_type,
            features=features,
            use_videos=dataset_config.use_videos,
            tolerance_s=dataset_config.tolerance_s,
            image_writer_processes=dataset_config.image_writer_processes,
            image_writer_threads=dataset_config.image_writer_threads,
            video_backend=dataset_config.video_backend,
        )

    def load_raw_images_per_camera(self, rmb_data) -> dict[str, np.ndarray]:
        images_per_camera = {}
        for camera_name in self.camera_name_list:
            rgb_image_key = DataKey.get_rgb_image_key(camera_name)
            images_per_camera[f"{camera_name}_rgb"] = rmb_data[rgb_image_key][:]
            print(f"{camera_name}_rgb", images_per_camera[f"{camera_name}_rgb"].shape)
        return images_per_camera

    def load_raw_episode_data(
        self, rmb_data
    ) -> tuple[
        dict[str, np.ndarray],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        state_joint = rmb_data[DataKey.MEASURED_JOINT_POS][:]
        action_joint = rmb_data[DataKey.COMMAND_JOINT_POS][:]
        if self.robot_type == "hsr":
            state_omni = rmb_data[DataKey.MEASURED_MOBILE_OMNI_VEL][:]
            state_all = np.concatenate([state_joint, state_omni], axis=1)

            action_omni = rmb_data[DataKey.COMMAND_MOBILE_OMNI_VEL][:]
            action_all = np.concatenate([action_joint, action_omni], axis=1)

            state = torch.from_numpy(state_all)
            action = torch.from_numpy(action_all)
        else:
            state = torch.from_numpy(state_joint)
            action = torch.from_numpy(action_joint)

        # "/measured_joint_vel" in ep and "/measured_mobile_omni_vel" in ep:
        if self.has_velocity:
            velocity_joint = rmb_data[DataKey.MEASURED_JOINT_VEL][:]
            if self.robot_type == "hsr":
                velocity_omni = rmb_data[DataKey.MEASURED_MOBILE_OMNI_VEL][:]
                velocity_all = np.concatenate([velocity_joint, velocity_omni], axis=1)
                velocity = torch.from_numpy(velocity_all)
            else:
                velocity = torch.from_numpy(velocity_joint)
        else:
            velocity = None

        if self.has_effort:
            raise NotImplementedError(
                f"[{self.__class__.__name__}] Conversion of effort data is not supported."
            )
        else:
            effort = None

        images_per_camera = self.load_raw_images_per_camera(rmb_data)

        return images_per_camera, state, action, velocity, effort

    def populate_dataset(
        self,
        episodes: list[int] | None = None,
    ):
        if episodes is None:
            episodes = range(len(self.rmb_path_list))

        for rmb_idx, rmb_path in tqdm.tqdm(enumerate(self.rmb_path_list)):
            with RmbData(rmb_path) as rmb_data:
                if "task_desc" in rmb_data.attrs:
                    task_desc = rmb_data.attrs["task_desc"]
                else:
                    env_name = rmb_data.attrs["env"]
                    if env_name == "MujocoUR5eCableEnv":
                        task_desc = "pass the cable between two poles"
                    elif env_name == "MujocoUR5eRingEnv":
                        task_desc = "pick a ring and put it around the pole"
                    elif env_name == "MujocoUR5eParticleEnv":
                        task_desc = "scoop up particles"
                    elif env_name == "MujocoUR5eClothEnv":
                        task_desc = "roll up the cloth"
                    elif env_name == "MujocoUR5eDoorEnv":
                        task_desc = "open the door"
                    elif env_name == "MujocoHsrTidyupEnv":
                        task_desc = "Bring the object to the box"

                images_per_camera, state, action, velocity, effort = (
                    self.load_raw_episode_data(rmb_data)
                )
                num_frames = len(rmb_data[DataKey.TIME])

                if rmb_idx == 0:
                    with open(self.dataset.root / "meta" / "modality.json", "w") as f:
                        modality = {
                            "state": {
                                "single_arm": {
                                    "start": 0,
                                    "end": len(self.joint_name_list),
                                }
                            },
                            "action": {
                                "single_arm": {
                                    "start": 0,
                                    "end": len(self.joint_name_list),
                                }
                            },
                            "video": {
                                f"{camera_name}_rgb": {
                                    "original_key": f"observation.images.{camera_name}_rgb"
                                }
                                for camera_name in self.camera_name_list
                            },
                            "annotation": {
                                "human.task_description": {"original_key": "task_index"}
                            },
                        }
                        json.dump(modality, f, indent=4)

                for i in range(num_frames):
                    frame = {
                        "observation.state": state[i],
                        "action": action[i],
                    }

                    for camera_name, image in images_per_camera.items():
                        frame[f"observation.images.{camera_name}"] = image[i]

                    if self.has_velocity:
                        frame["observation.velocity"] = velocity[i]
                    if self.has_effort:
                        frame["observation.effort"] = effort[i]

                    frame["task"] = task_desc

                    self.dataset.add_frame(frame)

                self.dataset.save_episode()

    def get_stats_einops_patterns(self, num_workers=0):
        """These einops patterns will be used to aggregate batches and compute statistics.

        Note: We assume the images are in channel first format
        """

        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            num_workers=num_workers,
            batch_size=2,
            shuffle=False,
        )
        batch = next(iter(dataloader))

        stats_patterns = {}

        for key in self.dataset.features:
            # sanity check that tensors are not float64
            assert batch[key].dtype != torch.float64

            # if isinstance(feats_type, (VideoFrame, Image)):
            if key in self.dataset.meta.camera_keys:
                # sanity check that images are channel first
                _, c, h, w = batch[key].shape
                assert (
                    c < h and c < w
                ), f"expect channel first images, but instead {batch[key].shape}"

                # sanity check that images are float32 in range [0,1]
                assert (
                    batch[key].dtype == torch.float32
                ), f"expect torch.float32, but instead {batch[key].dtype=}"
                assert (
                    batch[key].max() <= 1
                ), f"expect pixels lower than 1, but instead {batch[key].max()=}"
                assert (
                    batch[key].min() >= 0
                ), f"expect pixels greater than 1, but instead {batch[key].min()=}"

                stats_patterns[key] = "b c h w -> c 1 1"
            elif batch[key].ndim == 2:
                stats_patterns[key] = "b c -> c "
            elif batch[key].ndim == 1:
                stats_patterns[key] = "b -> 1"
            else:
                raise ValueError(f"{key}, {batch[key].shape}")

        return stats_patterns

    def create_seeded_dataloader(self, batch_size, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            num_workers=8,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
        return dataloader

    def flatten_dict(self, d: dict, parent_key: str = "", sep: str = "/") -> dict:
        """Flatten a nested dictionary structure by collapsing nested keys into one key with a separator.

        For example:
        ```
        >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}`
        >>> print(flatten_dict(dct))
        {"a/b": 1, "a/c/d": 2, "e": 3}
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def unflatten_dict(self, d: dict, sep: str = "/") -> dict:
        outdict = {}
        for key, value in d.items():
            parts = key.split(sep)
            d = outdict
            for part in parts[:-1]:
                if part not in d:
                    d[part] = {}
                d = d[part]
            d[parts[-1]] = value
        return outdict

    def serialize_dict(
        self, stats: dict[str, torch.Tensor | np.ndarray | dict]
    ) -> dict:
        serialized_dict = {
            key: value.tolist() for key, value in self.flatten_dict(stats).items()
        }
        return self.unflatten_dict(serialized_dict)

    def port_data(
        self,
        episodes: list[int] | None = None,
        push_to_hub: bool = False,
        mode: Literal["video", "image"] = "video",
        dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    ):
        if (HF_LEROBOT_HOME / self.repo_id).exists():
            shutil.rmtree(HF_LEROBOT_HOME / self.repo_id)

        self.create_empty_dataset(
            self.repo_id,
            robot_type=self.robot_type,
            mode=mode,
            dataset_config=dataset_config,
        )
        self.populate_dataset(
            episodes=episodes,
        )
        self.dataset.finalize()

        meta_stats = self.dataset.meta.stats
        self.dataset = LeRobotDataset(
            repo_id=self.repo_id, root=(HF_LEROBOT_HOME / self.repo_id)
        )

        stats_patterns = self.get_stats_einops_patterns(8)

        data_num = len(self.dataset)
        q01, q99 = {}, {}
        data_dir = {}

        for key, pattern in stats_patterns.items():
            if key in self.dataset.meta.camera_keys:
                continue
            data_dir[key] = []
            for i in range(data_num):
                data_dir[key].append(self.dataset[i][key].float())
            data_dir[key] = torch.stack(data_dir[key], dim=0)

            q01[key] = torch.quantile(data_dir[key], 0.01, 0)
            q99[key] = torch.quantile(data_dir[key], 0.99, 0)

        for key in stats_patterns:
            if key in self.dataset.meta.camera_keys:
                continue
            meta_stats[key]["q01"] = q01[key]
            meta_stats[key]["q99"] = q99[key]

        serialized_stats = self.serialize_dict(meta_stats)

        with open(self.dataset.root / "meta" / "stats.json", "w") as f:
            json.dump(serialized_stats, f, indent=4)

        if push_to_hub:
            self.dataset.push_to_hub()

        print("Finished converting dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--repo_id", type=str)
    parser.add_argument("--robot_type", type=str, default="ur5e")
    args = parser.parse_args()

    rmb_to_lerobot = ConvertRmbDataToLerobot(**vars(args))
    rmb_to_lerobot.port_data()
