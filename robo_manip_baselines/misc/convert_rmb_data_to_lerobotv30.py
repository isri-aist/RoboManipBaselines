#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generic script to convert HDF5-based data (including .rmb format) to the LeRobot dataset v3.0 format for local use.

This script uses a YAML configuration file to map the raw data structure to the LeRobot format,
supporting images stored either inside the HDF5 file or as external MP4 videos. It can also generate
ground-truth reward and done labels for demonstration data.

Key Features:
- Creates datasets directly in the LeRobot v3.0 format.
- Flexible data mapping via a YAML configuration file that supports concatenating multiple keys.
- Creates separate, optional features for `state`, `velocity`, and `effort`.
- Support for images embedded in HDF5 or as external MP4 videos.
- On-the-fly image cropping and resizing to a target resolution specified in the config.
- Automatic calculation of dataset statistics for normalization.
- Generation of success/failure rewards and done flags based on trajectory completion.
- Option for a flat reward (`--flat-reward`) or a proportional reward (`--proportional-reward`) for non-terminal steps.
- At the end, it outputs a summary of state and action dimensions from the original and final datasets.
- **DEBUG: Includes detailed logging and data truncation to handle frame count mismatches.**
- Designed for local-only conversion (no Hugging Face Hub integration).

Example usage for .rmb format with cropping and proportional reward:
    python /path/to/this/script.py \\
        --raw-dir /path/to/raw_rmb_data \\
        --repo-id my-local-dataset-name \\
        --config-path /path/to/your/config.yaml \\
        --camera-crops front_camera:10,20,224,224 \\
        --proportional-reward
"""

import dataclasses
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import cv2
import h5py
import numpy as np
import torch
import tqdm
import tyro
import videoio
import yaml
# MODIFICATION: Import the `datasets` library to explicitly define features.
import datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# --- Setup logging ---
LOGGER = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    """Initializes the logger for clear console output."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_lerobot_home() -> Path:
    """
    Determines the LeRobot cache directory path, prioritizing environment variables.

    Returns:
        The resolved path to the LeRobot home directory.
    """
    if hf_home := os.getenv("HF_LEROBOT_HOME"):
        return Path(hf_home)
    try:
        from lerobot.common.constants import HF_LEROBOT_HOME

        return Path(HF_LEROBOT_HOME)
    except ImportError:
        LOGGER.warning(
            "HF_LEROBOT_HOME not set and `lerobot` library not found. "
            "Using default: ~/.cache/lerobot/hub"
        )
        return Path.home() / ".cache" / "lerobot" / "hub"


HF_LEROBOT_HOME = get_lerobot_home()


# --- Configuration ---


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    """Configuration for the dataset writing process."""

    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: Optional[str] = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def load_config(config_path: Path) -> Dict[str, Any]:
    """Loads and validates the YAML configuration file."""
    LOGGER.info(f"Loading configuration from {config_path}")
    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    return config


# --- HELPER FUNCTIONS FOR HANDLING MULTIPLE KEYS ---


def _keys_exist(h5_file: h5py.File, keys: Optional[Union[str, List[str]]]) -> bool:
    """Checks if a single key or all keys in a list exist in the HDF5 file."""
    if keys is None:
        return False
    if isinstance(keys, str):
        return keys in h5_file
    if isinstance(keys, list):
        return all(key in h5_file for key in keys)
    return False


def _get_concatenated_shape(h5_file: h5py.File, keys: Union[str, List[str]]):
    """Calculates the total feature dimension when concatenating multiple keys."""
    if isinstance(keys, str):
        keys = [keys]
    # Sum the feature dimensions (axis 1) of all specified keys.
    total_dim = sum(h5_file[key].shape[1] for key in keys if key in h5_file)
    return (total_dim,)


def _load_and_concat_keys(h5_file: h5py.File, keys: Union[str, List[str]]):
    """Loads data for multiple keys and concatenates them into a single array."""
    if isinstance(keys, str):
        keys = [keys]

    # Load data for each key that exists in the file
    data_parts = [h5_file[key][:] for key in keys if key in h5_file]

    if not data_parts:
        return []

    # Concatenate along the feature axis (axis=1)
    concatenated_data = np.concatenate(data_parts, axis=1)

    # Return as a list of frame-by-frame tensors, as expected by the rest of the script
    return [torch.from_numpy(frame).float() for frame in concatenated_data]


# --- Dataset Creation and Schema ---


def create_empty_dataset(
    repo_id: str,
    config: Dict[str, Any],
    hdf5_files: List[Path],
    mode: Literal["video", "image"],
    dataset_config: DatasetConfig,
) -> Tuple[LeRobotDataset, Dict, Dict]:
    """
    Creates an empty LeRobotDataset with a schema defined by the config file.
    Returns the dataset object and the inferred original shapes for state and action.
    """
    features = {}
    data_mapping = config["data_mapping"]
    original_shapes = {}

    if "target_camera_resolution" in config:
        height, width = config["target_camera_resolution"]
        LOGGER.info(f"Using target resolution for dataset schema: {width}x{height}")
    else:
        height, width = config.get("camera_resolution", (224, 224))

    with h5py.File(hdf5_files[0], "r") as h5_file:
        state_keys = data_mapping.get("state")
        if state_keys and _keys_exist(h5_file, state_keys):
            state_shape = _get_concatenated_shape(h5_file, state_keys)
            features["observation.state"] = {"dtype": "float32", "shape": state_shape}
            original_shapes["state"] = state_shape

        velocity_keys = data_mapping.get("velocity")
        if velocity_keys and _keys_exist(h5_file, velocity_keys):
            velocity_shape = _get_concatenated_shape(h5_file, velocity_keys)
            features["observation.velocity"] = {"dtype": "float32", "shape": velocity_shape}
            original_shapes["velocity"] = velocity_shape

        effort_keys = data_mapping.get("effort")
        if effort_keys and _keys_exist(h5_file, effort_keys):
            effort_shape = _get_concatenated_shape(h5_file, effort_keys)
            features["observation.effort"] = {"dtype": "float32", "shape": effort_shape}
            original_shapes["effort"] = effort_shape

        action_shape = _get_concatenated_shape(h5_file, data_mapping["action"])
        features["action"] = {"dtype": "float32", "shape": action_shape}
        original_shapes["action"] = action_shape

    features["next.reward"] = {"dtype": "float32", "shape": (1,)}
    features["next.done"] = {"dtype": "bool", "shape": (1,)}

    for cam_name in config["camera_names"]:
        # MODIFICATION: Define both image and video features as dictionaries
        # to ensure the 'dtype' key is always present.
        features[f"observation.images.{cam_name}"] = {
            "dtype": mode,
            "shape": (height, width, 3), # Note: LeRobot expects H, W, C
            "names": ["height", "width", "channels"],
        }

    # Clean up any existing dataset directory
    dataset_path = HF_LEROBOT_HOME / repo_id
    if dataset_path.exists():
        LOGGER.warning(f"Deleting existing dataset at {dataset_path}")
        shutil.rmtree(dataset_path)
    LOGGER.info(f"Creating new dataset at: {dataset_path}")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=config["fps"],
        robot_type=config.get("robot_type", "unknown"),
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )
    return dataset, original_shapes


# --- Data Loading ---


def load_images_from_hdf5(ep_file: h5py.File, config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Loads and decodes images from datasets within an HDF5 file."""
    imgs_per_cam = {}
    image_key_template = config["data_mapping"]["image_template"]

    for camera_name in config["camera_names"]:
        image_key = image_key_template.format(camera_name=camera_name)
        if image_key not in ep_file:
            LOGGER.warning(f"Image key '{image_key}' not found in HDF5 file. Skipping camera '{camera_name}'.")
            continue

        image_dataset = ep_file[image_key]

        if image_dataset.ndim == 4 and image_dataset.dtype == np.uint8:
            imgs_array = image_dataset[:]
        else:
            imgs_array = [cv2.imdecode(data, cv2.IMREAD_COLOR) for data in image_dataset]
            imgs_array = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs_array])
        imgs_per_cam[camera_name] = imgs_array
    return imgs_per_cam


def load_images_from_mp4(ep_path: Path, config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Loads images from external .mp4 video files."""
    imgs_per_cam = {}
    video_dir = ep_path.parent
    image_key_template = config["data_mapping"]["image_template"]

    for camera_name in config["camera_names"]:
        video_filename = image_key_template.format(camera_name=camera_name)
        video_path = video_dir / video_filename

        if not video_path.exists():
            LOGGER.warning(f"Video file not found at {video_path}, skipping camera {camera_name}")
            continue

        imgs_array = videoio.videoread(str(video_path))
        imgs_per_cam[camera_name] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
    config: Dict[str, Any],
    flat_reward: float,
    proportional_reward: bool,
    camera_crops: Dict[str, Tuple[int, int, int, int]],
) -> Optional[Dict[str, Any]]:
    """Loads all data for a single episode from HDF5 and associated videos."""
    data = {}
    data_mapping = config["data_mapping"]

    try:
        with h5py.File(ep_path, "r") as ep_file:
            data_streams = {}

            state_keys = data_mapping.get("state")
            if state_keys and _keys_exist(ep_file, state_keys):
                data_streams["state"] = _load_and_concat_keys(ep_file, state_keys)

            velocity_keys = data_mapping.get("velocity")
            if velocity_keys and _keys_exist(ep_file, velocity_keys):
                data_streams["velocity"] = _load_and_concat_keys(ep_file, velocity_keys)

            effort_keys = data_mapping.get("effort")
            if effort_keys and _keys_exist(ep_file, effort_keys):
                data_streams["effort"] = _load_and_concat_keys(ep_file, effort_keys)
            
            data_streams["action"] = _load_and_concat_keys(ep_file, data_mapping["action"])

            task_key = config.get("task_description_key")
            if task_key and task_key in ep_file.attrs:
                task_desc = ep_file.attrs[task_key]
                data["task"] = task_desc.decode("utf-8") if isinstance(task_desc, bytes) else str(task_desc)
            else:
                data["task"] = ""

            image_storage = config.get("image_storage", "hdf5")
            if image_storage == "hdf5":
                data["images"] = load_images_from_hdf5(ep_file, config)
            elif image_storage == "external_mp4":
                data["images"] = load_images_from_mp4(ep_path, config)
            else:
                raise ValueError(f"Invalid 'image_storage' type in config: {image_storage}")

            if camera_crops:
                for cam_name, imgs_array in data["images"].items():
                    if cam_name in camera_crops:
                        x, y, w, h = camera_crops[cam_name]
                        cropped_imgs = [img[y : y + h, x : x + w] for img in imgs_array]
                        data["images"][cam_name] = np.array(cropped_imgs)

            if target_res := config.get("target_camera_resolution"):
                target_size = (target_res[1], target_res[0])
                for cam_name, imgs_array in data["images"].items():
                    resized_imgs = [
                        cv2.resize(img, target_size, interpolation=cv2.INTER_AREA) for img in imgs_array
                    ]
                    data["images"][cam_name] = np.array(resized_imgs)

            all_streams_to_check = list(data_streams.values()) + list(data.get("images", {}).values())
            
            non_empty_streams = [s for s in all_streams_to_check if len(s) > 0]
            if not non_empty_streams:
                LOGGER.warning(f"No data streams found in {ep_path}. Skipping episode.")
                return None
            min_len = min(len(s) for s in non_empty_streams)

            for key, stream in data_streams.items():
                if len(stream) > min_len:
                    LOGGER.warning(f"Truncating stream '{key}' from {len(stream)} to {min_len} in {ep_path}")
                    data_streams[key] = stream[:min_len]

            for cam_name, imgs in data.get("images", {}).items():
                if len(imgs) > min_len:
                    LOGGER.warning(f"Truncating images for '{cam_name}' from {len(imgs)} to {min_len} in {ep_path}")
                    data["images"][cam_name] = imgs[:min_len]
            
            data.update(data_streams)

            num_frames = min_len
            dones = np.zeros(num_frames, dtype=bool)

            if proportional_reward:
                rewards = np.linspace(0.0, 1.0, num_frames, dtype=np.float32) if num_frames > 1 else np.array([1.0], dtype=np.float32) if num_frames == 1 else np.array([], dtype=np.float32)
            else:
                rewards = np.full(num_frames, flat_reward, dtype=np.float32)

            if "reward_labeling" in config and config["reward_labeling"].get("positive_demonstration", False):
                steps_before_success = config["reward_labeling"].get("steps_before_success")
                num_pos_frames = min(steps_before_success, num_frames)
                rewards[num_frames - num_pos_frames :] = 1.0

            if num_frames > 0:
                dones[-1] = True

            data["rewards"] = [torch.tensor([r]) for r in rewards]
            data["dones"] = [torch.tensor([d]) for d in dones]

    except Exception as e:
        LOGGER.error(f"Failed to load episode from {ep_path}: {e}. Skipping episode.")
        return None

    return data


# --- Data Population ---


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: List[Path],
    config: Dict[str, Any],
    flat_reward: float,
    proportional_reward: bool,
    camera_crops: Dict[str, Tuple[int, int, int, int]],
):
    """Iterates through raw HDF5 files and adds their data to the LeRobot dataset."""
    for ep_path in tqdm.tqdm(hdf5_files, desc="Processing episodes"):
        LOGGER.info(f"Processing {ep_path}")
        raw_data = load_raw_episode_data(
            ep_path, config, flat_reward, proportional_reward, camera_crops
        )
        if raw_data is None:
            continue

        num_frames = len(raw_data["state"])
        if num_frames == 0:
            LOGGER.warning(f"Skipping episode {ep_path} as it has 0 frames after processing.")
            continue
            
        task_description = raw_data.get("task", "")

        for i in range(num_frames):
            frame = {
                "next.reward": raw_data["rewards"][i],
                "next.done": raw_data["dones"][i],
                "task": task_description,
            }
            
            if "state" in raw_data:
                frame["observation.state"] = raw_data["state"][i]
            if "velocity" in raw_data:
                frame["observation.velocity"] = raw_data["velocity"][i]
            if "effort" in raw_data:
                frame["observation.effort"] = raw_data["effort"][i]
            if "action" in raw_data:
                frame["action"] = raw_data["action"][i]
                 
            for camera_name in config["camera_names"]:
                frame_key = f"observation.images.{camera_name}"
                if camera_name in raw_data.get("images", {}):
                    frame[frame_key] = raw_data["images"][camera_name][i]

            dataset.add_frame(frame=frame)

        dataset.save_episode()


# --- Main Execution ---


def convert_raw_to_lerobot(
    raw_dir: Path,
    repo_id: str,
    config_path: Path,
    *,
    camera_crops: Optional[List[str]] = None,
    flat_reward: float = 0.0,
    proportional_reward: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    """
    Main function to perform the dataset conversion.

    Args:
        raw_dir: Path to the directory containing raw HDF5 episode files.
        repo_id: The name for the new local LeRobot dataset.
        config_path: Path to the YAML configuration file for data mapping.
        camera_crops: Define cropping regions for cameras.
            Format: `camera_name1:x,y,w,h camera_name2:x,y,w,h`.
            Example: `--camera-crops front:10,20,224,224`.
        flat_reward: A constant reward value to assign to all non-terminal steps.
                     The final successful steps will still receive a reward of 1.0. Defaults to 0.0.
        proportional_reward: If True, assign a reward that ramps from 0.0 to 1.0 over the episode.
                             This is mutually exclusive with `flat_reward`.
        mode: The storage format for images ('video' or 'image').
        dataset_config: Advanced configuration for the dataset writer.
    """
    setup_logging()
    
    if flat_reward != 0.0 and proportional_reward:
        raise ValueError("Cannot use --flat-reward and --proportional-reward simultaneously. Please choose one.")

    parsed_camera_crops = {}
    if camera_crops:
        LOGGER.info(f"Parsing camera crops: {camera_crops}")
        for crop_arg in camera_crops:
            parts = crop_arg.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid camera crop format: {crop_arg}")
            cam_name, crop_str = parts
            try:
                x, y, w, h = map(int, crop_str.split(","))
                parsed_camera_crops[cam_name] = (x, y, w, h)
            except ValueError:
                raise ValueError(f"Invalid crop values for {cam_name}: {crop_str}")
        LOGGER.info(f"Successfully parsed camera crops: {parsed_camera_crops}")

    config = load_config(config_path)

    hdf5_files = sorted(raw_dir.glob("**/main.rmb.hdf5"))
    if not hdf5_files:
        hdf5_files = sorted(raw_dir.glob("**/*.hdf5"))

    if not hdf5_files:
        raise FileNotFoundError(f"No '.hdf5' or 'main.rmb.hdf5' files found in {raw_dir}")

    LOGGER.info(f"Found {len(hdf5_files)} episode files.")

    dataset, original_shapes = create_empty_dataset(
        repo_id, config, hdf5_files, mode=mode, dataset_config=dataset_config
    )
    populate_dataset(
        dataset,
        hdf5_files,
        config,
        flat_reward,
        proportional_reward,
        parsed_camera_crops,
    )

    LOGGER.info("Finalizing dataset...")
    stats_path = dataset.root / "meta" / "stats.json"
    LOGGER.info(f"Statistics have been saved to {stats_path}")

    num_episodes = 0
    num_frames = 0
    if stats_path.exists():
        with stats_path.open("r") as f:
            stats = json.load(f)
            num_episodes = stats.get("episodes", 0)
            num_frames = stats.get("steps", 0)

        LOGGER.info("--- Dataset Statistics Summary ---")
        LOGGER.info(f"{'Total Episodes':<25} | {num_episodes}")
        LOGGER.info(f"{'Total Frames (Steps)':<25} | {num_frames}")
        LOGGER.info("-" * 40)

    LOGGER.info("--- Dimension Verification Summary ---")
    header = f"{'Feature':<25} | {'Original (from HDF5)':<25} | {'Final (in LeRobot Dataset)':<25}"
    LOGGER.info(header)
    LOGGER.info(f"{'-'*26}|{'-'*27}|{'-'*27}")
    
    for key in ["state", "velocity", "effort", "action"]:
        final_key_name = f"observation.{key}" if key != "action" else "action"
        if final_key_name in dataset.features:
            original_shape_str = str(original_shapes.get(key, 'N/A'))
            final_shape_str = str(dataset.features[final_key_name]["shape"])
            LOGGER.info(f"{key.capitalize()+' Dimension':<25} | {original_shape_str:<25} | {final_shape_str:<25}")
    LOGGER.info("-" * 81)


    LOGGER.info("✅ Conversion complete!")
    LOGGER.info(f"Your local LeRobot v3.0 dataset is ready at: {dataset.root}")


if __name__ == "__main__":
    tyro.cli(convert_raw_to_lerobot)

