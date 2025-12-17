#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generic script to convert HDF5-based data (including .rmb format) to the LeRobot dataset v2.0 format.

This script uses a YAML configuration file to map the raw data structure to the LeRobot format,
supporting images stored either inside the HDF5 file or as external MP4 videos.

Key Features:
- Flexible data mapping via a YAML configuration file.
- Support for images embedded in HDF5 or as external MP4 videos.
- Automatic calculation of dataset statistics for normalization.
- Optional upload to the Hugging Face Hub.

Example usage for .rmb format:
    uv run python /path/to/this/script.py \
        --raw-dir /path/to/raw_rmb_data \
        --repo-id <org>/<my-dataset-name> \
        --config-path /path/to/your/config.yaml \
        --push-to-hub
"""

import dataclasses
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import cv2
import h5py
import numpy as np
import torch
import tqdm
import tyro
import videoio
import yaml
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
    # Basic validation can be added here if needed
    return config


# --- Dataset Creation and Schema ---

def create_empty_dataset(
    repo_id: str,
    config: Dict[str, Any],
    hdf5_files: List[Path],
    mode: Literal["video", "image"],
    dataset_config: DatasetConfig,
) -> LeRobotDataset:
    """
    Creates an empty LeRobotDataset with a schema defined by the config file.

    Args:
        repo_id: The Hugging Face Hub repository ID for the dataset.
        config: The loaded YAML configuration dictionary.
        hdf5_files: A list of HDF5 file paths to infer data shapes from.
        mode: The storage mode for images ('video' or 'image').
        dataset_config: The dataset writing configuration.

    Returns:
        An initialized but empty LeRobotDataset object.
    """
    features = {}
    data_mapping = config["data_mapping"]
    motor_names = config.get("motor_names", [])
    height, width = config["camera_resolution"]

    # Infer data shapes from the first HDF5 file for robustness
    with h5py.File(hdf5_files[0], "r") as h5_file:
        state_shape = h5_file[data_mapping["state"]].shape[1:]
        action_shape = h5_file[data_mapping["action"]].shape[1:]
        velocity_shape = h5_file[data_mapping.get("velocity", "")].shape[1:] if data_mapping.get("velocity") in h5_file else None
        effort_shape = h5_file[data_mapping.get("effort", "")].shape[1:] if data_mapping.get("effort") in h5_file else None

    # Define state, action, and optional features
    features["observation.state"] = {"dtype": "float32", "shape": state_shape}
    features["action"] = {"dtype": "float32", "shape": action_shape}
    if velocity_shape:
        features["observation.velocity"] = {"dtype": "float32", "shape": velocity_shape}
    if effort_shape:
        features["observation.effort"] = {"dtype": "float32", "shape": effort_shape}

    # Assign names to dimensions if provided and dimensions match
    for key, shape in [("state", state_shape), ("action", action_shape), ("velocity", velocity_shape), ("effort", effort_shape)]:
        if shape and shape[0] == len(motor_names):
            feature_key = f"observation.{key}" if key != "action" else "action"
            features[feature_key]["names"] = motor_names

    # Define image features for each camera
    for cam_name in config["camera_names"]:
        features[f"observation.images.{cam_name}"] = {
            "dtype": mode,
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        }

    # Explicitly define the timestamp feature to prevent loading errors
    features["timestamp"] = {"dtype": "float64", "shape": ()}

    # Clean up any existing dataset directory
    dataset_path = HF_LEROBOT_HOME / repo_id
    if dataset_path.exists():
        LOGGER.warning(f"Deleting existing dataset at {dataset_path}")
        shutil.rmtree(dataset_path)
    LOGGER.info(f"Creating new dataset at: {dataset_path}")

    return LeRobotDataset.create(
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


# --- Data Loading ---

def load_images_from_hdf5(ep_file: h5py.File, config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Loads and decodes images from datasets within an HDF5 file."""
    imgs_per_cam = {}
    image_key_template = config["data_mapping"]["image_template"]

    for camera_name in config["camera_names"]:
        image_key = image_key_template.format(camera_name=camera_name)
        image_dataset = ep_file[image_key]

        if image_dataset.ndim == 4 and image_dataset.dtype == np.uint8:
            # Already in (N, H, W, C) or similar format
            imgs_array = image_dataset[:]
        else:
            # Assumes JPEG/PNG encoded bytes, decode them
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

        # Corrected line below - removed the 'verbosity' argument
        imgs_array = videoio.videoread(str(video_path))
        imgs_per_cam[camera_name] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(ep_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """Loads all data for a single episode from HDF5 and associated videos."""
    data = {}
    data_mapping = config["data_mapping"]

    with h5py.File(ep_path, "r") as ep_file:
        # Load each frame as a separate tensor in a list
        data["state"] = [torch.from_numpy(frame).float() for frame in ep_file[data_mapping["state"]][:]]
        data["action"] = [torch.from_numpy(frame).float() for frame in ep_file[data_mapping["action"]][:]]

        if "velocity" in data_mapping and data_mapping["velocity"] in ep_file:
            data["velocity"] = [torch.from_numpy(frame).float() for frame in ep_file[data_mapping["velocity"]][:]]
        if "effort" in data_mapping and data_mapping["effort"] in ep_file:
            data["effort"] = [torch.from_numpy(frame).float() for frame in ep_file[data_mapping["effort"]][:]]
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
            # Pass ep_path to find sibling video files
            data["images"] = load_images_from_mp4(ep_path, config)
        else:
            raise ValueError(f"Invalid 'image_storage' type in config: {image_storage}")

    return data


# --- Data Population and Statistics ---

def populate_dataset(dataset: LeRobotDataset, hdf5_files: List[Path], config: Dict[str, Any]):
    """Iterates through raw HDF5 files and adds their data to the LeRobot dataset."""
    for ep_path in tqdm.tqdm(hdf5_files, desc="Processing episodes"):
        raw_data = load_raw_episode_data(ep_path, config)
        num_frames = len(raw_data["state"])
        task_description = raw_data["task"]

        for i in range(num_frames):
            frame = {
                "observation.state": raw_data["state"][i],
                "action": raw_data["action"][i],
            }
            for camera_name, img_array in raw_data["images"].items():
                frame[f"observation.images.{camera_name}"] = img_array[i]

            if "velocity" in raw_data:
                frame["observation.velocity"] = raw_data["velocity"][i]
            if "effort" in raw_data:
                frame["observation.effort"] = raw_data["effort"][i]

            dataset.add_frame(frame=frame, task=task_description)

        dataset.save_episode()


def calculate_and_save_stats(dataset: LeRobotDataset):
    """Calculates and saves normalization statistics for the dataset."""
    LOGGER.info("Calculating dataset statistics...")
    # LeRobotDataset has a built-in method for this
    dataset.compute_stats()
    stats_path = dataset.root / "meta" / "stats.json"
    LOGGER.info(f"Statistics saved to {stats_path}")


# --- Main Execution ---

def convert_raw_to_lerobot(
    raw_dir: Path,
    repo_id: str,
    config_path: Path,
    *,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    """
    Main function to perform the dataset conversion.

    Args:
        raw_dir: Directory containing the raw HDF5/RMB episode files.
        repo_id: Hugging Face Hub repository ID (e.g., 'my-org/my-dataset').
        config_path: Path to the YAML configuration file.
        push_to_hub: If True, push the completed dataset to the Hub.
        mode: Storage format for images, either 'video' (MP4) or 'image' (PNG).
        dataset_config: Advanced configuration for the dataset writer.
    """
    setup_logging()
    config = load_config(config_path)

    # Find episode files, prioritizing the specific .rmb structure
    hdf5_files = sorted(raw_dir.glob("**/main.rmb.hdf5"))
    if not hdf5_files:
        hdf5_files = sorted(raw_dir.glob("**/*.hdf5"))

    if not hdf5_files:
        raise FileNotFoundError(f"No '.hdf5' or 'main.rmb.hdf5' files found in {raw_dir}")

    LOGGER.info(f"Found {len(hdf5_files)} episode files.")

    dataset = create_empty_dataset(repo_id, config, hdf5_files, mode=mode, dataset_config=dataset_config)
    populate_dataset(dataset, hdf5_files, config)

    # LeRobotDataset versions >= v2.1 handle stats calculation internally after `save_episode`.
    # For robustness, we can call it explicitly if needed, but it's often redundant.
    # calculate_and_save_stats(dataset)

    if push_to_hub:
        LOGGER.info(f"Pushing dataset to Hugging Face Hub: {repo_id}")
        dataset.push_to_hub()

    LOGGER.info("✅ Conversion complete!")


if __name__ == "__main__":
    tyro.cli(convert_raw_to_lerobot)