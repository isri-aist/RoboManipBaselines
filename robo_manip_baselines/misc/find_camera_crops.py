#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
An interactive script to find camera cropping parameters for the `convert_to_lerobot.py` script.

This tool loads the first frame from the first episode for each camera specified. It then
opens an OpenCV window and operates in one of two modes:

1. Two-Point Click Mode (Default):
- Click the TOP-LEFT corner of your desired crop area.
- Click the BOTTOM-RIGHT corner to define the rectangle.

2. Drag Mode (when --target-size is used):
- A rectangle of the specified size appears on the image.
- Click and drag this rectangle to the desired position.

In both modes:
- Press 'r' to reset the selection.
- Press 'q' to confirm the crop and move to the next camera.

After all cameras are processed, it prints a ready-to-use command-line argument string
that you can copy and paste into your `convert_to_lerobot.py` command.

Example usage (Click Mode):
    python /path/to/this/script.py \\
        --raw-dir /path/to/raw_data

Example usage (Drag Mode):
    python /path/to/this/script.py \\
        --raw-dir /path/to/raw_data \\
        --target-size 128,128
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import tyro
import videoio
import yaml

# --- Globals for OpenCV mouse callbacks ---
# For two-point click mode
points: List[Tuple[int, int]] = []
# For drag mode
roi_top_left: List[int] = [0, 0]
dragging: bool = False

LOGGER = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    """Initializes the logger for clear console output."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def mouse_callback_click(event, x, y, flags, param):
    """OpenCV mouse callback for the two-point click mode."""
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))
            LOGGER.info(f"Point {len(points)} recorded: ({x}, {y})")
        else:
            LOGGER.warning("Already have 2 points. Press 'r' to reset.")

# --- Configuration Functions ---

def load_config(config_path: Path) -> Dict[str, Any]:
    """Loads and validates the YAML configuration file."""
    LOGGER.info(f"Loading configuration from {config_path}")
    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    return config


def get_config_interactively() -> Dict[str, Any]:
    """
    Prompts the user for necessary parameters when no config file is provided.
    Offers default values that can be accepted by pressing Enter.
    """
    print("\n" + "-" * 50)
    print("🚀 Interactive Configuration Setup")
    print("Press Enter to accept the default value shown in [brackets].")
    print("-" * 50)

    default_camera_names = "front,hand,side"
    prompt = f"▶ Enter camera names, separated by commas [{default_camera_names}]: "
    camera_names_str = input(prompt) or default_camera_names
    camera_names = [name.strip() for name in camera_names_str.split(",") if name.strip()]
    if not camera_names:
        raise ValueError("At least one camera name must be provided.")

    default_storage = "external_mp4"
    image_storage = ""
    while image_storage not in ["hdf5", "external_mp4"]:
        prompt = f"▶ Enter image storage type ('hdf5' or 'external_mp4') [{default_storage}]: "
        user_input = input(prompt).lower().strip()
        image_storage = user_input or default_storage
        if image_storage not in ["hdf5", "external_mp4"]:
            print("Invalid input. Please choose 'hdf5' or 'external_mp4'.")

    print("\n▶ Enter the image template string.")
    print("  Use '{camera_name}' as a placeholder for the camera name.")

    if image_storage == "hdf5":
        default_template = "/observations/images/{camera_name}"
        print(f"  Example for HDF5: {default_template}")
    else:
        default_template = "{camera_name}_rgb_image.rmb.mp4"
        print(f"  Example for MP4: {default_template}")

    prompt = f"▶ Enter image template [{default_template}]: "
    image_template = input(prompt).strip() or default_template

    config = {
        "camera_names": camera_names,
        "image_storage": image_storage,
        "data_mapping": {"image_template": image_template},
    }
    print("-" * 50)
    LOGGER.info(f"Generated config: {config}")
    return config


# --- Data Loading Functions ---


def load_first_image_from_hdf5(
    ep_file: h5py.File, config: Dict[str, Any], camera_name: str
) -> Optional[np.ndarray]:
    """Loads and decodes the first image for a specific camera from an HDF5 file."""
    image_key_template = config["data_mapping"]["image_template"]
    image_key = image_key_template.format(camera_name=camera_name)

    if image_key not in ep_file:
        LOGGER.warning(f"Image key '{image_key}' not found. Skipping camera '{camera_name}'.")
        return None

    image_dataset = ep_file[image_key]
    first_frame_data = image_dataset[0]

    img = (
        first_frame_data
        if image_dataset.ndim == 4 and image_dataset.dtype == np.uint8
        else cv2.imdecode(first_frame_data, cv2.IMREAD_COLOR)
    )
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_first_image_from_mp4(
    ep_path: Path, config: Dict[str, Any], camera_name: str
) -> Optional[np.ndarray]:
    """Loads the first image for a specific camera from an external MP4 file."""
    video_dir = ep_path.parent
    image_key_template = config["data_mapping"]["image_template"]
    video_filename = image_key_template.format(camera_name=camera_name)
    video_path = video_dir / video_filename

    if not video_path.exists():
        LOGGER.warning(f"Video file not found at {video_path}, skipping camera {camera_name}")
        return None

    imgs_array = videoio.videoread(str(video_path))
    return imgs_array[0] if len(imgs_array) > 0 else None


# --- Main Execution ---


def find_crop_parameters(
    raw_dir: Path,
    config_path: Optional[Path] = None,
    target_size: Optional[str] = None,
):
    """
    Main function to run the interactive cropping tool.

    Args:
        raw_dir: Path to the directory containing raw HDF5 episode files.
        config_path: Optional path to the YAML config file. If not provided, the script
                     will ask for parameters interactively.
        target_size: Optional fixed window size for cropping, in "WIDTH,HEIGHT" format.
                     If provided, enables drag mode.
    """
    global points, roi_top_left, dragging
    setup_logging()

    parsed_target_size: Optional[Tuple[int, int]] = None
    if target_size:
        try:
            w_str, h_str = target_size.split(",")
            parsed_target_size = (int(w_str), int(h_str))
            LOGGER.info(f"Using fixed target window size: {parsed_target_size}")
        except ValueError:
            raise ValueError("Invalid format for --target-size. Use WIDTH,HEIGHT (e.g., '128,128').")

    # FIX: Reverted to using the dedicated load_config function which correctly uses pyyaml.
    config = load_config(config_path) if config_path else get_config_interactively()

    hdf5_files = sorted(raw_dir.glob("**/main.rmb.hdf5")) or sorted(raw_dir.glob("**/*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No '.hdf5' or 'main.rmb.hdf5' files found in {raw_dir}")

    first_episode_path = hdf5_files[0]
    LOGGER.info(f"Using first episode for images: {first_episode_path}")

    crop_results = {}
    image_storage = config.get("image_storage", "hdf5")

    with h5py.File(first_episode_path, "r") as ep_file:
        for camera_name in config["camera_names"]:
            print("\n" + "=" * 50)
            print(f"Processing camera: {camera_name}")
            print("=" * 50)

            if image_storage == "hdf5":
                frame = load_first_image_from_hdf5(ep_file, config, camera_name)
            elif image_storage == "external_mp4":
                frame = load_first_image_from_mp4(first_episode_path, config, camera_name)
            else:
                raise ValueError(f"Invalid 'image_storage' type in config: {image_storage}")

            if frame is None:
                continue

            window_name = f"ROI Finder: {camera_name} | (r)eset | (q)uit to confirm"
            cv2.namedWindow(window_name)

            # --- MODE 1: Drag a fixed-size window ---
            if parsed_target_size:
                w, h = parsed_target_size
                frame_h, frame_w, _ = frame.shape
                roi_top_left = [int(frame_w / 2 - w / 2), int(frame_h / 2 - h / 2)]
                dragging = False

                def mouse_callback_drag(event, x, y, flags, param):
                    global roi_top_left, dragging
                    if event == cv2.EVENT_LBUTTONDOWN:
                        dragging = True
                    elif event == cv2.EVENT_MOUSEMOVE and dragging:
                        new_x = x - w // 2
                        new_y = y - h // 2
                        roi_top_left[0] = max(0, min(new_x, frame_w - w))
                        roi_top_left[1] = max(0, min(new_y, frame_h - h))
                    elif event == cv2.EVENT_LBUTTONUP:
                        dragging = False

                cv2.setMouseCallback(window_name, mouse_callback_drag)
                print("1. Click and drag the rectangle to the desired position.")
                print("2. Press 'q' to save and continue, or 'r' to reset to center.")

                while True:
                    display_image = frame.copy()
                    tl_point = tuple(roi_top_left)
                    br_point = (tl_point[0] + w, tl_point[1] + h)
                    cv2.rectangle(display_image, tl_point, br_point, (0, 255, 0), 2)
                    cv2.imshow(window_name, cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR))

                    key = cv2.waitKey(20) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("r"):
                        roi_top_left = [int(frame_w / 2 - w / 2), int(frame_h / 2 - h / 2)]
                        print("Rectangle reset to center.")

                crop_results[camera_name] = (roi_top_left[0], roi_top_left[1], w, h)

            # --- MODE 2: Click two points ---
            else:
                points = []
                cv2.setMouseCallback(window_name, mouse_callback_click)
                print("1. Click the TOP-LEFT corner of your desired region.")
                print("2. Click the BOTTOM-RIGHT corner.")
                print("3. Press 'q' to save and continue, or 'r' to reset.")

                while True:
                    display_image = frame.copy()
                    if len(points) > 0:
                        cv2.circle(display_image, points[0], 5, (0, 255, 0), -1)
                    if len(points) == 2:
                        cv2.circle(display_image, points[1], 5, (0, 0, 255), -1)
                        cv2.rectangle(display_image, points[0], points[1], (0, 255, 0), 2)

                    cv2.imshow(window_name, cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR))

                    key = cv2.waitKey(20) & 0xFF
                    if key == ord("q"):
                        if len(points) == 2:
                            break
                        else:
                            print("Please select 2 points before quitting.")
                    elif key == ord("r"):
                        points = []
                        print("Points reset for this camera.")

                x1, y1 = points[0]
                x2, y2 = points[1]
                crop_results[camera_name] = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

            cv2.destroyWindow(window_name)
            x, y, w, h = crop_results[camera_name]
            print(f"✅ ROI for {camera_name} saved: (x={x}, y={y}, w={w}, h={h})")

    if crop_results:
        print("\n" + "=" * 70)
        print("✅ Interactive cropping complete!")
        print("Copy the following argument to use with 'convert_to_lerobot.py':")
        print("=" * 70)
        crop_strings = [f"{name}:{x},{y},{w},{h}" for name, (x, y, w, h) in crop_results.items()]
        print(f"--camera_crops {' '.join(crop_strings)}")
        print("=" * 70)
    else:
        print("\nNo crops were defined.")


if __name__ == "__main__":
    tyro.cli(find_crop_parameters)