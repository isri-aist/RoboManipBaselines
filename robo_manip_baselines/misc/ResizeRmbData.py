#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to resize a robomanipbaseline (or other HDF5-based) dataset and save it
in the native RoboManipBaseline (RMB) format.

This script takes a directory of raw HDF5 episodes, a target resolution, and 
optional cropping parameters to produce a new, resized dataset in a specified 
output directory. It automatically detects the data structure from the source files.

Key Features (Optimized Version):
- Resizes all camera images to a specified target resolution.
- Uses a single-pass loop for cropping and resizing to improve memory efficiency.
- Saves the output dataset in the original RoboManipBaseline format (`.rmb` or `.hdf5`).
- Uses multiprocessing for speed, with robust error handling for worker crashes.

Example usage (with crops):
    python ResizeRmbData.py \
        --input-dir /path/to/original_dataset \
        --output-dir /path/to/resized_dataset \
        --target-resolution 64,64 \
        --output-format rmb \
        --camera-crops "front:135,92,256,256" "side:192,112,256,256"
        
Example usage (processing a range of subfolders, e.g., '000' to '004'):
    python ResizeRmbData.py \
        --input-dir /path/to/parent_of_episodes \
        --output-dir /path/to/resized_dataset \
        --target-resolution 64,64 \
        --episode-range "000-004"

Example usage (for debugging the BrokenProcessPool error, using --max-workers 1):
    python ResizeRmbData.py \
        --input-dir ./dataset/RealUR10eDemo \
        --output-dir ./dataset/RealUR10eDemo_64 \
        --target-resolution 64,64 \
        --output-format rmb \
        --max-workers 1
"""

import concurrent.futures
import logging
import os
import re # Added for file name parsing/filtering
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import cv2
import h5py
import numpy as np
import tqdm
import tyro
import videoio
# Assuming these imports work and the library is installed
from robo_manip_baselines.common import (
    DataManager,
    DataKey,
    RmbData,
    find_rmb_files,
)

# --- Setup logging ---
LOGGER = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    """Initializes the logger for clear console output."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# --- Data Loading and Processing ---


def load_and_process_episode(
    ep_path: str,  # Expect a string path to be compatible with multiprocessing
    target_resolution: Tuple[int, int],
    camera_crops: Dict[str, Tuple[int, int, int, int]],
    flat_reward: float,
    proportional_reward: bool,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Loads, processes, and resizes all data for a single episode.
    Returns the original string path and the processed data.
    """
    processed_data = {}
    image_streams = {}
    # target_resolution is (height, width), target_size for cv2 is (width, height)
    target_size = (target_resolution[1], target_resolution[0]) 
    
    # 🐛 DEBUGGING PRINT/LOG: Indicate which episode is starting processing
    print(f"[DEBUG] Starting processing episode: {ep_path}")
    
    try:
        # RmbData expects a string path.
        with RmbData(ep_path, "r") as rmb_data:
            for key in rmb_data.keys():
                # OPTIMIZATION: Use full slicing [:] to read the data stream 
                # in a single bulk operation, which is much faster.
                stream = rmb_data[key][:] 
                
                if DataKey.is_rgb_image_key(key):
                    # Ensure stream is a list for iteration if it came from RmbVideo object
                    if not isinstance(stream, list):
                        stream = list(stream)

                    cam_name = DataKey.get_camera_name(key)
                    resized_imgs = []
                    
                    if stream:
                        print(f"[DEBUG] Processing RGB stream '{key}' with {len(stream)} frames.")

                    if cam_name in camera_crops:
                        x, y, w, h = camera_crops[cam_name]
                        # Single-pass loop: crop and resize frame-by-frame
                        for img in stream:
                            # Ensure image is not empty before processing
                            if img is None or img.size == 0: continue
                            # 1. Crop
                            cropped_img = img[y : y + h, x : x + w]
                            # 2. Resize (using INTER_AREA for downsampling quality)
                            resized_imgs.append(cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_AREA))
                    else:
                        # Only resize if no crop is specified
                        for img in stream:
                            if img is None or img.size == 0: continue
                            resized_imgs.append(cv2.resize(img, target_size, interpolation=cv2.INTER_AREA))
                            
                    image_streams[key] = resized_imgs
                        
                elif DataKey.is_depth_image_key(key):
                    # Ensure stream is a list for iteration if it came from RmbVideo object
                    if not isinstance(stream, list):
                        stream = list(stream)

                    if stream:
                        print(f"[DEBUG] Processing Depth stream '{key}' with {len(stream)} frames.")
                        
                    # Resize depth images using INTER_NEAREST (appropriate for discrete depth values)
                    resized_depths = []
                    for img in stream:
                        if img is None or img.size == 0: continue
                        resized_depths.append(cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST))
                    image_streams[key] = resized_depths
                else:
                    # For non-image data, ensure it's a list for later manipulation
                    processed_data[key] = list(stream)
            
            # --- Stream synchronization and reward generation ---
            all_streams = list(processed_data.values()) + list(image_streams.values())
            if not (non_empty_streams := [s for s in all_streams if len(s) > 0]):
                LOGGER.warning(f"No data streams found in {ep_path}. Skipping.")
                return None
            min_len = min(len(s) for s in non_empty_streams)

            # Trim all streams to the minimum length
            for key, stream in processed_data.items():
                if len(stream) > min_len:
                    processed_data[key] = stream[:min_len]
            for key, stream in image_streams.items():
                if len(stream) > min_len:
                    image_streams[key] = stream[:min_len]
            
            processed_data.update(image_streams)
            num_frames = min_len

            if proportional_reward or flat_reward != 0.0:
                if proportional_reward:
                    rewards = np.linspace(0.0, 1.0, num_frames, dtype=np.float32) if num_frames > 1 else np.array([1.0], dtype=np.float32)
                else:
                    rewards = np.full(num_frames, flat_reward, dtype=np.float32)
                processed_data[DataKey.REWARD] = list(rewards)

    except Exception as e:
        # Crucial for catching non-crash exceptions in the worker process
        LOGGER.error(f"Failed to process episode {ep_path}: {e}", exc_info=True)
        return None
    
    # 🐛 DEBUGGING PRINT/LOG: Indicate successful completion
    print(f"[DEBUG] Successfully processed episode: {ep_path}")
    return ep_path, processed_data


# --- Main Execution ---


def resize_rmb_dataset(
    input_dir: Path,
    output_dir: Path,
    target_resolution: str,
    *,
    episode_range: Optional[str] = None, # NEW ARGUMENT
    output_format: Literal["rmb", "hdf5"] = "rmb",
    camera_crops: Optional[List[str]] = None,
    flat_reward: float = 0.0,
    proportional_reward: bool = False,
    max_workers: Optional[int] = None, 
):
    """
    Main function to resize a RoboManipBaseline dataset.

    Args:
        input_dir: Path to the directory with raw RMB episodes. This is also the 
                   parent directory if 'episode_range' is used.
        output_dir: Path to the directory where resized episodes will be saved.
        target_resolution: New image resolution as 'height,width' (e.g., '224,224').
        episode_range: Optional range of episode folder numbers to process, 
                       e.g., '000-004'. If provided, input_dir is treated as the 
                       parent folder.
        output_format: The output format, either 'rmb' or 'hdf5'.
        camera_crops: Crop regions as 'camera_name:x,y,w,h'. Can be specified multiple times.
        flat_reward: Constant reward for non-terminal steps.
        proportional_reward: If True, reward ramps from 0.0 to 1.0.
        max_workers: The maximum number of processes to use. Defaults to 4 or CPU count.
    """
    setup_logging()
    if flat_reward != 0.0 and proportional_reward:
        raise ValueError("Cannot use --flat-reward and --proportional-reward simultaneously.")

    # 1. PARSE CAMERA CROPS
    parsed_camera_crops = {}
    if camera_crops:
        for crop_arg in camera_crops:
            parts = crop_arg.split(":")
            if len(parts) != 2: raise ValueError(f"Invalid crop format: {crop_arg}")
            cam_name, crop_str = parts
            try:
                x, y, w, h = map(int, crop_str.split(","))
                parsed_camera_crops[cam_name] = (x, y, w, h)
            except ValueError:
                raise ValueError(f"Invalid crop values for {cam_name}: {crop_str}")
        LOGGER.info(f"Parsed camera crops: {parsed_camera_crops}")

    # 2. PARSE TARGET RESOLUTION
    try:
        h, w = map(int, target_resolution.split(","))
        parsed_target_resolution = (h, w)
        LOGGER.info(f"Setting target resolution to {h}x{w}")
    except ValueError:
        raise ValueError(f"Invalid --target-resolution: '{target_resolution}'. Expected 'height,width'.")

    # 3. FIND INPUT FILES (MODIFIED TO HANDLE RANGE)
    rmb_files = []
    
    if episode_range:
        # Parse episode range (e.g., "000-004")
        try:
            start_str, end_str = episode_range.split("-")
            start = int(start_str)
            end = int(end_str)
            padding = len(start_str) # Get padding length (e.g., 3 for '000')
            if start > end:
                raise ValueError("Start of range must be less than or equal to end of range.")

            # Check for the expected folder structure (e.g., input_dir/000)
            ep_dir_check = input_dir / f"{start:0{padding}d}"
            
            if ep_dir_check.is_dir():
                # --- Case 1: Folder-based structure (Original logic) ---
                LOGGER.info("Detected folder-based episode structure. Processing range by folder.")
                for i in range(start, end + 1):
                    folder_name = f"{i:0{padding}d}" 
                    ep_dir = input_dir / folder_name
                    
                    if ep_dir.is_dir():
                        LOGGER.info(f"Searching for episodes in folder {ep_dir}")
                        rmb_files.extend(find_rmb_files(str(ep_dir)))
                    else:
                        LOGGER.warning(f"Episode directory {ep_dir} not found. Skipping.")

            else:
                # --- Case 2: Flat structure (Fix for user's immediate need) ---
                # This handles cases where files are named like 'episode_000.rmb' directly in input_dir
                LOGGER.info(f"Numbered subdirectories not found. Scanning flat directory {input_dir} for files in range {episode_range}.")
                
                all_rmb_files = find_rmb_files(str(input_dir))
                target_indices = {f"{i:0{padding}d}" for i in range(start, end + 1)}
                
                # Regex to find the numeric index right before the .rmb or .hdf5 extension.
                # It looks for a sequence of 'padding' digits (e.g., '000') before the extension.
                index_pattern = re.compile(rf'([0-9]{{{padding}}})\.(rmb|hdf5)$', re.IGNORECASE)
                
                for f in all_rmb_files:
                    path_stem = Path(f).stem
                    match = index_pattern.search(f)
                    if match:
                        index_str = match.group(1)
                        if index_str in target_indices:
                            rmb_files.append(f)
                        # Files outside the range are automatically skipped

        except ValueError as e:
            raise ValueError(f"Invalid --episode-range format: '{episode_range}'. Expected 'start-end' (e.g., '000-004'). Error: {e}")

    else:
        # Original behavior: scan the entire input_dir
        rmb_files = find_rmb_files(str(input_dir))
        
    if not rmb_files:
        raise FileNotFoundError(f"No .rmb or .hdf5 files found in the specified location(s).")
    
    LOGGER.info(f"Found {len(rmb_files)} episode files to process.")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_manager = DataManager(env=None)
    
    # Check if max_workers is set too high for I/O bound tasks
    if max_workers is None:
        cpu_count = os.cpu_count() or 4
        # Lower the default recommendation for stability
        default_workers = min(cpu_count, 4)
        if cpu_count > default_workers:
            LOGGER.warning(f"Defaulting to {default_workers} workers to prevent disk/memory saturation. For I/O-heavy tasks, you may get better performance by explicitly setting --max-workers to a lower value.")
        max_workers = default_workers

    # --- Parallel Processing ---
    # The rest of the function remains the same, operating on the aggregated rmb_files list.
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all processing jobs to the executor
        futures = [
            executor.submit(
                load_and_process_episode,
                ep_path, # Pass the string path directly
                parsed_target_resolution, # Now correctly defined
                parsed_camera_crops,
                flat_reward,
                proportional_reward,
            )
            for ep_path in rmb_files
        ]

        # Process results as they are completed
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(rmb_files),
            desc="Resizing episodes",
        ):
            try:
                # CRITICAL: This is where we attempt to retrieve the result or the exception
                result = future.result() 
            except Exception as e:
                # CRITICAL: Catch and log the worker exception in the main process
                # This helps identify the file even if the worker crashes (BrokenProcessPool)
                LOGGER.error(f"Worker process failed with an unhandled exception: {type(e).__name__}: {e}. Try running with --max-workers 1 to identify the problematic file.", exc_info=True)
                continue # Skip this episode and continue with others

            if result is None:
                continue
            
            ep_path_str, processed_data = result
            ep_path = Path(ep_path_str)

            meta_data = {}
            try:
                # Use the string path for RmbData
                with RmbData(ep_path_str, "r") as rmb_data:
                    # Copy metadata attributes from the source file
                    for key, value in rmb_data.attrs.items():
                        meta_data[key] = value
            except Exception as e:
                LOGGER.warning(f"Could not read metadata from {ep_path_str}: {e}")

            # Determine the output filename stem
            stem = ep_path.name.replace(".rmb", "").replace(".hdf5", "")
            # If the file name is 'main.rmb.hdf5' (a common pattern for data loaders), use the parent directory name as the stem
            if ep_path.name == 'main.rmb.hdf5': 
                stem = ep_path.parent.name.replace('.rmb','')
                
            output_filename = output_dir / f"{stem}.{output_format}"
            
            # Save the processed data
            data_manager.save_data(str(output_filename), processed_data, meta_data)

    LOGGER.info("✅ Resizing and conversion complete!")
    LOGGER.info(f"Your resized RoboManipBaseline dataset is ready at: {output_dir}")


if __name__ == "__main__":
    # Use tyro to parse command-line arguments for the resize_rmb_dataset function.
    tyro.cli(resize_rmb_dataset, default=dict(max_workers=1))
