import json
import logging
import sys
from pathlib import Path

import cv2
import matplotlib.pylab as plt
import numpy as np
import torch
import yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg

# --- Path Setup ---
# This setup ensures that the script can find the necessary modules from
# both the RoboManipBaselines and the vendored lerobot libraries.
try:
    ROBOMANIP_PATH = Path(__file__).resolve().parents[3]
    if str(ROBOMANIP_PATH) not in sys.path:
        sys.path.append(str(ROBOMANIP_PATH))
    LEROBOT_PATH = ROBOMANIP_PATH / "third_party" / "lerobot" / "src"
    if str(LEROBOT_PATH) not in sys.path:
        sys.path.append(str(LEROBOT_PATH))
except NameError:
    # Handles cases where __file__ is not defined (e.g., in interactive notebooks).
    print("Warning: Could not determine ROBOMANIP_PATH. Assuming paths are correctly set in the environment.")

# --- LeRobot Imports ---
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device

# --- RoboManipBaselines Imports ---
from robo_manip_baselines.common.data.DataKey import DataKey
from robo_manip_baselines.common.utils.MiscUtils import remove_prefix

logger = logging.getLogger(__name__)


class RolloutLerobotBase:
    """
    A base class for lerobot rollouts, containing common logic for argument parsing,
    metadata setup, observation processing, policy inference, and visualization.
    """

    @property
    def policy_name(self):
        """Returns the name of the policy, derived from the class name."""
        return remove_prefix(self.__class__.__name__, "Rollout")

    def set_additional_args(self, parser):
        """Adds policy-specific command-line arguments."""
        super().set_additional_args(parser)
        parser.add_argument(
            "--data_config_path",
            type=str,
            required=False,
            default=None,
            help="Path to the YAML data configuration file for this rollout script.",
        )

    def setup_model_meta_info(self):
        """
        Sets up metadata (state/action keys, camera names) using the provided YAML config.
        This information bridges the RoboManipBaselines environment with the lerobot policy.
        Command-line arguments for camera settings will override YAML values.
        """
        # Initialize model_meta_info only if it doesn't exist
        if not hasattr(self, 'model_meta_info'):
            self.model_meta_info = {
                "state": {"keys": [], "dim": 0},
                "action": {"keys": [], "dim": 0},
                "image": {"camera_names": [], "camera_crops": {}},
            }

        # Early return if no config path provided
        if not self.args.data_config_path:
            logger.warning("`--data_config_path` not provided. Using default empty metadata in RolloutLerobotBase.")
            # Still run camera parsing logic in case args are provided
        else:
            # Load YAML configuration
            try:
                with open(self.args.data_config_path, "r") as f:
                    self.rollout_config = yaml.safe_load(f)
            except FileNotFoundError:
                logger.error(f"Data configuration file not found at {self.args.data_config_path}")
                sys.exit(1)
            except yaml.YAMLError as e:
                logger.error(f"Failed to parse YAML config at {self.args.data_config_path}: {e}")
                sys.exit(1)

            # Extract data mapping
            data_mapping = self.rollout_config.get("data_mapping", {})
            if not data_mapping:
                logger.error("No 'data_mapping' found in the configuration file.")
                sys.exit(1)

            # Assign state and action keys, ensuring they are lists
            self.state_keys = data_mapping.get("state", [])
            if isinstance(self.state_keys, str):
                self.state_keys = [self.state_keys]
            self.action_keys = data_mapping.get("action", [])
            if isinstance(self.action_keys, str):
                self.action_keys = [self.action_keys]

            # Handle velocity and effort keys, ensuring they are lists
            self.velocity_keys = data_mapping.get("velocity", [])
            if isinstance(self.velocity_keys, str):
                self.velocity_keys = [self.velocity_keys]
            self.effort_keys = data_mapping.get("effort", [])
            if isinstance(self.effort_keys, str):
                self.effort_keys = [self.effort_keys]

            # Combine all state-like keys for dimension calculation
            all_state_keys = self.state_keys + self.velocity_keys + self.effort_keys
            logger.info(f"State keys: {all_state_keys}")
            logger.info(f"Action keys: {self.action_keys}")

            # Calculate dimensions
            try:
                self.model_meta_info["state"]["dim"] = sum(DataKey.get_dim(key, self.env) for key in all_state_keys)
                self.model_meta_info["action"]["dim"] = sum(DataKey.get_dim(key, self.env) for key in self.action_keys)
                logger.info(f"State dimension: {self.model_meta_info['state']['dim']}")
                logger.info(f"Action dimension: {self.model_meta_info['action']['dim']}")
            except Exception as e:
                logger.error(f"Error calculating dimensions: {e}")
                sys.exit(1)

            # Update state and action keys in model_meta_info
            self.model_meta_info["state"]["keys"] = all_state_keys
            self.model_meta_info["action"]["keys"] = self.action_keys

            # Handle camera configurations from YAML
            self.model_meta_info["image"]["camera_names"] = self.rollout_config.get("camera_names", [])
            # These are read from YAML and will be used as defaults
            self.model_meta_info["image"]["camera_crops"] = self.rollout_config.get("camera_crops", {})
            self.model_meta_info["image"]["target_camera_resolution"] = self.rollout_config.get("target_camera_resolution", None)

        # --- Added from TeleopRolloutBase to allow arg override ---

        # Parse camera crops from args, using YAML/defaults as fallback
        if not hasattr(self, "parsed_camera_crops") or self.parsed_camera_crops is None:
            self.parsed_camera_crops = self.model_meta_info.get("image", {}).get("camera_crops", {})

        if self.args.camera_crops:
            # Always override if explicitly provided via args
            self.parsed_camera_crops = {}
            for crop_arg in self.args.camera_crops:
                cam_name, crop_str = crop_arg.split(":")
                x, y, w, h = map(int, crop_str.split(","))
                self.parsed_camera_crops[cam_name] = (x, y, w, h)
        
        self.model_meta_info.setdefault("image", {})["parsed_camera_crops"] = self.parsed_camera_crops

        # Parse target camera resolution from args, using YAML/defaults as fallback
        if not hasattr(self, "target_camera_resolution") or self.target_camera_resolution is None:
            self.target_camera_resolution = self.model_meta_info.get("image", {}).get("target_camera_resolution")

        if self.args.target_camera_resolution:
            try:
                h_str, w_str = self.args.target_camera_resolution.split(",")
                self.target_camera_resolution = (int(h_str), int(w_str))
            except Exception as e:
                raise ValueError(
                    f"Invalid --target_camera_resolution format '{self.args.target_camera_resolution}'. "
                    f"Expected 'height,width' (e.g. 480,640). Error: {e}"
                )
        
        self.model_meta_info["image"]["target_camera_resolution"] = self.target_camera_resolution
        
        # --- End of added block ---

        # Finalize dimensions and keys for the base classes
        self.state_keys = self.model_meta_info["state"]["keys"]
        self.action_keys = self.model_meta_info["action"]["keys"]
        self.camera_names = self.model_meta_info["image"]["camera_names"]
        self.state_dim = self.model_meta_info["state"]["dim"]
        self.action_dim = self.model_meta_info["action"]["dim"]
        
        # Set skip from args if provided, otherwise default (e.g., from YAML via RolloutBase)
        if self.args.skip is None:
            self.args.skip = self.model_meta_info.get("data", {}).get("skip", 1)
        if self.args.skip_draw is None:
            self.args.skip_draw = self.args.skip

        logger.info('model_meta_info: %s', self.model_meta_info)

    def _log_model_info(self):
        """Logs key information about the loaded model and configuration for debugging."""
        logger.info("--- Model and Configuration Loaded ---")
        logger.info(f"  Device: {self.device}")
        logging.info(f"  State Keys: {self.state_keys} (Dim: {self.state_dim})")
        logging.info(f"  Action Keys: {self.action_keys} (Dim: {self.action_dim})")
        logger.info(f"  Camera Names: {self.camera_names}")
        logger.info(f"  Camera Crops: {self.parsed_camera_crops or 'None'}")
        logger.info(f"  Policy Config: {self.policy_cfg}")
        logger.info("--------------------------------------")

    def setup_policy(self):
        """Placeholder for policy setup. Must be implemented by subclasses."""
        raise NotImplementedError("Each rollout must implement its own `setup_policy` method.")

    def reset_variables(self):
        """Resets variables at the start of each new episode."""
        super().reset_variables()
        if hasattr(self, "policy") and self.policy:
            self.policy.reset()
        if self.action_dim > 0:
            self.policy_action_list = np.empty((0, self.action_dim), dtype=np.float64)

    def get_state(self, debug=False):
        """Concatenates ALL state data (pos, vel, effort) from the environment and converts to a Torch tensor."""
        all_state_keys = self.state_keys + self.velocity_keys + self.effort_keys
        if not all_state_keys:
            state_data = np.zeros(0, dtype=np.float32)
        else:
            state_data = np.concatenate(
                [self.motion_manager.get_data(state_key, self.obs) for state_key in all_state_keys]
            )
        state_tensor = torch.tensor(state_data[np.newaxis], dtype=torch.float32).to(self.device)
        if debug:
            logger.info(f"Step {self.rollout_time_idx}: Combined State tensor | Shape: {state_tensor.shape}")
        return state_tensor

    def get_images(self, debug=False):
        """Prepares images from the environment for policy inference."""
        images = {}
        for name in self.camera_names:
            img_hwc = self.info["rgb_images"][name]
            if debug:
                logger.info(
                    f"Step {self.rollout_time_idx}: Retrieved image for camera {name} | Shape: "
                    f"{img_hwc.shape}"
                )

            if self.parsed_camera_crops and name in self.parsed_camera_crops:
                x, y, w, h = self.parsed_camera_crops[name]
                img_hwc = img_hwc[y : y + h, x : x + w]
                if debug:
                    logger.info(
                        f"Step {self.rollout_time_idx}: Cropped image for camera {name} | Crop: "
                        f"(x={x}, y={y}, w={w}, h={h}) | New shape: {img_hwc.shape}"
                    )

            if self.target_camera_resolution:
                target_h, target_w = self.target_camera_resolution
                img_hwc = cv2.resize(img_hwc, (target_w, target_h), interpolation=cv2.INTER_AREA)
                if debug:
                    logger.info(
                        f"Step {self.rollout_time_idx}: Resized image for camera {name} | "
                        f"Target resolution: ({target_w}, {target_h}) | New shape: {img_hwc.shape}"
                    )

            img_bchw = np.transpose(img_hwc[np.newaxis], (0, 3, 1, 2))
            image_tensor = torch.from_numpy(img_bchw.copy()).to(self.device).to(torch.float32) / 255.0
            if debug:
                logger.info(
                    f"Step {self.rollout_time_idx}: Converted image for camera {name} to tensor | "
                    f"Shape: {image_tensor.shape}, Device: {image_tensor.device}, Dtype: {image_tensor.dtype}"
                )
            
            images[f"observation.images.{name}"] = image_tensor

        if debug:
            logger.info(
                f"Step {self.rollout_time_idx}: Prepared images for {len(images)} cameras"
            )
        
        return images

    def _get_raw_observation_batch(self, debug=False):
        """Gathers the current raw observation and returns a batch for the policy/buffer."""
        batch = {"task": [self.args.target_task]}
        
        if self.state_keys:
            state_data = np.concatenate([self.motion_manager.get_data(key, self.obs) for key in self.state_keys])
            batch["observation.state"] = torch.tensor(state_data[np.newaxis], dtype=torch.float32).to(self.device)

        if self.velocity_keys:
            vel_data = np.concatenate([self.motion_manager.get_data(key, self.obs) for key in self.velocity_keys])
            batch["observation.velocity"] = torch.tensor(vel_data[np.newaxis], dtype=torch.float32).to(self.device)

        if self.effort_keys:
            effort_data = np.concatenate([self.motion_manager.get_data(key, self.obs) for key in self.effort_keys])
            batch["observation.effort"] = torch.tensor(effort_data[np.newaxis], dtype=torch.float32).to(self.device)    
                
        # Add images to the batch
        images = self.get_images(debug)
        batch.update(images)
        # --- END FIX ---
                
        return batch

    def infer_policy(self, debug=False):
        """Performs a single step of policy inference using the full lerobot pipeline."""
        batch = self._get_raw_observation_batch(debug)
        processed_batch = self.preprocessor(batch)

        with torch.inference_mode():
            raw_action = self.policy.select_action(processed_batch)

        final_action_tensor = self.postprocessor(raw_action)
        self.policy_action = final_action_tensor.to("cpu").numpy().squeeze().astype(np.float64)

        if debug:
            logger.info(f"Step {self.rollout_time_idx}: Final Action Sent to Env: {self.policy_action}")

        if self.action_dim > 0 and not self.args.no_plot:
            action_to_append = np.expand_dims(self.policy_action, axis=0)
            self.policy_action_list = np.concatenate([self.policy_action_list, action_to_append])

    def setup_plot(self, fig_ax=None):
        """Initializes the matplotlib figure for visualization."""
        if self.args.no_plot:
            return
        num_cameras = len(self.camera_names)
        num_cols = max(num_cameras + 1, 4)
        self.fig, self.axes = plt.subplots(
            1, num_cols, figsize=(num_cols * 4, 4), dpi=100, squeeze=False, constrained_layout=True
        )
        self.canvas = FigureCanvasAgg(self.fig)
        super().setup_plot(fig_ax=(self.fig, self.axes[0]))

    def draw_plot(self):
        """Draws the visualization plot for the current rollout step."""
        if self.args.no_plot or not hasattr(self, "canvas"):
            return

        axes_row = self.axes[0]
        for ax in axes_row:
            ax.cla()
            ax.axis("off")

        self.plot_images(axes_row[: len(self.camera_names)])
        self.plot_action(axes_row[len(self.camera_names)])

        self.canvas.draw()
        frame = np.asarray(self.canvas.buffer_rgba())
        cv2.imshow(self.policy_name, cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR))

    def cleanup(self):
        """Cleans up resources (e.g., OpenCV windows) upon exit."""
        cv2.destroyAllWindows()