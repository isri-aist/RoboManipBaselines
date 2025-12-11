# --- ruff: noqa: E501 ---
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import torch
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.sac.configuration_sac import SACConfig

from robo_manip_baselines.policy.lerobot_hil_serl.config import RmbTrainRLServerPipelineConfig

logger = logging.getLogger(__name__)

HYDRATED_ACTOR_CONFIG_NAME = "actor_config.yaml"

def create_hydrated_lerobot_config(
    policy_config_path: str,
    model_meta_info: dict,
    camera_names: list,
    camera_resolution: list,
    output_dir: str = None,
    resume: bool = False,
) -> Tuple[RmbTrainRLServerPipelineConfig, LeRobotDataset]:
    """
    Loads, bridges, and hydrates the LeRobot configuration.
    Now uses dataset features as the source of truth, following the ACT rollout pattern.

    Args:
        policy_config_path: Path to the policy configuration YAML.
        model_meta_info: Dictionary containing model metadata (kept for compatibility).
        camera_names: List of camera names.
        camera_resolution: Target camera resolution.
        output_dir: Path to the output directory (used by learner to save hydrated config).
        resume: Resume flag.

    Returns:
        A tuple containing the hydrated config object and the LeRobotDataset instance.
    """
    cfg = RmbTrainRLServerPipelineConfig.from_pretrained(policy_config_path)

    if output_dir:
        cfg.output_dir = Path(output_dir)
    cfg.resume = resume
    cfg.policy.resume = resume

    # Load dataset to get features - dataset is the source of truth
    logger.info("Loading dataset metadata to bridge features and stats.")
    dataset_repo_id = cfg.dataset.repo_id
    if not dataset_repo_id:
        raise ValueError("`dataset.repo_id` not found in policy config.")
    lerobot_dataset = LeRobotDataset(repo_id=dataset_repo_id, root=cfg.dataset.root)
    
    # Get features from dataset metadata
    all_features = lerobot_dataset.meta.info.get("features", {})
    if not all_features:
        raise ValueError("Could not find 'features' in the dataset metadata (ds_meta.info).")
    
    # Bridge features from dataset (following ACT rollout pattern)
    logger.info("Bridging features from dataset metadata (source of truth).")
    cfg.bridge_dataset_features(all_features)
    
    # Bridge stats from the lerobot dataset
    cfg.policy.dataset_stats = lerobot_dataset.meta.stats
    logger.info("Successfully bridged features and stats from LeRobotDataset.")

    if not hasattr(cfg.policy, "dataset_stats") or cfg.policy.dataset_stats is None:
        raise ValueError(
            "`policy.dataset_stats` not found after bridging. Check config and dataset compatibility."
        )

    # For the learner: Save the complete, hydrated config for the actor to use.
    if output_dir:
        hydrated_config_path = cfg.output_dir / HYDRATED_ACTOR_CONFIG_NAME
        hydrated_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(hydrated_config_path, "w") as f:
            yaml.dump(asdict(cfg), f, default_flow_style=False)
        logger.info(f"Saved hydrated actor config to {hydrated_config_path}")

    return cfg, lerobot_dataset

def setup_policy_and_processors(
    cfg: RmbTrainRLServerPipelineConfig,
    lerobot_dataset: LeRobotDataset,
    device: torch.device,
    pretrained_path: str | Path | None = None,
):
    """
    Initializes the policy and its pre/post-processors.

    Args:
        cfg: The hydrated configuration object.
        lerobot_dataset: The LeRobotDataset instance with metadata.
        device: The torch device to move the policy to.
        pretrained_path: Optional path to a pretrained model checkpoint.

    Returns:
        A tuple containing the policy, preprocessor, and postprocessor.
    """
    policy_cfg = cfg.policy
    if pretrained_path:
        policy_cfg.pretrained_path = str(pretrained_path)

    env_cfg = cfg.get_env_cfg()
    
    # MODIFICATION: Use ds_meta=None and pass env_cfg, matching lerobot/learner.py and actor.py 
    policy = make_policy(cfg=policy_cfg, ds_meta=None, env_cfg=env_cfg).to(device)

    # --- MODIFICATION START ---
    # Build processor overrides
    processor_kwargs = {}
    postprocessor_kwargs = {}

    # Pass dataset_stats only when not resuming from saved processor state
    # (or if there's no pretrained path)
    if (policy_cfg.pretrained_path and not cfg.resume) or not policy_cfg.pretrained_path:
        # This kwarg is for the base call, not an override
        processor_kwargs["dataset_stats"] = lerobot_dataset.meta.stats 

    if policy_cfg.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": str(device.type)}, 
            "normalizer_processor": {
                "stats": lerobot_dataset.meta.stats, # Use stats from the bridged config
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            }, 
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": lerobot_dataset.meta.stats, # Use stats from the bridged config
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            }, 
        }
    # --- MODIFICATION END ---


    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path, # Pass pretrained_path
        **processor_kwargs,                         # Pass kwargs
        **postprocessor_kwargs,                     # Pass kwargs
    )
    logger.info("Successfully created policy, pre-processor, and post-processor.")
    cfg.policy.dataset_stats
    return policy, preprocessor, postprocessor


class CommonLerobotHilSerlBase:
    """A base class providing common setup functionalities for HIL-SERL scripts."""

    def set_common_args(self, parser):
        """Adds arguments common to both training and rollout scripts."""
        parser.add_argument("--policy_config_path", type=str, required=True)
        parser.add_argument("--resume", action="store_true")

    def load_data_config(self):
        """Loads data config from YAML and sets instance attributes."""
        try:
            with open(self.args.data_config_path) as f:
                self.data_config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Error: Data configuration file not found at {self.args.data_config_path}")
            sys.exit(1)

        self.state_keys = self.data_config["data_mapping"]["state"]
        self.action_keys = self.data_config["data_mapping"]["action"]
        
        # --- MODIFICATION START ---
        # Load optional velocity and effort keys. Fallback to empty lists if not present.
        self.velocity_keys = self.data_config["data_mapping"].get("velocity", [])
        self.effort_keys = self.data_config["data_mapping"].get("effort", [])
        
        # Ensure they are lists for consistent processing, as YAML can have a single string.
        if isinstance(self.velocity_keys, str):
            self.velocity_keys = [self.velocity_keys]
        if isinstance(self.effort_keys, str):
            self.effort_keys = [self.effort_keys]
        # --- MODIFICATION END ---

        self.camera_names = self.data_config.get("camera_names", [])
        self.camera_resolution = self.data_config.get("camera_resolution")
        self.target_camera_resolution = self.data_config.get("target_camera_resolution")
        logger.info(
            f"Loaded data config. State keys: {self.state_keys}, Action keys: {self.action_keys}, Velocity keys: {self.velocity_keys}, Effort keys: {self.effort_keys}"
        )

    def _log_model_info(self):
        """Logs key information about the loaded model and configuration."""
        if not all(
            hasattr(self, attr)
            for attr in ["device", "state_keys", "action_keys", "camera_names", "cfg"]
        ):
            logger.warning("Could not log model info, some attributes are missing.")
            return

        if not hasattr(self, "state_dim") or not hasattr(self, "action_dim"):
             # Fallback to model_meta_info if dims not pre-calculated
            if hasattr(self, "model_meta_info"):
                self.state_dim = self.model_meta_info["state"].get("dim", "N/A")
                self.action_dim = self.model_meta_info["action"].get("dim", "N/A")
            else:
                 self.state_dim, self.action_dim = "N/A", "N/A"


        logger.info("--- Model and Configuration Summary ---")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  State Keys: {self.state_keys}")
        logger.info(f"  State Dim: {self.state_dim}")
        logger.info(f"  Action Keys: {self.action_keys}")
        logger.info(f"  Action Dim: {self.action_dim}")
        logger.info(f"  Cameras: {self.camera_names if self.camera_names else 'None'}")
        logger.info(f"  Policy Config: {self.cfg.policy}")
        logger.info("--------------------------------------")
