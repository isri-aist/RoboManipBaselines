# ruff: noqa: E501
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Tuple
from enum import Enum
import numpy as np

import torch
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from robo_manip_baselines.policy.lerobot_acfql.factory import make_policy, make_pre_post_processors

from robo_manip_baselines.policy.lerobot_acfql.config import RmbAcfqlTrainRLServerPipelineConfig
from robo_manip_baselines.policy.lerobot_acfql.acfql_policies.configuration_acfql import ACFQLConfig

logger = logging.getLogger(__name__)

HYDRATED_ACTOR_CONFIG_NAME = "actor_config.yaml"


def create_hydrated_lerobot_config(
    policy_config_path: str,
    model_meta_info: dict,
    camera_names: list,
    camera_resolution: list,
    output_dir: str = None,
    resume: bool = False,
) -> Tuple[RmbAcfqlTrainRLServerPipelineConfig, LeRobotDataset]:
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
    cfg = RmbAcfqlTrainRLServerPipelineConfig.from_pretrained(policy_config_path)

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
    cfg: RmbAcfqlTrainRLServerPipelineConfig,
    lerobot_dataset: LeRobotDataset,
    device: torch.device,
    pretrained_path: str | Path | None = None,
):
    """
    Initializes the ACFQL policy and its pre/post-processors.
    """
    policy_cfg = cfg.policy
    if pretrained_path:
        policy_cfg.pretrained_path = str(pretrained_path)
        
    env_cfg = cfg.get_env_cfg()

    policy = make_policy(cfg=policy_cfg, ds_meta=None, env_cfg=env_cfg).to(device)

    # Simplified kwargs based on LeRobot's internal factory logic.
    processor_kwargs = {"dataset_stats": lerobot_dataset.meta.stats}
    postprocessor_kwargs = {}

    # Overrides are only needed if a pretrained path is specified.
    if policy_cfg.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": str(device.type)},
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )
    logger.info("Successfully created policy, pre-processor, and post-processor.")
    return policy, preprocessor, postprocessor


class CommonLerobotAcfqlBase:
    """A base class providing common setup functionalities for ACFQL scripts."""

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

        if isinstance(self.state_keys, str):
            self.state_keys = [self.state_keys]
        if isinstance(self.action_keys, str):
            self.action_keys = [self.action_keys]

        self.velocity_keys = []
        self.effort_keys = []

        self.camera_names = self.data_config.get("camera_names", [])
        self.camera_resolution = self.data_config.get("camera_resolution")
        self.target_camera_resolution = self.data_config.get("target_camera_resolution")
        logger.info(f"Loaded data config. State keys: {self.state_keys}, Action keys: {self.action_keys}")

    def _log_model_info(self):
        """Logs key information about the loaded model and configuration."""
        if not all(hasattr(self, attr) for attr in ["device", "state_keys", "action_keys", "camera_names", "cfg"]):
            logger.warning("Could not log model info, some attributes are missing.")
            return

        state_dim = self.model_meta_info.get("state", {}).get("dim", "N/A")
        action_dim = self.model_meta_info.get("action", {}).get("dim", "N/A")

        logger.info("--- Model and Configuration Summary ---")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  State Keys: {self.state_keys}")
        logger.info(f"  State Dim: {state_dim}")
        logger.info(f"  Action Keys: {self.action_keys}")
        logger.info(f"  Action Dim: {action_dim}")
        logger.info(f"  Cameras: {self.camera_names or 'None'}")
        logger.info(f"  Policy Config: {self.cfg.policy}")
        logger.info("--------------------------------------")
