# ruff: noqa: E501
import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import draccus
import yaml

from robo_manip_baselines.common.utils.EnvUtils import get_env_names

try:
    ROBOMIP_PATH = Path(__file__).resolve().parents[4]
except (NameError, IndexError):
    ROBOMIP_PATH = Path(".")


from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from robo_manip_baselines.policy.lerobot_acfql.acfql_rl.configs import ACFQLTrainRLServerPipelineConfig
from robo_manip_baselines.policy.lerobot_acfql.acfql_policies.configuration_acfql import ACFQLConfig

# Flags to ensure registration happens only once
_RMB_ENVS_REGISTERED = False
_RMB_TELEOP_DEVICES_REGISTERED = False


@dataclass
class RmbAcfqlEnvConfig(EnvConfig):
    """Base environment configuration for RoboManipBaselines integration."""
    # These fields are populated dynamically by bridge_dataset_features
    features: Dict[str, Any] = field(default_factory=dict)
    features_map: Dict[str, str] = field(default_factory=dict)
    max_episode_steps: int = 1000

    @property
    def gym_kwargs(self) -> Dict[str, Any]:
        return {}


@dataclass
class RmbTeleopConfig(TeleoperatorConfig):
    """Base teleoperator configuration for RoboManipBaselines integration."""
    use_gripper: bool | None = None


def create_env_config_from_dataset_features(
    env_cfg: RmbAcfqlEnvConfig,
    dataset_features: Dict[str, Any],
) -> RmbAcfqlEnvConfig:
    """
    Populates the environment config with features derived from dataset metadata.
    
    This function defines the observation and action spaces for the LeRobot policy
    based on the features from the dataset, following the pattern used in ACT rollout.
    
    Args:
        env_cfg: The environment configuration to populate.
        dataset_features: The 'features' dict from dataset.meta.info (ds_meta.info['features']).
    
    Returns:
        The populated environment configuration.
    """
    logging.debug("Creating environment config from dataset features...")
    features = {}
    features_map = {}

    # Process all features from the dataset
    for key, feature_info in dataset_features.items():
        if key.startswith("observation."):
            # Determine feature type based on key
            if "state" in key:
                feature_type = FeatureType.STATE
            elif "images" in key:
                feature_type = FeatureType.VISUAL
            else:
                # Handles velocity, effort, etc.
                feature_type = FeatureType.STATE
            
            # Create the feature
            features[key] = PolicyFeature(type=feature_type, shape=feature_info['shape'])
            features_map[key] = key
            logging.debug(f"Added observation feature '{key}' with shape: {feature_info['shape']}")
        
        elif key.startswith("action"):
            # Action features
            features[key] = PolicyFeature(type=FeatureType.ACTION, shape=feature_info['shape'])
            features_map[key] = key
            logging.debug(f"Added action feature '{key}' with shape: {feature_info['shape']}")

    env_cfg.features = features
    env_cfg.features_map = features_map
    logging.debug(f"Final features: {env_cfg.features}")
    logging.debug(f"Final features_map: {env_cfg.features_map}")
    return env_cfg


def register_rmb_envs():
    """Dynamically registers RoboManipBaselines environments with LeRobot."""
    global _RMB_ENVS_REGISTERED
    if _RMB_ENVS_REGISTERED:
        logging.debug("RoboManipBaselines envs already registered.")
        return

    rmb_env_names = get_env_names()
    envs_root = ROBOMIP_PATH / "robo_manip_baselines" / "envs"

    for env_name in rmb_env_names:
        # Dynamically create a composite config class that inherits from our base
        # This allows LeRobot's factory to instantiate it correctly.
        composite_config_class = dataclass(type(f"Acfql{env_name}EnvConfig", (RmbAcfqlEnvConfig,), {}))
        EnvConfig.register_subclass(env_name)(composite_config_class)
        logging.debug(f"Registered robomanip_baseline env '{env_name}' with lerobot for ACFQL.")
    
    _RMB_ENVS_REGISTERED = True


def register_rmb_teleop_devices():
    """Dynamically registers RoboManipBaselines teleop devices with LeRobot."""
    global _RMB_TELEOP_DEVICES_REGISTERED
    if _RMB_TELEOP_DEVICES_REGISTERED:
        logging.debug("RoboManipBaselines teleop devices already registered.")
        return

    teleop_dir = ROBOMIP_PATH / "teleop"
    device_names = [f.stem.replace("InputDevice", "").lower() for f in teleop_dir.glob("*InputDevice.py")]

    for device_name in device_names:
        if not device_name:
            continue
        config_class_name = f"{device_name.capitalize()}TeleopConfig"
        docstring = f"Dynamically generated configuration for the {device_name} teleop device."
        new_config_class = type(config_class_name, (RmbTeleopConfig,), {"__doc__": docstring})
        new_config_class = dataclass(new_config_class)
        globals()[config_class_name] = new_config_class
        TeleoperatorConfig.register_subclass(device_name)(new_config_class)
        logging.debug(f"Registered robomanip_baseline teleop device '{device_name}' with lerobot.")
    
    _RMB_TELEOP_DEVICES_REGISTERED = True


@dataclass
class RmbAcfqlTrainRLServerPipelineConfig(ACFQLTrainRLServerPipelineConfig):
    """Main configuration class that bridges RoboManipBaselines with LeRobot's ACFQL."""
    env: EnvConfig = field(default_factory=RmbAcfqlEnvConfig)

    def bridge_dataset_features(self, dataset_features: Dict[str, Any]):
        """
        Hydrates the configuration by bridging features from the LeRobot dataset.
        This follows the pattern used in ACT rollout, using dataset metadata as the source of truth.
        
        Args:
            dataset_features: The 'features' dict from dataset.meta.info
        """
        create_env_config_from_dataset_features(self.env, dataset_features)

        # Map the dataset features to the policy's expected inputs/outputs.
        self.policy.input_features = {
            k: v for k, v in self.env.features.items() if k.startswith("observation.")
        }
        self.policy.output_features = {
            k: v for k, v in self.env.features.items() if k.startswith("action")
        }
        logging.info("Successfully bridged dataset features to policy input/output features.")

    def get_env_cfg(self) -> EnvConfig:
        """Returns the hydrated environment configuration."""
        if not self.env.features or not self.env.features_map:
            raise RuntimeError("`bridge_dataset_features` must be called before `get_env_cfg`.")
        return self.env

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str | Path, **kwargs):
        """Loads configuration from a YAML file, registering custom components first."""
        config_file = str(pretrained_name_or_path)
        if not Path(config_file).is_file():
            raise FileNotFoundError(f"Config file not found at path: {config_file}")

        # Register custom envs and teleop devices so draccus can decode them.
        register_rmb_envs()
        register_rmb_teleop_devices()

        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)

        try:
            main_config = draccus.decode(cls, config_dict)
        except Exception as e:
            logging.error(f"Failed to decode configuration from {config_file}: {e}")
            raise

        return main_config
