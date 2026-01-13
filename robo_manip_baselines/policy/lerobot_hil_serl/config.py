# ruff: noqa: E501
import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import draccus
import yaml

# --- robomanipbaseline Imports ---
from robo_manip_baselines.common.utils.EnvUtils import get_env_names

ROBOMANIP_PATH = Path(__file__).resolve().parents[3]

# --- lerobot Imports ---
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.policies.sac.configuration_sac import SACConfig

# Flags to ensure registration happens only once
_RMB_ENVS_REGISTERED = False
_RMB_TELEOP_DEVICES_REGISTERED = False

# New dataclasses to match the expected nested structure for the processor
@dataclass
class RewardClassifierConfig:
    """Configuration for the reward classifier."""

    pretrained_path: str | None = None
    success_threshold: float | None = None
    success_reward: float | None = None
    terminate_on_success: bool | None = None


@dataclass
class ImagePreprocessingConfig:
    """Configuration for image preprocessing."""

    resize_size: Optional[List[int]] = None


@dataclass
class ProcessorConfig:
    """Configuration for environment processors."""

    reward_classifier: Optional[RewardClassifierConfig] = field(default_factory=RewardClassifierConfig)
    image_preprocessing: Optional[ImagePreprocessingConfig] = field(default_factory=ImagePreprocessingConfig)


@dataclass
class RmbHilSerlEnvConfig(EnvConfig):
    """
    A base class to hold HIL-SERL specific fields for robomanipbaseline environments.
    This now includes a 'processor' field to handle components like the reward classifier.
    """

    camera_names: List[str] = field(default_factory=list)
    state_keys: List[str] = field(default_factory=list)
    action_keys: List[str] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)
    features_map: Dict[str, str] = field(default_factory=dict)
    processor: Optional[ProcessorConfig] = field(default_factory=ProcessorConfig)
    max_episode_steps: int = 1000

    @property
    def gym_kwargs(self) -> Dict[str, Any]:
        """
        Implementation of the abstract property from lerobot's EnvConfig.
        Returns an empty dict as robomanipbaseline envs are not instantiated
        with gym.make() in the same way by the lerobot framework itself.
        """
        return {}


@dataclass
class RmbTeleopConfig(TeleoperatorConfig):
    """Intermediary config for robomanipbaseline teleoperation devices."""

    use_gripper: bool | None = None


def create_env_config_from_dataset_features(
    env_cfg: RmbHilSerlEnvConfig,
    dataset_features: Dict[str, Any],
) -> RmbHilSerlEnvConfig:
    """
    Populates an EnvConfig object with features derived from the LeRobot dataset.
    This follows the pattern used in ACT rollout, using dataset metadata as the source of truth.
    
    Args:
        env_cfg: The base environment configuration object to populate.
        dataset_features: The 'features' dict from dataset.meta.info (ds_meta.info['features']).

    Returns:
        The populated environment configuration object.
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

    # Update the provided env_cfg object in-place
    env_cfg.features = features
    env_cfg.features_map = features_map
    logging.debug(f"Final features: {env_cfg.features}")
    logging.debug(f"Final features_map: {env_cfg.features_map}")

    return env_cfg


def register_rmb_envs():
    """
    Finds all environments in robomanipbaseline, dynamically imports their
    config classes, creates a new composite config class that includes HIL-SERL
    specific fields, and registers it with draccus.
    """
    global _RMB_ENVS_REGISTERED
    if _RMB_ENVS_REGISTERED:
        logging.debug("RoboManipBaselines envs already registered.")
        return

    rmb_env_names = get_env_names()
    envs_root = ROBOMANIP_PATH / "robo_manip_baselines" / "envs"

    for env_name in rmb_env_names:
        base_config_class_name = f"{env_name}EnvConfig"
        config_file_name = f"Operation{env_name}.py"  # Configs are often in the operation file
        found_path = next(envs_root.rglob(config_file_name), None)
        BaseEnvConfigClass = None

        if found_path:
            relative_path = found_path.relative_to(ROBOMANIP_PATH)
            module_path = str(relative_path).replace(".py", "").replace("/", ".")
            try:
                module = importlib.import_module(module_path)
                # Look for a config class within the operation module
                if hasattr(module, base_config_class_name):
                    BaseEnvConfigClass = getattr(module, base_config_class_name)
            except (ImportError, AttributeError) as e:
                logging.warning(f"Could not import or find config for env '{env_name}': {e}.")

        if BaseEnvConfigClass is None:
            composite_config_class = dataclass(type(f"HilSerl{env_name}EnvConfig", (RmbHilSerlEnvConfig,), {}))
        else:
            composite_config_class_name = f"HilSerl{base_config_class_name}"
            composite_config_class = dataclass(
                type(
                    composite_config_class_name,
                    (BaseEnvConfigClass, RmbHilSerlEnvConfig),
                    {},
                )
            )

        EnvConfig.register_subclass(env_name)(composite_config_class)
        logging.debug(f"Registered robomanip_baseline env '{env_name}' with lerobot.")
    
    _RMB_ENVS_REGISTERED = True


def register_rmb_teleop_devices():
    """
    Finds all teleoperation devices in robomanipbaseline and dynamically
    registers a corresponding TeleoperatorConfig subclass for each one.
    """
    global _RMB_TELEOP_DEVICES_REGISTERED
    if _RMB_TELEOP_DEVICES_REGISTERED:
        logging.debug("RoboManipBaselines teleop devices already registered.")
        return

    teleop_dir = ROBOMANIP_PATH / "teleop"
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
class RmbTrainRLServerPipelineConfig(TrainRLServerPipelineConfig):
    """
    A custom training pipeline configuration that overrides `from_pretrained`
    to ensure custom environments and teleoperation devices are registered
    before the configuration is parsed by draccus.
    """

    env: EnvConfig = field(default_factory=RmbHilSerlEnvConfig)

    def bridge_dataset_features(self, dataset_features: Dict[str, Any]):
        """
        Populates the configuration object with features from the LeRobot dataset.
        This follows the pattern used in ACT rollout, using dataset metadata as the source of truth.
        
        Args:
            dataset_features: The 'features' dict from dataset.meta.info
        """
        # Call the factory function to populate self.env
        create_env_config_from_dataset_features(self.env, dataset_features)

        # The rest of the logic populates the policy features from the newly created env features
        self.policy.input_features = {
            k: v for k, v in self.env.features.items() if k.startswith("observation.")
        }
        self.policy.output_features = {
            k: v for k, v in self.env.features.items() if k.startswith("action")
        }
        logging.info("Successfully bridged dataset features to policy input/output features.")

    def get_env_cfg(self) -> EnvConfig:
        """
        Returns the populated environment configuration. This method should be called
        after `bridge_dataset_features` has been executed.
        """
        if not self.env.features or not self.env.features_map:
            raise RuntimeError(
                "`bridge_dataset_features` must be called before `get_env_cfg` to populate the environment configuration."
            )
        return self.env

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str | Path, **kwargs):
        """
        Loads configuration from a YAML file, handling dynamic registration
        of custom robomanipbaseline components.
        """
        config_file = str(pretrained_name_or_path)
        if not Path(config_file).is_file():
            raise FileNotFoundError(f"Config file not found at path: {config_file}")

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
