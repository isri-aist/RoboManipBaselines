import json
import logging
from pathlib import Path

import torch

# --- LeRobot Imports ---
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device

# --- RoboManipBaselines Imports ---
from robo_manip_baselines.common.base.TeleopRolloutBase import TeleopRolloutBase

# --- Local Imports ---
from robo_manip_baselines.policy.lerobot_base.RolloutLerobotBase import RolloutLerobotBase

# Configure logging for clear and informative output.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


class RolloutLerobotPi0(RolloutLerobotBase, TeleopRolloutBase):
    """
    A refined rollout script for the lerobot Pi0 policy, inheriting common
    functionality from RolloutLerobotBase.
    """

    def setup_policy(self):
        """
        Loads the Pi0 policy, its configuration, and pre/post-processors
        from a checkpoint directory.
        """
        set_seed(self.rollout_config.get("seed", 42))
        self.device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu", log=True)

        logger.info(f"Loading policy from checkpoint: {self.args.checkpoint}")
        pretrained_path = Path(self.args.checkpoint)

        train_config_file = pretrained_path / "train_config.json"
        if not train_config_file.exists():
            raise FileNotFoundError(f"Required `train_config.json` not found in: {train_config_file}")

        with train_config_file.open("r") as f:
            train_cfg_dict = json.load(f)

        dataset_repo_id = train_cfg_dict["dataset"]["repo_id"]
        logger.info(f"Loading dataset metadata from: {dataset_repo_id}")
        dataset = LeRobotDataset(dataset_repo_id)
        ds_meta = dataset.meta

        config_file = pretrained_path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Required `config.json` not found in: {config_file}")
        with config_file.open("r") as f:
            config_dict = json.load(f)
        self.policy_cfg = PI0Config(**config_dict)

        self.policy_cfg.pretrained_path = str(pretrained_path)
        policy = make_policy(self.policy_cfg, ds_meta=ds_meta)
        self.policy = policy.to(self.device).eval()

        logger.info("Loading pre-processor and post-processor...")
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy_cfg,
            pretrained_path=str(pretrained_path),
            dataset_stats=ds_meta.stats,
        )
        self._log_model_info()
