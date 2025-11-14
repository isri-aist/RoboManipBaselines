# ruff: noqa: E501
"""
This script provides an improved training pipeline for HIL-SERL, integrating
`robo-manip-baselines` with the `lerobot` distributed learning framework.

Key Improvements:
- Modular Training Loop: The main `train_loop` has been broken down into smaller,
  more manageable functions for initialization, the core training cycle, and cleanup.
- Enhanced Readability: Logic for critic, actor, and temperature updates is
  encapsulated in separate helper methods, making the main loop cleaner.
- Robust Error Handling: A comprehensive try/except/finally block ensures
  graceful shutdown and resource cleanup, even in case of errors.
- Clearer Logging: TensorBoard logging is centralized, and evaluation metrics
  are consistently logged.
- Refined Imports and Structure: Imports are logically grouped, and helper
  functions from `lerobot.rl.learner` are leveraged more effectively.
"""

import contextlib
import logging
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path
from queue import Empty
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.multiprocessing import Queue
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# --- Path Setup ---
try:
    ROBOMANIP_PATH = Path(__file__).resolve().parents[3]
    sys.path.append(str(ROBOMANIP_PATH))
    LEROBOT_PATH = ROBOMANIP_PATH / "third_party" / "lerobot" / "src"
    sys.path.append(str(LEROBOT_PATH))
except (NameError, IndexError):
    print("Warning: Could not determine ROBOMANIP_PATH. Assuming paths are set.")

# --- LeRobot Imports ---
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.buffer import ReplayBuffer, concatenate_batch_transitions
from lerobot.rl.learner import (
    add_actor_information_and_train,
    check_nan_in_transition,
    get_observation_features,
    handle_resume_logic,
    initialize_replay_buffer,
    load_training_state,
    log_training_info,
    make_optimizers_and_scheduler,
    process_interaction_messages,
    process_transitions,
    push_actor_policy_to_queue,
    save_training_checkpoint,
    start_learner,
    use_threads,
)
from lerobot.rl.process import ProcessSignalHandler
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device

# --- RoboManipBaselines Imports ---
from robo_manip_baselines.common.base.CroppedTrainBase import CroppedTrainBase
from robo_manip_baselines.policy.lerobot_hil_serl.HilSerlDataset import (
    HilSerlDataset,
)
from robo_manip_baselines.policy.lerobot_hil_serl.common import (
    CommonLerobotHilSerlBase,
    create_hydrated_lerobot_config,
    setup_policy_and_processors,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


class TrainLerobotHilSerl(CroppedTrainBase, CommonLerobotHilSerlBase):
    """
    A training script for HIL-SERL that integrates `robo-manip-baselines` with the
    `lerobot` distributed learning framework.
    """

    DatasetClass = HilSerlDataset

    def set_additional_args(self, parser):
        """Set command-line arguments for the training script."""
        super().set_additional_args(parser)
        self.set_common_args(parser)
        parser.add_argument("--data_config_path", type=str, required=True)
        parser.add_argument(
            "--reset_while_resuming",
            action="store_true",
            help="Reset training step and optimizers when resuming.",
        )
        parser.add_argument(
            "--use_lerobot_learner",
            action="store_true",
            help="Use the original add_actor_information_and_train from lerobot.rl.learner.",
        )
        parser.add_argument(
            "--plot_buffer_stats",
            action="store_true",
            help="Periodically save buffer statistics plots and a summary text file during training.",
        )

    def setup_policy(self):
        """Loads the policy, its configuration, and pre/post-processors."""
        self.device = get_safe_torch_device(
            "cuda" if torch.cuda.is_available() else "cpu", log=True
        )
        if self.args.reset_while_resuming:
            self.args.resume = True

        self.cfg, self.lerobot_dataset = create_hydrated_lerobot_config(
            policy_config_path=self.args.policy_config_path,
            model_meta_info=self.model_meta_info,
            camera_names=self.camera_names,
            camera_resolution=self.target_camera_resolution
            or self.camera_resolution,
            output_dir=self.args.checkpoint_dir,
            resume=self.args.resume,
        )

        self.policy, self.preprocessor, self.postprocessor = setup_policy_and_processors(
            cfg=self.cfg,
            lerobot_dataset=self.lerobot_dataset,
            device=self.device,
        )

        self.cfg.policy.dataset_stats = {} #saving the dataset_stats causes serialisation troubles so we remove them after creating the policy

        set_seed(self.cfg.seed)
        self.args.skip = self.args.skip or 1

        self._log_model_info()

    def setup_model_meta_info(self):
        """Loads data configuration and sets up metadata for the model."""
        self.load_data_config()

        self.args.state_keys = self.state_keys
        self.args.action_keys = self.action_keys
        self.args.camera_names = self.camera_names

        logger.info(f"Using state keys: {self.args.state_keys}")
        logger.info(f"Using action keys: {self.args.action_keys}")
        logger.info(f"Using camera_names: {self.args.camera_names}")

        super().setup_model_meta_info()

    def _initialize_buffers(self) -> Tuple[Optional[ReplayBuffer], ReplayBuffer, int]:
        """
        Initializes offline and online replay buffers.

        This implementation directly uses the pre-loaded `self.lerobot_dataset` to
        initialize the offline buffer when not resuming, avoiding an unnecessary
        reload of the dataset. For resuming, it loads the cached offline dataset from disk.
        """
        cfg = self.cfg
        batch_size = cfg.batch_size
        offline_buffer = None

        if cfg.dataset is not None and cfg.policy.offline_buffer_capacity > 0:
            offline_path = cfg.output_dir / "dataset_offline"
            is_resuming_offline = offline_path.is_dir() or self.args.resume
            logger.info(f"Initializing offline buffer (cache found: {is_resuming_offline}).")

            offline_dataset_to_load = None
            if is_resuming_offline and offline_path.is_dir():
                offline_dataset_to_load = LeRobotDataset(
                    repo_id=cfg.dataset.repo_id, root=offline_path
                )
            elif not is_resuming_offline:
                offline_dataset_to_load = self.lerobot_dataset

            if offline_dataset_to_load:
                offline_buffer = ReplayBuffer.from_lerobot_dataset(
                    offline_dataset_to_load,
                    capacity=cfg.policy.offline_buffer_capacity,
                    device=self.device,
                    state_keys=list(cfg.policy.input_features.keys()),
                    storage_device=cfg.policy.storage_device,
                    optimize_memory=True,
                )

            if cfg.policy.online_steps > 0:
                batch_size //= 2

        online_path = cfg.output_dir / "dataset"
        online_cfg = deepcopy(cfg)
        online_cfg.resume = online_path.is_dir() or self.args.resume
        logger.info(f"Initializing online buffer (cache found: {online_cfg.resume}).")
        online_buffer = initialize_replay_buffer(
            online_cfg, self.device, cfg.policy.storage_device
        )
        return offline_buffer, online_buffer, batch_size

    def _setup_logging(self) -> None:
        """Initializes WandB or TensorBoard loggers."""
        self.wandb_logger = (
            WandBLogger(self.cfg) if self.cfg.wandb.enable else None
        )
        self.tb_logger = (
            None
            if self.wandb_logger
            else SummaryWriter(log_dir=Path(self.cfg.output_dir) / "tensorboard_logs")
        )

    def _initialize_training_state(
        self,
    ) -> Tuple[int, int, Dict[str, Optimizer], Any]:
        """Handles resume logic and initializes optimizers and schedulers."""
        cfg = handle_resume_logic(self.cfg)
        self.policy.train()
        optimizers, lr_scheduler = make_optimizers_and_scheduler(
            cfg=cfg, policy=self.policy
        )

        opt_step, resume_interaction_step = load_training_state(
            cfg=cfg, optimizers=optimizers
        )

        if cfg.resume:
            self.policy.update_temperature()
            logger.info(
                f"Resumed training. Temperature set to: {self.policy.temperature:.4f}"
            )

        interaction_step_shift = resume_interaction_step or 0
        if self.args.reset_while_resuming and self.args.resume:
            logger.info("Resetting training step and optimizers.")
            opt_step, interaction_step_shift = 0, 0
            optimizers, lr_scheduler = make_optimizers_and_scheduler(
                cfg=cfg, policy=self.policy
            )

        optimization_step = opt_step or 0
        log_training_info(cfg=cfg, policy=self.policy)
        return optimization_step, interaction_step_shift, optimizers, lr_scheduler

    def _log_evaluation_metrics(self, interaction_message: Dict[str, Any]):
        """Logs evaluation metrics from interaction messages to TensorBoard."""
        if not self.tb_logger or not interaction_message:
            return

        step = interaction_message.get("Interaction step", 0)
        eval_metrics = [
            "Episodic reward",
            "Intervention rate",
            "Episode intervention",
            "Policy frequency [Hz]",
        ]
        for key in eval_metrics:
            if key in interaction_message:
                self.tb_logger.add_scalar(
                    f"eval/{key}", interaction_message[key], step
                )

    def _update_critic(self, batch: Dict, optimizers: Dict[str, Optimizer]) -> Dict[str, Any]:
        """Performs a single update step for the critic networks and returns training info."""
        cfg = self.cfg
        check_nan_in_transition(
            observations=batch["state"],
            actions=batch["action"],
            next_state=batch["next_state"],
        )
        obs_features, next_obs_features = get_observation_features(
            self.policy, batch["state"], batch["next_state"]
        )
        forward_batch = {
            **batch,
            "observation_feature": obs_features,
            "next_observation_feature": next_obs_features,
        }

        # Main critic update
        critic_output = self.policy.forward(forward_batch, model="critic")
        optimizers["critic"].zero_grad(set_to_none=True)
        critic_output["loss_critic"].backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.critic_ensemble.parameters(), cfg.policy.grad_clip_norm
        )
        optimizers["critic"].step()

        training_infos = {
            "loss_critic": critic_output["loss_critic"].item(),
            "critic_grad_norm": critic_grad_norm.item(),
        }

        # Discrete critic update
        if hasattr(self.policy, "discrete_critic") and self.policy.discrete_critic:
            discrete_critic_output = self.policy.forward(
                forward_batch, model="discrete_critic"
            )
            optimizers["discrete_critic"].zero_grad(set_to_none=True)
            discrete_critic_output["loss_discrete_critic"].backward()
            discrete_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.discrete_critic.parameters(), cfg.policy.grad_clip_norm
            )
            optimizers["discrete_critic"].step()
            training_infos.update(
                {
                    "loss_discrete_critic": discrete_critic_output[
                        "loss_discrete_critic"
                    ].item(),
                    "discrete_critic_grad_norm": discrete_critic_grad_norm.item(),
                }
            )
        return training_infos

    def _update_actor_and_temperature(
        self, batch: Dict, optimizers: Dict[str, Optimizer]
    ) -> Dict[str, Any]:
        """Performs actor and temperature updates, returning training info."""
        cfg = self.cfg
        check_nan_in_transition(
            observations=batch["state"],
            actions=batch["action"],
            next_state=batch["next_state"],
        )
        obs_features, next_obs_features = get_observation_features(
            self.policy, batch["state"], batch["next_state"]
        )
        forward_batch = {
            **batch,
            "observation_feature": obs_features,
            "next_observation_feature": next_obs_features,
        }

        training_infos = {}

        # --- Actor and Temperature Updates ---
        # Actor update
        actor_output = self.policy.forward(forward_batch, model="actor")
        optimizers["actor"].zero_grad(set_to_none=True)
        actor_output["loss_actor"].backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.actor.parameters(), cfg.policy.grad_clip_norm
        )
        optimizers["actor"].step()

        # Temperature update
        temp_output = self.policy.forward(forward_batch, model="temperature")
        optimizers["temperature"].zero_grad(set_to_none=True)
        temp_output["loss_temperature"].backward()
        temp_grad_norm = torch.nn.utils.clip_grad_norm_(
            [self.policy.log_alpha], cfg.policy.grad_clip_norm
        )
        optimizers["temperature"].step()

        training_infos = {
            "loss_actor": actor_output["loss_actor"].item(),
            "actor_grad_norm": actor_grad_norm.item(),
            "loss_temperature": temp_output["loss_temperature"].item(),
            "temperature_grad_norm": temp_grad_norm.item(),
            "temperature": self.policy.temperature,
        }
        self.policy.update_temperature()

        return training_infos

    def _main_training_loop(
        self,
        q_in: Queue,
        q_msg: Queue,
        q_out: Queue,
        shutdown_event: threading.Event,
        offline_replay_buffer: Optional[ReplayBuffer],
        replay_buffer: ReplayBuffer,
        batch_size: int,
        optimization_step: int,
        interaction_step_shift: int,
        optimizers: Dict[str, Optimizer],
    ):
        """The core online training loop with corrected UTD and target update logic."""
        cfg = self.cfg
        last_time_policy_pushed = time.time()
        online_iterator = None
        offline_iterator = None
        interaction_message = None

        if self.args.plot_buffer_stats:
            logger.info("Saving initial buffer statistics plot and summary.")
            self._save_plot_snapshot(offline_replay_buffer, replay_buffer, "initial")

        while optimization_step < cfg.policy.online_steps:
            if shutdown_event.is_set():
                logger.info("[LEARNER] Shutdown signal received. Exiting...")
                break

            process_transitions(
                q_in,
                replay_buffer,
                offline_replay_buffer,
                self.device,
                cfg.dataset.repo_id if cfg.dataset else None,
                shutdown_event,
            )

            last_msg = process_interaction_messages(q_msg, interaction_step_shift, self.wandb_logger, shutdown_event)
            if last_msg:
                interaction_message = last_msg
                self._log_evaluation_metrics(interaction_message)

            if len(replay_buffer) < cfg.policy.online_step_before_learning:
                time.sleep(0.1)
                continue

            if online_iterator is None:
                online_iterator = replay_buffer.get_iterator(batch_size, cfg.policy.async_prefetch, queue_size=2)
            if offline_replay_buffer and offline_iterator is None:
                offline_iterator = offline_replay_buffer.get_iterator(
                    batch_size, cfg.policy.async_prefetch, queue_size=2
                )

            start_time = time.time()

            # --- Start of Fix ---

            # 1. Perform `utd_ratio` critic updates for one optimization step.
            for _ in range(cfg.policy.utd_ratio):
                batch = next(online_iterator)
                if offline_iterator:
                    batch = concatenate_batch_transitions(batch, next(offline_iterator))
                critic_infos = self._update_critic(batch, optimizers)

            # 2. Conditionally perform `policy_update_freq` actor and temperature updates.
            # This happens once per optimization step, after all critic updates.
            if optimization_step % cfg.policy.policy_update_freq == 0:
                for _ in range(cfg.policy.policy_update_freq):
                    batch = next(online_iterator)
                    if offline_iterator:
                        batch = concatenate_batch_transitions(batch, next(offline_iterator))
                    actor_temp_infos = self._update_actor_and_temperature(batch, optimizers)

            # 3. Update target networks ONCE per optimization step, after all other updates.
            self.policy.update_target_networks()

            # --- End of Fix ---

            # Merge the training info from the last updates for logging.
            training_infos = {**critic_infos, **actor_temp_infos} if "actor_temp_infos" in locals() else critic_infos

            # Push updated policy to actor.
            if (
                time.time() - last_time_policy_pushed
                > cfg.policy.actor_learner_config.policy_parameters_push_frequency
            ):
                push_actor_policy_to_queue(q_out, self.policy)
                last_time_policy_pushed = time.time()

            freq = 1 / (time.time() - start_time + 1e-9)
            if optimization_step % cfg.log_freq == 0:
                logger.info(
                    f"[LEARNER] Step: {optimization_step} | Freq: {freq:.2f}Hz | "
                    f"loss_critic: {training_infos.get('loss_critic', float('nan')):.4f}"
                )
                log_dict = {
                    **training_infos,
                    "online/optimization_frequency_hz": freq,
                    "online/optimization_step": optimization_step,
                }
                self._log_training_metrics(log_dict, optimization_step)

                if self.args.plot_buffer_stats:
                    self._save_plot_snapshot(offline_replay_buffer, replay_buffer, "live")

            # 4. Increment optimization_step AFTER all updates for this step are complete.
            optimization_step += 1

            if cfg.save_checkpoint and (
                optimization_step % cfg.save_freq == 0 or optimization_step >= cfg.policy.online_steps
            ):
                self._save_checkpoint(
                    optimization_step,
                    interaction_message,
                    optimizers,
                    replay_buffer,
                    offline_replay_buffer,
                )

        return optimization_step, interaction_message

    def _log_training_metrics(self, log_dict: Dict, step: int):
        """Logs training metrics to WandB or TensorBoard."""
        if self.wandb_logger:
            self.wandb_logger.log_dict(
                log_dict, mode="train", custom_step_key="online/optimization_step"
            )
        elif self.tb_logger:
            for k, v in log_dict.items():
                if k != "online/optimization_step":
                    self.tb_logger.add_scalar(f"train/{k}", v, step)

    def _save_checkpoint(
        self,
        optimization_step: int,
        interaction_message: Optional[Dict],
        optimizers: Dict,
        replay_buffer: ReplayBuffer,
        offline_replay_buffer: Optional[ReplayBuffer],
    ):
        """Saves a training checkpoint."""
        save_training_checkpoint(
            self.cfg,
            optimization_step,
            self.cfg.policy.online_steps,
            interaction_message,
            self.policy,
            optimizers,
            replay_buffer,
            offline_replay_buffer,
            self.cfg.dataset.repo_id if self.cfg.dataset else None,
            self.cfg.env.fps if self.cfg.env else 30,
        )

    def _save_plot_snapshot(
        self,
        offline_buffer: Optional[ReplayBuffer],
        online_buffer: ReplayBuffer,
        prefix: str,
    ):
        """
        Calculates buffer stats for states, actions, rewards, and images,
        then saves plots to PNG files and a summary text file. For action
        and state vectors, it separates the last dimension (e.g., gripper)
        into its own plot.
        """
        if not self.args.plot_buffer_stats:
            return

        output_dir = Path(self.cfg.output_dir) / "buffer_stats_snapshots"
        output_dir.mkdir(parents=True, exist_ok=True)

        buffers = {"offline": offline_buffer, "online": online_buffer}
        stats = {}

        for name, buffer in buffers.items():
            if buffer is None or len(buffer) == 0:
                continue
            stats[name] = {}

            # Process all state keys (including images)
            for key in buffer.states.keys():
                data = buffer.states[key][: len(buffer)].cpu().numpy()
                if "image" in key:
                    # For images, calculate per-channel statistics
                    stats[name][key] = {
                        "mean": np.mean(data, axis=(0, 2, 3)),
                        "min": np.min(data, axis=(0, 2, 3)),
                        "max": np.max(data, axis=(0, 2, 3)),
                    }
                else:
                    # For other state vectors, calculate per-dimension statistics
                    stats[name][key] = {
                        "mean": np.mean(data, axis=0),
                        "min": np.min(data, axis=0),
                        "max": np.max(data, axis=0),
                    }

            # Process all next_state keys if not memory optimized
            # This avoids redundant calculations if next_states is just a reference to states
            if not buffer.optimize_memory:
                for key in buffer.next_states.keys():
                    data = buffer.next_states[key][: len(buffer)].cpu().numpy()
                    # Use a distinct key for next_state features to avoid overwriting state stats
                    feature_key = f"next_{key}"
                    if "image" in key:
                        stats[name][feature_key] = {
                            "mean": np.mean(data, axis=(0, 2, 3)),
                            "min": np.min(data, axis=(0, 2, 3)),
                            "max": np.max(data, axis=(0, 2, 3)),
                        }
                    else:
                        stats[name][feature_key] = {
                            "mean": np.mean(data, axis=0),
                            "min": np.min(data, axis=0),
                            "max": np.max(data, axis=0),
                        }

            # Process actions and rewards
            if len(buffer) > 0:
                # Actions
                actions_data = buffer.actions[: len(buffer)].cpu().numpy()
                stats[name]["action"] = {
                    "mean": np.mean(actions_data, axis=0),
                    "min": np.min(actions_data, axis=0),
                    "max": np.max(actions_data, axis=0),
                }
                # Rewards
                rewards_data = buffer.rewards[: len(buffer)].cpu().numpy()
                stats[name]["reward"] = {
                    "mean": np.mean(rewards_data),
                    "min": np.min(rewards_data),
                    "max": np.max(rewards_data),
                }
                # Dones
                dones_data = buffer.dones[: len(buffer)].cpu().numpy().astype(float)
                stats[name]["dones"] = {
                    "mean": np.mean(dones_data),
                    "min": np.min(dones_data),
                    "max": np.max(dones_data),
                }
                # Truncateds
                truncateds_data = buffer.truncateds[: len(buffer)].cpu().numpy().astype(float)
                stats[name]["truncateds"] = {
                    "mean": np.mean(truncateds_data),
                    "min": np.min(truncateds_data),
                    "max": np.max(truncateds_data),
                }

        if not stats:
            return  # No data to plot or save

        # --- Save Summary Text File ---
        summary_file_path = output_dir / f"{prefix}_buffer_stats_summary.txt"
        try:
            with open(summary_file_path, "w") as f:
                f.write(f"--- Buffer Statistics Summary ({prefix.capitalize()}) ---\n\n")
                for buffer_name, buffer_stats in stats.items():
                    f.write(f"===== {buffer_name.upper()} BUFFER =====\n")
                    for feature, feature_stats in buffer_stats.items():
                        f.write(f"\n  --- Feature: {feature} ---\n")
                        for stat_name, values in feature_stats.items():
                            f.write(f"    {stat_name.capitalize()}:\n")
                            if hasattr(values, "__iter__"):
                                formatted_values = ", ".join(
                                    f"{v:.4f}" for v in values
                                )
                                f.write(f"      [{formatted_values}]\n")
                            else:
                                f.write(f"      {values:.4f}\n")
                    f.write("\n")
            logger.info(f"Saved buffer statistics summary to {summary_file_path}")
        except Exception as e:
            logger.error(f"Failed to write buffer stats summary: {e}")

        # --- Generate and Save Plots ---
        features_to_plot = {
            key for buffer_stats in stats.values() for key in buffer_stats.keys()
        }

        for feature in features_to_plot:
            # Identify special features
            is_image = "image" in feature
            is_action = feature == "action"
            is_reward = feature == "reward"
            is_done = feature == "dones"
            is_truncated = feature == "truncateds"

            # Assume anything else is a state vector
            is_observation_state_vector = (
                not is_image and not is_action and not is_reward and not is_done and not is_truncated
            )

            # Check if the vector is multi-dimensional and should be split
            is_splittable = False
            if is_action or is_observation_state_vector:
                # Check across both buffers if the feature exists and has length > 1
                if any(
                    feature in stats.get(b, {})
                    and hasattr(stats[b][feature]["mean"], "__len__")
                    and len(stats[b][feature]["mean"]) > 1
                    for b in ["offline", "online"]
                ):
                    is_splittable = True

            if is_splittable:
                # Plot 1: Main dimensions (all but the last)
                self._create_bar_plot(
                    stats,
                    feature,
                    "main_dims",
                    prefix,
                    output_dir,
                    slice(None, -1),
                )
                # Plot 2: Last dimension (gripper)
                self._create_bar_plot(
                    stats, feature, "last_dim", prefix, output_dir, -1
                )
            else:
                # Plot everything together (scalars, images, or single-dim vectors)
                self._create_bar_plot(
                    stats, feature, "all", prefix, output_dir, None
                )

    def _create_bar_plot(
        self,
        stats: Dict,
        feature: str,
        plot_suffix: str,
        file_prefix: str,
        output_dir: Path,
        data_slice: Optional[Any] = None,
    ):
        """Helper to create and save a single bar plot for buffer statistics."""
        fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        title_feature = (
            f"{feature}_{plot_suffix}" if plot_suffix != "all" else feature
        )
        fig.suptitle(f"Buffer Statistics for: {title_feature}", fontsize=16)

        labels, x = None, None
        width = 0.35
        plot_data_exists = False

        if "offline" in stats and feature in stats["offline"]:
            original_data = stats["offline"][feature]
            if data_slice is not None:
                sliced_data = {k: v[data_slice] for k, v in original_data.items()}
                # Ensure it's still a 1D array for single-element slices
                if isinstance(data_slice, int):
                    sliced_data = {k: np.array([v]) for k, v in sliced_data.items()}
            else:
                sliced_data = original_data

            # Handle scalar values which don't have a len()
            mean_val = sliced_data["mean"]
            if not hasattr(mean_val, "__len__"):
                num_dims = 1
                sliced_data = {k: np.array([v]) for k, v in sliced_data.items()}
            else:
                num_dims = len(mean_val)

            if num_dims > 0:
                plot_data_exists = True
                orig_mean_val = original_data["mean"]
                if not hasattr(orig_mean_val, "__len__"):
                    total_dims = 1
                else:
                    total_dims = len(orig_mean_val)

                if plot_suffix == "last_dim":
                    labels = [f"dim_{total_dims - 1} (gripper)"]
                elif feature in ["reward", "dones", "truncateds"]:
                    labels = [feature]
                elif "image" in feature:
                    labels = [f"channel_{i}" for i in range(num_dims)]
                else:  # main_dims or other vectors
                    labels = [f"dim_{i}" for i in range(num_dims)]

                x = np.arange(len(labels))
                ax[0].bar(
                    x - width / 2, sliced_data["mean"], width, label="Offline Mean"
                )
                ax[1].bar(
                    x - width / 2, sliced_data["min"], width, label="Offline Min"
                )
                ax[2].bar(
                    x - width / 2, sliced_data["max"], width, label="Offline Max"
                )

        if "online" in stats and feature in stats["online"]:
            original_data = stats["online"][feature]
            if data_slice is not None:
                sliced_data = {k: v[data_slice] for k, v in original_data.items()}
                if isinstance(data_slice, int):
                    sliced_data = {k: np.array([v]) for k, v in sliced_data.items()}
            else:
                sliced_data = original_data

            # Handle scalar values which don't have a len()
            mean_val = sliced_data["mean"]
            if not hasattr(mean_val, "__len__"):
                num_dims = 1
                sliced_data = {k: np.array([v]) for k, v in sliced_data.items()}
            else:
                num_dims = len(mean_val)

            if num_dims > 0:
                plot_data_exists = True
                if labels is None:
                    orig_mean_val = original_data["mean"]
                    if not hasattr(orig_mean_val, "__len__"):
                        total_dims = 1
                    else:
                        total_dims = len(orig_mean_val)

                    if plot_suffix == "last_dim":
                        labels = [f"dim_{total_dims - 1} (gripper)"]
                    elif feature in ["reward", "dones", "truncateds"]:
                        labels = [feature]
                    elif "image" in feature:
                        labels = [f"channel_{i}" for i in range(num_dims)]
                    else:
                        labels = [f"dim_{i}" for i in range(num_dims)]
                    x = np.arange(len(labels))

                ax[0].bar(
                    x + width / 2, sliced_data["mean"], width, label="Online Mean"
                )
                ax[1].bar(
                    x + width / 2, sliced_data["min"], width, label="Online Min"
                )
                ax[2].bar(
                    x + width / 2, sliced_data["max"], width, label="Online Max"
                )

        if not plot_data_exists:
            plt.close(fig)
            return

        ax[0].set_ylabel("Mean")
        ax[0].legend()
        ax[0].grid(True)

        ax[1].set_ylabel("Min")
        ax[1].legend()
        ax[1].grid(True)

        ax[2].set_ylabel("Max")
        ax[2].legend()
        ax[2].grid(True)

        if x is not None:
            plt.xticks(x, labels, rotation=45, ha="right")

        plt.xlabel("Dimension")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_dir / f"{file_prefix}_{title_feature}_stats.png")
        plt.close(fig)

    def _cleanup_resources(
        self,
        learner_process,
        shutdown_event,
        q_in,
        q_msg,
        q_out,
        optimization_step,
        interaction_message,
        optimizers,
        replay_buffer,
        offline_replay_buffer,
    ):
        """Cleans up processes, queues, and saves a final checkpoint."""
        logger.info("Training stopped. Cleaning up resources.")
        shutdown_event.set()

        # Save one final plot snapshot.
        if self.args.plot_buffer_stats:
            self._save_plot_snapshot(
                offline_replay_buffer, replay_buffer, "final"
            )

        if self.cfg.save_checkpoint and replay_buffer and len(replay_buffer) > 0:
            logger.info("Saving final checkpoint.")
            self._save_checkpoint(
                optimization_step,
                interaction_message,
                optimizers,
                replay_buffer,
                offline_replay_buffer,
            )
        else:
            logger.warning(
                "Replay buffer is empty or checkpointing is disabled. Skipping final save."
            )

        with contextlib.suppress(Exception):
            learner_process.join(timeout=5)
            q_in.close()
            q_msg.close()
            q_out.close()
        if self.tb_logger:
            self.tb_logger.close()
        logger.info("Cleanup complete.")

    def train_loop(self):
        """
        Main entry point for the training process.

        Can run either the custom modular training loop or the original
        `add_actor_information_and_train` from `lerobot.rl.learner` based on the
        `--use_lerobot_learner` command-line flag.
        """
        self._setup_logging()

        # Common setup for both training loops: communication process
        shutdown_event = ProcessSignalHandler(use_threads=use_threads(self.cfg)).shutdown_event
        q_in, q_msg, q_out = Queue(), Queue(), Queue()
        process_class = (
            threading.Thread if use_threads(self.cfg) else torch.multiprocessing.Process
        )
        learner_process = process_class(
            target=start_learner,
            args=(q_out, q_in, q_msg, shutdown_event, self.cfg),
            daemon=True,
        )
        learner_process.start()
        logger.info("Learner communication process started.")

        if self.args.use_lerobot_learner:
            # OPTION 1: Use the original learner from the `lerobot` library.
            # This learner handles its own policy, buffer, and optimizer initialization.
            logger.info(
                "Starting training with `add_actor_information_and_train` from `lerobot.rl.learner`."
            )
            try:
                add_actor_information_and_train(
                    cfg=self.cfg,
                    wandb_logger=self.wandb_logger,
                    shutdown_event=shutdown_event,
                    transition_queue=q_in,
                    interaction_message_queue=q_msg,
                    parameters_queue=q_out,
                )
            except (KeyboardInterrupt, Exception) as e:
                if isinstance(e, KeyboardInterrupt):
                    logger.info("Ctrl+C detected! Cleaning up...")
                else:
                    logger.exception(f"Unexpected error in training loop: {e}")
            finally:
                # Simplified cleanup: just stop the communication process.
                # The `add_actor_information_and_train` handles its own checkpointing.
                logger.info("Training stopped. Cleaning up communication resources.")
                shutdown_event.set()
                with contextlib.suppress(Exception):
                    learner_process.join(timeout=5)
                    q_in.close()
                    q_msg.close()
                    q_out.close()
                if self.tb_logger:
                    self.tb_logger.close()
                logger.info("Cleanup complete.")
        else:
            # OPTION 2: Use the modular training loop defined in this class.
            # This loop uses helper methods for initialization and updates.
            logger.info("Starting training with the custom modular training loop.")
            offline_replay_buffer, replay_buffer, batch_size = self._initialize_buffers()
            (
                optimization_step,
                interaction_step_shift,
                optimizers,
                _,
            ) = self._initialize_training_state()

            push_actor_policy_to_queue(q_out, self.policy)
            interaction_message = None

            try:
                optimization_step, interaction_message = self._main_training_loop(
                    q_in,
                    q_msg,
                    q_out,
                    shutdown_event,
                    offline_replay_buffer,
                    replay_buffer,
                    batch_size,
                    optimization_step,
                    interaction_step_shift,
                    optimizers,
                )
            except (KeyboardInterrupt, Exception) as e:
                if isinstance(e, KeyboardInterrupt):
                    logger.info("Ctrl+C detected! Cleaning up...")
                else:
                    logger.exception(f"Unexpected error in training loop: {e}")
            finally:
                # Full cleanup, including saving a final checkpoint.
                self._cleanup_resources(
                    learner_process,
                    shutdown_event,
                    q_in,
                    q_msg,
                    q_out,
                    optimization_step,
                    interaction_message,
                    optimizers,
                    replay_buffer,
                    offline_replay_buffer,
                )


if __name__ == "__main__":
    trainer = TrainLerobotHilSerl()
    trainer.run()
