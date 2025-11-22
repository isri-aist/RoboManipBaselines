# ruff: noqa: E501
"""
This script provides an improved training pipeline for ACFQL, integrating
`robo-manip-baselines` with the `lerobot` distributed learning framework.

Key Improvements:
- Modular Training Loop: The main `train_loop` has been broken down into smaller,
  more manageable functions for initialization, the core training cycle, and cleanup.
- Enhanced Readability: Logic for critic and actor updates is encapsulated in separate
  helper methods, making the main loop cleaner.
- Robust Error Handling: A comprehensive try/except/finally block ensures
  graceful shutdown and resource cleanup, even in case of errors.
- Clearer Logging: TensorBoard and WandB logging are centralized, and evaluation
  metrics are consistently logged.
- Refined Imports and Structure: Imports are logically grouped, and helper
  functions from `lerobot.rl.learner` are leveraged more effectively.
- Buffer Statistics Plotting: Added functionality to save buffer statistics plots
  and summaries during training for better debugging and analysis.
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

try:
    ROBOMANIP_PATH = Path(__file__).resolve().parents[4]
    sys.path.append(str(ROBOMANIP_PATH))
    LEROBOT_PATH = ROBOMANIP_PATH / "third_party" / "lerobot" / "src"
    sys.path.append(str(LEROBOT_PATH))
except (NameError, IndexError):
    print("Warning: Could not determine ROBOMANIP_PATH. Assuming paths are set.")

# LeRobot Imports
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.learner import (
    check_nan_in_transition,
    handle_resume_logic,
    load_training_state,
    log_training_info,
    process_interaction_messages,
    start_learner,
    use_threads,
)
from lerobot.rl.process import ProcessSignalHandler
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.transport.utils import bytes_to_transitions
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device
from lerobot.processor import TransitionKey

# RoboManipBaselines Imports
from robo_manip_baselines.common.base.CroppedTrainBase import CroppedTrainBase
from robo_manip_baselines.policy.lerobot_acfql.AcfqlDataset import AcfqlDataset
from robo_manip_baselines.policy.lerobot_acfql.acfql_rl.buffer import ReplayBufferNSteps
from robo_manip_baselines.policy.lerobot_acfql.acfql_rl.learner import (
    push_actor_policy_to_queue,
    save_training_checkpoint,
)
from robo_manip_baselines.policy.lerobot_acfql.common import (
    CommonLerobotAcfqlBase,
    create_hydrated_lerobot_config,
    setup_policy_and_processors,
)
from dataclasses import is_dataclass, asdict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True
)
logger = logging.getLogger(__name__)

def _clean_for_json(data):
    """Recursively convert dataclasses, numpy/tensor types, and other non-JSON types."""
    if is_dataclass(data):
        data = asdict(data)
    if hasattr(data, "__dict__") and not isinstance(data, (torch.nn.Module, np.ndarray)):
        data = vars(data)
    if isinstance(data, dict):
        return {k: _clean_for_json(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple, set)):
        return [_clean_for_json(v) for v in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().tolist()
    elif isinstance(data, (np.integer, np.floating)):
        return data.item()
    else:
        return data

class TrainLerobotAcfql(CroppedTrainBase, CommonLerobotAcfqlBase):
    """Training script for ACFQL, structured similarly to the HIL-SERL trainer."""

    DatasetClass = AcfqlDataset

    def set_additional_args(self, parser):
        super().set_additional_args(parser)
        self.set_common_args(parser)
        parser.add_argument("--data_config_path", type=str, required=True)
        parser.add_argument("--reset_while_resuming", action="store_true")
        parser.add_argument(
            "--plot_buffer_stats",
            action="store_true",
            help="Periodically save buffer statistics plots and a summary text file during training."
        )

    def setup_policy(self):
        """Loads the policy, configuration, and processors."""
        self.device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu", log=True)
        if self.args.reset_while_resuming:
            self.args.resume = True
        self.cfg, self.lerobot_dataset = create_hydrated_lerobot_config(
            policy_config_path=self.args.policy_config_path,
            model_meta_info=self.model_meta_info,
            camera_names=self.camera_names,
            camera_resolution=self.target_camera_resolution or self.camera_resolution,
            output_dir=self.args.checkpoint_dir,
            resume=self.args.resume,
        )
        self.policy, self.preprocessor, self.postprocessor = setup_policy_and_processors(
            cfg=self.cfg, lerobot_dataset=self.lerobot_dataset, device=self.device
        )
        set_seed(self.cfg.seed)
        self.cfg.policy.dataset_stats = {} #saving the dataset_stats causes serialisation troubles so we remove them after creating the policy
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

    def make_optimizers(self) -> Dict[str, Optimizer]:
        """Creates optimizers for all policy components."""
        optimizer_params = self.policy.get_optim_params()
        return {
            "critic": torch.optim.Adam(optimizer_params["critic"], lr=self.cfg.policy.critic_lr),
            "actor_bc_flow": torch.optim.Adam(
                optimizer_params["actor_bc_flow"], lr=self.cfg.policy.actor_lr
            ),
            "actor_onestep_flow": torch.optim.Adam(
                optimizer_params["actor_onestep_flow"], lr=self.cfg.policy.actor_lr
            ),
        }

    def _initialize_buffers(self) -> Tuple[Optional[ReplayBufferNSteps], ReplayBufferNSteps, int]:
        """Initializes offline and online replay buffers."""
        cfg = self.cfg
        offline_buffer = None
        batch_size = cfg.batch_size
        if cfg.dataset and cfg.policy.offline_buffer_capacity > 0:
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
                offline_buffer = ReplayBufferNSteps.from_lerobot_dataset(
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
        online_buffer = ReplayBufferNSteps(
            capacity=cfg.policy.online_buffer_capacity,
            device=self.device,
            state_keys=list(cfg.policy.input_features.keys()),
            storage_device=cfg.policy.storage_device,
            optimize_memory=True,
        )
        return offline_buffer, online_buffer, batch_size

    def _initialize_training_state(self) -> Tuple[int, int, Dict[str, Optimizer]]:
        """Loads training state if resuming, otherwise initializes it."""
        self.cfg = handle_resume_logic(self.cfg)
        self.policy.train()
        optimizers = self.make_optimizers()
        opt_step, resume_interaction_step = load_training_state(cfg=self.cfg, optimizers=optimizers)
        if self.args.reset_while_resuming and self.args.resume:
            logger.info("Resetting training state due to --reset_while_resuming flag.")
            opt_step, resume_interaction_step = 0, 0
            optimizers = self.make_optimizers()
        log_training_info(cfg=self.cfg, policy=self.policy)
        return opt_step or 0, resume_interaction_step or 0, optimizers

    def _setup_logging(self) -> None:
        """Initializes WandB or TensorBoard loggers."""
        self.wandb_logger = WandBLogger(self.cfg) if self.cfg.wandb.enable else None
        self.tb_logger = (
            None
            if self.wandb_logger
            else SummaryWriter(log_dir=Path(self.cfg.output_dir) / "tensorboard_logs")
        )

    def _prepare_batch(self, batch: Dict) -> Dict:
        """Prepares a raw batch from the buffer for policy forward pass."""
        transition_for_preproc = {
            "observation": batch.pop("state"),
            "next_observation": batch.pop("next_state"),
            "action": batch["action"],
        }
        transition_for_preproc[TransitionKey.COMPLEMENTARY_DATA] = None
        processed_transition = self.preprocessor._forward(transition_for_preproc)
        processed_batch = batch
        processed_batch["state"] = processed_transition.pop("observation")
        processed_batch["next_state"] = processed_transition.pop("next_observation")
        processed_batch["action"] = processed_transition.pop("action")
        all_features = list(self.cfg.policy.input_features.keys())
        processed_batch["state"] = {k: v for k, v in processed_batch["state"].items() if k in all_features}
        processed_batch["next_state"] = {
            k: v for k, v in processed_batch["next_state"].items() if k in all_features
        }
        check_nan_in_transition(
            observations=processed_batch["state"],
            actions=processed_batch["action"],
            next_state=processed_batch["next_state"],
        )
        if "mask" not in processed_batch:
            if "masks" in processed_batch:
                processed_batch["mask"] = processed_batch["masks"]
            else:
                batch_size = None
                if isinstance(processed_batch.get("state"), dict) and len(processed_batch["state"]) > 0:
                    first_tensor = next(iter(processed_batch["state"].values()))
                    batch_size = first_tensor.shape[0]
                else:
                    batch_size = 1
                processed_batch["mask"] = torch.ones((batch_size,), dtype=torch.bool, device=self.device)
        if "valid" not in processed_batch:
            processed_batch["valid"] = torch.ones_like(processed_batch["mask"], dtype=torch.bool, device=self.device)
        return processed_batch

    def _update_critic(self, batch: Dict, optimizers: Dict[str, Optimizer]) -> Dict[str, Any]:
        """Performs a single update step for the critic networks and returns training info."""
        check_nan_in_transition(
            observations=batch["state"],
            actions=batch["action"],
            next_state=batch["next_state"],
        )
        critic_output = self.policy.forward(batch, model="critic")
        optimizers["critic"].zero_grad(set_to_none=True)
        critic_output["loss_critic"].backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.critic_ensemble.parameters(), self.cfg.policy.grad_clip_norm
        )
        optimizers["critic"].step()
        training_infos = {
            **{f"critic/{k}": v.item() for k, v in critic_output["info"].items()},
            "critic_grad_norm": critic_grad_norm.item(),
        }
        return training_infos

    def _update_actor(self, batch: Dict, optimizers: Dict[str, Optimizer]) -> Dict[str, Any]:
        """Performs actor updates for both bc_flow and onestep_flow, returning training info."""
        training_infos = {}
        if self.cfg.policy.policy_update_freq == 0:
            return training_infos
        actor_bc_output = self.policy.forward(batch, model="actor_bc_flow")
        optimizers["actor_bc_flow"].zero_grad()
        actor_bc_output["loss_actor_bc_flow"].backward()
        actor_bc_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.actor_bc_flow.parameters(), self.cfg.policy.grad_clip_norm
        )
        optimizers["actor_bc_flow"].step()
        training_infos.update({f"actor_bc/{k}": v.item() for k, v in actor_bc_output["info"].items()})
        training_infos["actor_bc_grad_norm"] = actor_bc_grad_norm.item()
        actor_onestep_output = self.policy.forward(batch, model="actor_onestep_flow")
        optimizers["actor_onestep_flow"].zero_grad()
        actor_onestep_output["loss_actor_onestep_flow"].backward()
        actor_onestep_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.actor_onestep_flow.parameters(), self.cfg.policy.grad_clip_norm
        )
        optimizers["actor_onestep_flow"].step()
        training_infos.update(
            {f"actor_one/{k}": v.item() for k, v in actor_onestep_output["info"].items()}
        )
        training_infos["actor_one_grad_norm"] = actor_onestep_grad_norm.item()
        return training_infos

    def _main_training_loop(
        self,
        q_in: Queue,
        q_msg: Queue,
        q_out: Queue,
        shutdown_event: threading.Event,
        offline_buffer: Optional[ReplayBufferNSteps],
        online_buffer: ReplayBufferNSteps,
        batch_size: int,
        opt_step: int,
        interaction_step_shift: int,
        optimizers: Dict[str, Optimizer],
    ):
        cfg = self.cfg
        offline_steps = getattr(cfg.policy, "offline_steps", 0)
        online_steps = cfg.policy.online_steps
        utd_ratio = cfg.policy.utd_ratio
        interaction_message = None

        if self.args.plot_buffer_stats:
            logger.info("Saving initial buffer statistics plot and summary.")
            self._save_plot_snapshot(offline_buffer, online_buffer, "initial")

        if offline_buffer and opt_step < offline_steps:
            logger.info(f"Starting offline training phase for {offline_steps} steps.")
            iterator = offline_buffer.get_iterator_nstep(
                batch_size, cfg.policy.chunk_size, cfg.policy.discount, cfg.policy.async_prefetch
            )
            for step in tqdm(range(opt_step, offline_steps), desc="Offline Training"):
                if shutdown_event.is_set():
                    break
                
                # UTD loop for critic-only updates
                for _ in range(utd_ratio - 1):
                    critic_only_batch = self._prepare_batch(next(iterator))
                    self._update_critic(critic_only_batch, optimizers)
                    self.policy.update_target_networks()

                prepared_batch = self._prepare_batch(next(iterator))
                critic_infos = self._update_critic(prepared_batch, optimizers)
                
                actor_infos = {}
                if step % cfg.policy.policy_update_freq == 0:
                    actor_infos = self._update_actor(prepared_batch, optimizers)

                self.policy.update_target_networks()
                
                infos = {**critic_infos, **actor_infos}
                if step % cfg.log_freq == 0:
                    logger.info(
                        f"[LEARNER] Offline Step: {step} | "
                        f"loss_critic: {infos.get('critic/critic_loss', float('nan')):.4f}"
                    )
                    log_dict = {**infos, "offline/optimization_step": step}
                    self._log_training_metrics(log_dict, step)
                    if self.args.plot_buffer_stats:
                        self._save_plot_snapshot(offline_buffer, online_buffer, "live")
                opt_step += 1

        logger.info("Starting online training phase.")
        push_actor_policy_to_queue(q_out, self.policy)
        last_policy_push = time.time()
        online_iterator = None

        while opt_step < online_steps + offline_steps:
            if shutdown_event.is_set():
                break
            
            while not q_in.empty():
                raw_data = q_in.get()
                try:
                    transitions = bytes_to_transitions(raw_data)
                    for transition in transitions:
                        online_buffer.add(**transition)
                except Exception as e:
                    logger.warning(f"Failed to process transition data from queue: {e}")
            
            last_msg = process_interaction_messages(
                q_msg, interaction_step_shift, self.wandb_logger, shutdown_event
            )
            if last_msg:
                interaction_message = last_msg
                self._log_evaluation_metrics(interaction_message)
            
            if len(online_buffer) < cfg.policy.online_step_before_learning:
                time.sleep(0.1)
                continue
            
            if online_iterator is None:
                online_iterator = online_buffer.get_iterator_nstep(
                    batch_size, cfg.policy.chunk_size, cfg.policy.discount, cfg.policy.async_prefetch
                )
            
            start_time = time.time()
            
            # UTD loop for critic-only updates
            for _ in range(utd_ratio - 1):
                critic_only_batch = self._prepare_batch(next(online_iterator))
                self._update_critic(critic_only_batch, optimizers)
                self.policy.update_target_networks()

            prepared_batch = self._prepare_batch(next(online_iterator))
            critic_infos = self._update_critic(prepared_batch, optimizers)
            
            actor_infos = {}
            if opt_step % cfg.policy.policy_update_freq == 0:
                actor_infos = self._update_actor(prepared_batch, optimizers)
            
            self.policy.update_target_networks()
            
            infos = {**critic_infos, **actor_infos}
            freq = 1 / (time.time() - start_time + 1e-9)
            
            if opt_step % cfg.log_freq == 0:
                logger.info(
                    f"[LEARNER] Online Step: {opt_step} | Freq: {freq:.2f}Hz | "
                    f"loss_critic: {infos.get('critic/critic_loss', float('nan')):.4f}"
                )
                log_dict = {
                    **infos,
                    "online/optimization_frequency_hz": freq,
                    "online/optimization_step": opt_step,
                }
                self._log_training_metrics(log_dict, opt_step)
                if self.args.plot_buffer_stats:
                    self._save_plot_snapshot(offline_buffer, online_buffer, "live")
            
            if (
                time.time() - last_policy_push
                > cfg.policy.actor_learner_config.policy_parameters_push_frequency
            ):
                push_actor_policy_to_queue(q_out, self.policy)
                last_policy_push = time.time()
            
            if cfg.save_checkpoint and (opt_step % cfg.save_freq == 0) and opt_step > 0:
                self._save_checkpoint(
                    opt_step, interaction_message, optimizers, online_buffer, offline_buffer
                )
            
            opt_step += 1
            
        return opt_step, interaction_message

    def _log_training_metrics(self, log_dict: Dict, step: int):
        """Logs training metrics to WandB or TensorBoard."""
        if self.wandb_logger:
            step_key = "online/optimization_step" if "online/optimization_step" in log_dict else "offline/optimization_step"
            self.wandb_logger.log_dict(log_dict, mode="train", custom_step_key=step_key)
        elif self.tb_logger:
            for k, v in log_dict.items():
                if "optimization_step" not in k:
                    self.tb_logger.add_scalar(f"train/{k}", v, step)

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
                self.tb_logger.add_scalar(f"eval/{key}", interaction_message[key], step)

    def _save_checkpoint(self, opt_step, interaction_msg, optimizers, online_buf, offline_buf):
        """Saves a training checkpoint."""
        save_training_checkpoint(
            self.cfg,
            opt_step,
            self.cfg.policy.online_steps,
            interaction_msg,
            self.policy,
            optimizers,
            online_buf,
            offline_buf,
            self.cfg.dataset.repo_id if self.cfg.dataset else None,
            self.cfg.env.fps if self.cfg.env else 30,
            self.preprocessor,
            self.postprocessor,
        )

    def _save_plot_snapshot(
        self,
        offline_buffer: Optional[ReplayBufferNSteps],
        online_buffer: ReplayBufferNSteps,
        prefix: str,
    ):
        """Calculates buffer stats for states, actions, rewards, and images, then saves plots to PNG files and a summary text file."""
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
            for key in buffer.states.keys():
                data = buffer.states[key][: len(buffer)].cpu().numpy()
                if "image" in key:
                    stats[name][key] = {
                        "mean": np.mean(data, axis=(0, 2, 3)),
                        "min": np.min(data, axis=(0, 2, 3)),
                        "max": np.max(data, axis=(0, 2, 3)),
                    }
                else:
                    stats[name][key] = {
                        "mean": np.mean(data, axis=0),
                        "min": np.min(data, axis=0),
                        "max": np.max(data, axis=0),
                    }
            if not buffer.optimize_memory:
                for key in buffer.next_states.keys():
                    data = buffer.next_states[key][: len(buffer)].cpu().numpy()
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
            if len(buffer) > 0:
                actions_data = buffer.actions[: len(buffer)].cpu().numpy()
                stats[name]["action"] = {
                    "mean": np.mean(actions_data, axis=0),
                    "min": np.min(actions_data, axis=0),
                    "max": np.max(actions_data, axis=0),
                }
                rewards_data = buffer.rewards[: len(buffer)].cpu().numpy()
                stats[name]["reward"] = {
                    "mean": np.mean(rewards_data),
                    "min": np.min(rewards_data),
                    "max": np.max(rewards_data),
                }
                dones_data = buffer.dones[: len(buffer)].cpu().numpy().astype(float)
                stats[name]["dones"] = {
                    "mean": np.mean(dones_data),
                    "min": np.min(dones_data),
                    "max": np.max(dones_data),
                }
                truncateds_data = buffer.truncateds[: len(buffer)].cpu().numpy().astype(float)
                stats[name]["truncateds"] = {
                    "mean": np.mean(truncateds_data),
                    "min": np.min(truncateds_data),
                    "max": np.max(truncateds_data),
                }
        if not stats:
            return
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
                                formatted_values = ", ".join(f"{v:.4f}" for v in values)
                                f.write(f"      [{formatted_values}]\n")
                            else:
                                f.write(f"      {values:.4f}\n")
                    f.write("\n")
            logger.info(f"Saved buffer statistics summary to {summary_file_path}")
        except Exception as e:
            logger.error(f"Failed to write buffer stats summary: {e}")
        features_to_plot = {key for buffer_stats in stats.values() for key in buffer_stats.keys()}
        for feature in features_to_plot:
            is_image = "image" in feature
            is_action = feature == "action"
            is_reward = feature == "reward"
            is_done = feature == "dones"
            is_truncated = feature == "truncateds"
            is_observation_state_vector = (
                not is_image and not is_action and not is_reward and not is_done and not is_truncated
            )
            is_splittable = False
            if is_action or is_observation_state_vector:
                if any(
                    feature in stats.get(b, {})
                    and hasattr(stats[b][feature]["mean"], "__len__")
                    and len(stats[b][feature]["mean"]) > 1
                    for b in ["offline", "online"]
                ):
                    is_splittable = True
            if is_splittable:
                self._create_bar_plot(stats, feature, "main_dims", prefix, output_dir, slice(None, -1))
                self._create_bar_plot(stats, feature, "last_dim", prefix, output_dir, -1)
            else:
                self._create_bar_plot(stats, feature, "all", prefix, output_dir, None)

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
        title_feature = f"{feature}_{plot_suffix}" if plot_suffix != "all" else feature
        fig.suptitle(f"Buffer Statistics for: {title_feature}", fontsize=16)
        labels, x = None, None
        width = 0.35
        plot_data_exists = False
        if "offline" in stats and feature in stats["offline"]:
            original_data = stats["offline"][feature]
            if data_slice is not None:
                sliced_data = {k: v[data_slice] for k, v in original_data.items()}
                if isinstance(data_slice, int):
                    sliced_data = {k: np.array([v]) for k, v in sliced_data.items()}
            else:
                sliced_data = original_data
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
                else:
                    labels = [f"dim_{i}" for i in range(num_dims)]
                x = np.arange(len(labels))
                ax[0].bar(x - width / 2, sliced_data["mean"], width, label="Offline Mean")
                ax[1].bar(x - width / 2, sliced_data["min"], width, label="Offline Min")
                ax[2].bar(x - width / 2, sliced_data["max"], width, label="Offline Max")
        if "online" in stats and feature in stats["online"]:
            original_data = stats["online"][feature]
            if data_slice is not None:
                sliced_data = {k: v[data_slice] for k, v in original_data.items()}
                if isinstance(data_slice, int):
                    sliced_data = {k: np.array([v]) for k, v in sliced_data.items()}
            else:
                sliced_data = original_data
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
                ax[0].bar(x + width / 2, sliced_data["mean"], width, label="Online Mean")
                ax[1].bar(x + width / 2, sliced_data["min"], width, label="Online Min")
                ax[2].bar(x + width / 2, sliced_data["max"], width, label="Online Max")
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

    def _cleanup_resources(self, learner_process, shutdown_event, q_in, q_msg, q_out, *args):
        """Cleans up processes, queues, and saves a final checkpoint."""
        logger.info("Training stopped. Cleaning up resources.")
        shutdown_event.set()
        if self.args.plot_buffer_stats:
            logger.info("Saving final buffer statistics plot and summary.")
            self._save_plot_snapshot(args[3], args[4], "final")
        if self.cfg.save_checkpoint and args[3] and len(args[3]) > 0:
            logger.info("Saving final checkpoint.")
            self._save_checkpoint(*args)
        else:
            logger.warning("Online buffer is empty or checkpointing is disabled. Skipping final save.")
        with contextlib.suppress(Exception):
            learner_process.join(timeout=5)
            q_in.close()
            q_msg.close()
            q_out.close()
        if self.tb_logger:
            self.tb_logger.close()
        logger.info("Cleanup complete.")

    def train_loop(self):
        """Main entry point for the training process."""
        self._setup_logging()
        shutdown_event = ProcessSignalHandler(use_threads=use_threads(self.cfg)).shutdown_event
        q_in, q_msg, q_out = Queue(), Queue(), Queue()
        process_class = threading.Thread if use_threads(self.cfg) else torch.multiprocessing.Process
        learner_process = process_class(
            target=start_learner, args=(q_out, q_in, q_msg, shutdown_event, self.cfg), daemon=True
        )
        learner_process.start()
        logger.info("Learner communication process started.")
        offline_buffer, online_buffer, batch_size = self._initialize_buffers()
        opt_step, interaction_step_shift, optimizers = self._initialize_training_state()
        final_state = (opt_step, None, optimizers, online_buffer, offline_buffer)
        try:
            final_opt_step, final_interaction_msg = self._main_training_loop(
                q_in,
                q_msg,
                q_out,
                shutdown_event,
                offline_buffer,
                online_buffer,
                batch_size,
                opt_step,
                interaction_step_shift,
                optimizers,
            )
            final_state = (
                final_opt_step,
                final_interaction_msg,
                optimizers,
                online_buffer,
                offline_buffer,
            )
        except (KeyboardInterrupt, Exception) as e:
            if isinstance(e, KeyboardInterrupt):
                logger.info("Ctrl+C detected! Cleaning up...")
            else:
                logger.exception(f"Unexpected error in training loop: {e}")
        finally:
            self._cleanup_resources(learner_process, shutdown_event, q_in, q_msg, q_out, *final_state)

if __name__ == "__main__":
    trainer = TrainLerobotAcfql()
    trainer.run()

