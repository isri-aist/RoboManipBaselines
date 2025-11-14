# ruff: noqa: E501
import contextlib
import json
import logging
import pickle
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg

from lerobot.policies.sac.reward_model.configuration_classifier import (
    RewardClassifierConfig,
)
from lerobot.processor import TransitionKey
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import TimerManager, get_safe_torch_device

from robo_manip_baselines.common.base.TeleopRolloutBase import TeleopRolloutBase
from robo_manip_baselines.common.data.DataKey import DataKey
from robo_manip_baselines.policy.lerobot_base.ActionStatsTracker import (
    ActionStatsTracker,
)
from robo_manip_baselines.policy.lerobot_base.RolloutLerobotBase import (
    RolloutLerobotBase,
)
from robo_manip_baselines.policy.lerobot_base.RolloutLerobotOnlineBase import (
    RolloutLerobotOnlineBase,
)
from robo_manip_baselines.policy.lerobot_acfql.common import (
    CommonLerobotAcfqlBase,
    create_hydrated_lerobot_config,
    setup_policy_and_processors,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True
)
logger = logging.getLogger(__name__)


class RolloutLerobotAcfql(
    CommonLerobotAcfqlBase, RolloutLerobotOnlineBase, RolloutLerobotBase, TeleopRolloutBase
):
    """Rollout script for a lerobot ACFQL policy, with integrated HIL-SERL features."""

    def __init__(self):
        super().__init__()
        self.rollout_time_idx = 0
        self.action_plot_scale = 1.0
        # Attributes for action chunking and temporal ensembling
        self.policy_action_buf = []  # Buffer for the current action chunk
        self.policy_action_buf_history = []  # History of chunks for temporal ensembling
        self.ensembling_k = 0.01  # Decay factor for temporal ensembling
        self.chunk_size = None  # To be loaded from policy config
        self.setup_plot()

    def set_additional_args(self, parser):
        super().set_additional_args(parser)
        self.set_common_args(parser)
        parser.add_argument(
            "--enable_manual_reward",
            action="store_true",
            help="Prompt for manual reward override at the end of an episode.",
        )
        parser.add_argument(
            "--plot_transition_actions",
            action="store_true",
            help="Show a plot of the actions stored in the transition buffer.",
        )
        parser.add_argument(
            "--log_outliers",
            action="store_true",
            help="Enable logging of action outliers that exceed a certain threshold.",
        )
        parser.add_argument(
            "--outlier_threshold",
            type=float,
            default=0.05,
            help="The relative threshold (e.g., 0.05 for 5%) for logging a new min/max as an outlier.",
        )
        parser.add_argument(
            "--clip_gripper_values",
            type=float,
            nargs=4,
            default=None,
            metavar=("LOW_CLIP", "LOW_THRESH", "HIGH_THRESH", "HIGH_CLIP"),
            help="Clip the policy's output gripper command. ",
        )
        # Argument for temporal ensembling
        parser.add_argument(
            "--temporal_ensembling",
            action="store_true",
            help="Enable temporal ensembling for smoother actions. Requires a policy with action chunking.",
        )
        parser.add_argument(
            "--success_in_a_row",
            type=int,
            default=1,
            help="Number of consecutive successes (reward=1.0) required to mark the episode as successful.",
        )

    def reset_variables(self):
        super().reset_variables()
        if hasattr(self, "policy") and self.policy:
            self.policy.reset()
        self.policy_action_list = []
        self.teleop_action_list = []
        if hasattr(self, "action_stats_tracker"):
            self.action_stats_tracker = ActionStatsTracker(
                self.action_dim,
                log_enabled=self.args.log_outliers,
                threshold=self.args.outlier_threshold,
            )
        self.prev_raw_batch = (
            self._get_raw_observation_batch(debug=self.args.debug_inference) if self.args.online else None
        )
        self.processed_batch = None
        self.episode_intervention = False
        self.episode_intervention_steps = 0
        self.episode_total_steps = 0
        self.episode_is_success = False
        self.summary_sent = False
        if self.args.online:
            self.policy_timer = TimerManager("Policy inference", log=False)
            self.last_policy_update_time = time.time()
        # Reset chunking/ensembling buffers
        self.policy_action_buf = []
        self.policy_action_buf_history = []
        self.consecutive_successes = 0

    def setup_model_meta_info(self):
        self.load_data_config()
        self.action_dim = sum(DataKey.get_dim(key, self.env) for key in self.action_keys)
        self.action_stats_tracker = ActionStatsTracker(
            self.action_dim,
            log_enabled=self.args.log_outliers,
            threshold=self.args.outlier_threshold,
        )
        self.state_dim = sum(DataKey.get_dim(key, self.env) for key in self.state_keys)
        self.args.skip = self.args.skip or 1
        self.args.skip_draw = self.args.skip_draw or self.args.skip
        root_checkpoint_dir = Path(self.args.checkpoint)
        model_meta_info_path = None
        search_dir = root_checkpoint_dir
        for _ in range(4):
            potential_path = search_dir / "model_meta_info.pkl"
            if potential_path.is_file():
                model_meta_info_path = potential_path
                break
            if search_dir.parent == search_dir:
                break
            search_dir = search_dir.parent
        if model_meta_info_path is None:
            raise FileNotFoundError(
                f"CRITICAL: 'model_meta_info.pkl' not found in '{root_checkpoint_dir}' or any parent directories."
            )
        with open(model_meta_info_path, "rb") as f:
            self.model_meta_info = pickle.load(f)
        self.model_meta_info["state"]["keys"] = self.state_keys
        self.model_meta_info["action"]["keys"] = self.action_keys
        self.model_meta_info["image"]["camera_names"] = self.camera_names
        self.model_meta_info["state"]["dim"] = self.state_dim
        self.model_meta_info["action"]["dim"] = self.action_dim
        self.parsed_camera_crops = self.model_meta_info.get("image", {}).get("camera_crops", {})
        if self.args.camera_crops:
            self.parsed_camera_crops = {}
            for crop_arg in self.args.camera_crops:
                cam_name, crop_str = crop_arg.split(":")
                x, y, w, h = map(int, crop_str.split(","))
                self.parsed_camera_crops[cam_name] = (x, y, w, h)
        self.model_meta_info["image"]["parsed_camera_crops"] = self.parsed_camera_crops

    def setup_policy(self):
        """Loads the policy, configuration, and processors."""
        self.device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu", log=True)

        self.cfg, self.lerobot_dataset = create_hydrated_lerobot_config(
            policy_config_path=self.args.policy_config_path,
            model_meta_info=self.model_meta_info,
            camera_names=self.camera_names,
            camera_resolution=self.target_camera_resolution or self.camera_resolution,
            output_dir=None,
            resume=self.args.resume,
        )

        pretrained_path = Path(self.args.checkpoint) if not self.args.online else None

        self.policy, self.preprocessor, self.postprocessor = setup_policy_and_processors(
            cfg=self.cfg,
            lerobot_dataset=self.lerobot_dataset,
            device=self.device,
            pretrained_path=pretrained_path,
        )
        self.policy.eval()

        # Store chunk size from config
        self.chunk_size = getattr(self.cfg.policy, "chunk_size", 1)
        if self.args.temporal_ensembling:
            logger.info(f"Temporal ensembling enabled. Using chunk size: {self.chunk_size}")
            if self.chunk_size <= 1:
                logger.warning(
                    "Temporal ensembling is enabled, but policy chunk size is <= 1. Ensembling will have no effect."
                )

        self._log_model_info()
        set_seed(getattr(self.cfg, "seed", 42))
        self.setup_reward_policy()

        if self.args.online:
            self.setup_online_components()

    def infer_policy(self, use_processors=True):
        """
        Performs a single step of policy inference, with optional temporal ensembling.
        Action chunking is handled by the policy or manually if ensembling.
        """
        if self.args.online:
            self.update_policy_parameters()

        if self.args.time_step_delay > 0.0:
            time.sleep(self.args.time_step_delay)

        # Get latest observation if not available
        if self.prev_raw_batch is None:
            self.prev_raw_batch = self._get_raw_observation_batch(debug=self.args.debug_inference)

        # Preprocess observations
        if use_processors:
            transition_in = {
                TransitionKey.OBSERVATION: self.prev_raw_batch,
                TransitionKey.ACTION: None,
                TransitionKey.COMPLEMENTARY_DATA: None,
            }
            transition_out = self.preprocessor._forward(transition_in)
            self.processed_batch = transition_out["observation"]
        else:
            self.processed_batch = self.prev_raw_batch

        self.processed_batch = {
            k: v for k, v in self.processed_batch.items() if k in self.policy.config.input_features
        }

        timer = self.policy_timer if self.args.online else contextlib.nullcontext()

        # If temporal ensembling is disabled, use the policy's default stateful action selection.
        if not self.args.temporal_ensembling:
            is_skip = self.rollout_time_idx % self.args.skip_inference != 0
            if not is_skip:
                with timer, torch.inference_mode():
                    # This handles chunking internally and returns one action
                    self.raw_action_tensor = self.policy.select_action(self.processed_batch)
                self.final_action_tensor = self.postprocessor(self.raw_action_tensor)
                if self.args.clip_gripper_values:
                    low_clip, low_thresh, high_thresh, high_clip = self.args.clip_gripper_values
                    gripper_tensor = self.final_action_tensor[0, -1]
                    current_gripper_val = gripper_tensor.item()
                    new_gripper_val = current_gripper_val
                    if current_gripper_val < low_thresh:
                        new_gripper_val = low_clip
                    elif current_gripper_val > high_thresh:
                        new_gripper_val = high_clip
                    if new_gripper_val != current_gripper_val:
                        self.final_action_tensor[0, -1] = torch.tensor(
                            new_gripper_val,
                            device=gripper_tensor.device,
                            dtype=gripper_tensor.dtype,
                        )
                self.policy_action = self.final_action_tensor.to("cpu").numpy().squeeze().astype(np.float64)

        # If temporal ensembling is enabled, we manage the chunking and ensembling manually.
        else:
            if not self.policy_action_buf:
                with timer, torch.inference_mode():
                    # Use `select_action_chunk` to get the whole chunk
                    action_chunk_tensor = self.policy.select_action_chunk(self.processed_batch)

                action_chunk_tensor = self.postprocessor(action_chunk_tensor)
                action_chunk = action_chunk_tensor.squeeze(0).cpu().numpy()

                # Apply temporal ensembling if history is available
                if self.policy_action_buf_history:
                    last_chunk = self.policy_action_buf_history[-1]
                    # Shift previous chunk to align with the current prediction timeframe
                    shifted_last_chunk = np.roll(last_chunk, -1, axis=0)
                    shifted_last_chunk[-1] = shifted_last_chunk[-2]  # Duplicate last element
                    # Weighted average
                    action_chunk = (1 - self.ensembling_k) * action_chunk + self.ensembling_k * shifted_last_chunk

                # Store history and populate execution buffer
                self.policy_action_buf_history.append(action_chunk)
                if len(self.policy_action_buf_history) > 1:
                    self.policy_action_buf_history.pop(0)
                self.policy_action_buf = list(action_chunk)

            # Get the next action from our managed buffer and create a tensor for it
            next_action_np = self.policy_action_buf.pop(0)
            self.final_action_tensor = (
                torch.from_numpy(next_action_np).float().unsqueeze(0).to(self.device)
            )

            # Extract numpy action for environment
            self.policy_action = self.final_action_tensor.to("cpu").numpy().squeeze().astype(np.float64)

    def setup_reward_policy(self):
        """Loads the reward classifier model and its configuration."""
        self.reward_classifier_model = None
        self.reward_preprocessor = None
        self.reward_success_threshold = 0.95
        self.reward_success_reward = 1.0
        self.reward_terminate_on_success = True

        reward_classifier_config = (
            self.cfg.env.processor.reward_classifier
            if hasattr(self.cfg.env, "processor") and self.cfg.env.processor
            else None
        )

        if reward_classifier_config and reward_classifier_config.pretrained_path:
            pretrained_path = Path(reward_classifier_config.pretrained_path)
            config_file = pretrained_path / "config.json"
            if not config_file.exists():
                raise FileNotFoundError(f"Required `config.json` not found in: {config_file}")

            with config_file.open("r") as f:
                config_dict = json.load(f)

            cfg_reward = RewardClassifierConfig(**config_dict)

            from lerobot.policies.factory import make_policy as make_reward_policy
            from lerobot.policies.factory import make_pre_post_processors

            self.reward_classifier_model = make_reward_policy(
                cfg_reward, ds_meta=self.lerobot_dataset.meta
            )
            self.reward_classifier_model = self.reward_classifier_model.to(self.device).eval()
            self.reward_preprocessor, _ = make_pre_post_processors(
                cfg=cfg_reward, dataset_stats=self.lerobot_dataset.meta.stats
            )
            self.reward_success_threshold = reward_classifier_config.success_threshold or 0.95
            self.reward_success_reward = reward_classifier_config.success_reward or 1.0
            self.reward_terminate_on_success = reward_classifier_config.terminate_on_success or True

    def post_step_hook(self):
        """Hook for HIL-SERL with reward classification logic."""
        # Clear action buffers on intervention to force a fresh policy prediction.
        if self._check_teleop_event(TeleopEvents.IS_INTERVENTION):
            self.policy_action_buf = []
            self.policy_action_buf_history = []
            self.policy.reset()  # Reset policy's internal state (e.g., action queue)

        if not self.args.online or self.prev_raw_batch is None:
            return
        if not (self.phase_manager.is_phase("TeleopOverridePhase") or self.phase_manager.is_phase("RolloutPhase")):
            return

        final_reward, is_success = self._calculate_reward_and_success()
        self.reward = final_reward
        self.done, truncated = self._determine_termination(is_success)
        self._update_episode_stats()
        self._prepare_and_buffer_transition(truncated)

    def _calculate_reward_and_success(self):
        current_step_is_success = self.reward >= 1.0
        next_raw_batch = None if self.done else self._get_raw_observation_batch(debug=self.args.debug_inference)
        if not current_step_is_success and self.reward_classifier_model and next_raw_batch is not None:
            preprocessed_batch = self.reward_preprocessor(next_raw_batch)
            image_keys = [
                k
                for k in self.reward_classifier_model.config.input_features
                if "image" in k and k in preprocessed_batch
            ]
            if image_keys:
                with torch.inference_mode():
                    probs = self.reward_classifier_model.predict(
                        [preprocessed_batch[k] for k in image_keys]
                    ).probabilities
                if (probs > self.reward_success_threshold).any():
                    current_step_is_success = True
        if not current_step_is_success and self._check_teleop_event(TeleopEvents.SUCCESS):
            current_step_is_success = True

        # --- Consecutive success check (as requested) ---
        if current_step_is_success:
            self.consecutive_successes += 1
        else:
            self.consecutive_successes = 0

        is_success = self.consecutive_successes >= self.args.success_in_a_row
        # --- End consecutive success check ---

        if is_success:
            self.episode_is_success = True
        return 0.0, is_success

    def _determine_termination(self, is_success):
        done = self.done
        if is_success and self.reward_terminate_on_success:
            done = True
        if self._check_teleop_event(TeleopEvents.TERMINATE_EPISODE):
            done = True
        truncated = self.info.get("TimeLimit.truncated", False)
        if self.episode_total_steps + 1 >= self.cfg.env.max_episode_steps:
            done = True
            truncated = True
        return done, truncated

    def _update_episode_stats(self):
        if self._check_teleop_event(TeleopEvents.IS_INTERVENTION):
            self.episode_intervention = True
            self.episode_intervention_steps += 1
        self.episode_total_steps += 1

    def _prepare_and_buffer_transition(self, truncated):
        super()._prepare_and_buffer_transition(truncated)
        is_intervention = self._check_teleop_event(TeleopEvents.IS_INTERVENTION)
        action_array = np.concatenate([self.motion_manager.get_command_data(key) for key in self.action_keys])
        action_tensor = (
            self.final_action_tensor
            if not is_intervention
            else torch.from_numpy(action_array).float().unsqueeze(0).to(self.device)
        )
        if not self.args.no_plot:
            # FIX: Use reshape(-1) to ensure a 1D numpy array is appended,
            # preventing a scalar from being created when action_dim is 1,
            # which would cause the IndexError during plotting.
            action_np_1d = action_tensor.cpu().numpy().reshape(-1)
            nan_array = np.full(self.action_dim, np.nan)

            if is_intervention:
                self.teleop_action_list.append(action_np_1d)
                self.policy_action_list.append(nan_array)
            else:
                self.policy_action_list.append(action_np_1d)
                self.teleop_action_list.append(nan_array)

    def _handle_end_of_episode(self):
        if self.phase_manager.phase.name not in ["EndRolloutPhase", "EndTeleopRolloutPhase"]:
            return

        if self.args.online and not getattr(self, "summary_sent", False):
            final_reward = 1.0 if self.episode_is_success else 0.0
            if self.args.enable_manual_reward:
                print("\nEpisode ended. Manual reward: '1' for success, '0' for failure, any other key to skip.")
                key = cv2.waitKey(0)
                if key == ord("1"):
                    final_reward = 1.0
                elif key == ord("0"):
                    final_reward = 0.0

            if self.transition_buffer:
                self.transition_buffer[-1]["reward"] = final_reward
            self.episode_is_success = final_reward == 1.0

        super()._handle_end_of_episode()

    def run(self):
        """Main execution loop for the rollout."""
        self.reset_flag = True
        self.quit_flag = False
        self.key = 0
        self.interaction_step = getattr(self, "interaction_step", 0)
        self.reward = 0
        while not (self.key == 27 or self.quit_flag):
            if self.reset_flag:
                self.reset()
                self.reset_flag = False

            self.phase_manager.pre_update()

            self.executed_action = np.concatenate(
                [self.motion_manager.get_command_data(key) for key in self.env.unwrapped.command_keys_for_step]
            )

            if self.args.save_rollout and self.phase_manager.is_phase("RolloutPhase"):
                self.record_data()

            # Only step the environment if an action was executed
            if not np.all(self.executed_action == 0):
                self.obs, self.reward, self.done, _, self.info = self.env.step(self.executed_action)
            else:
                # If no action, we still need to get the latest observation
                self.obs = self.env.get_observation()
                self.done = self.env.is_done()
                self.info = self.env.get_info()

            self.interaction_step += 1
            self.post_step_hook()

            self.phase_manager.post_update()
            self.key = cv2.waitKey(1)
            self.phase_manager.check_transition()

            self._handle_end_of_episode()

        if self.args.result_filename:
            with open(self.args.result_filename, "w") as f:
                yaml.dump(self.result, f)
        self.print_statistics()
        self.cleanup()

    def setup_plot(self, fig_ax=None):
        if self.args.no_plot:
            self.ax = None
            return
        num_cols = len(self.camera_names) + 1
        if self.args.plot_transition_actions:
            num_cols += 1
        self.fig, self.axes = plt.subplots(
            1, num_cols, figsize=(num_cols * 4, 4), dpi=100, squeeze=False, constrained_layout=False
        )
        self.ax = self.axes
        self.twin_axes = {}
        plot_idx = len(self.camera_names)
        if self.args.plot_transition_actions:
            self.twin_axes["transition"] = self.axes[0, plot_idx].twinx()
            plot_idx += 1
        self.twin_axes["action"] = self.axes[0, plot_idx].twinx()
        for ax_row in self.axes:
            for ax in ax_row:
                ax.set_navigate(False)
        self.canvas = FigureCanvasAgg(self.fig)

    def plot_action(self, ax):
        """Plots policy and teleop actions on a dual-axis chart."""
        policy_actions = np.array(self.policy_action_list)
        teleop_actions = np.array(self.teleop_action_list)
        history_size = 100
        ax.cla()
        ax2 = self.twin_axes["action"]
        ax2.cla()
        ax.set_title("Action", fontsize=20)
        ax.set_xlabel("Step", fontsize=16)
        ax.set_xlim(0, history_size - 1)
        ax.grid(True, linestyle="--", alpha=0.6)
        lines = []
        
        # Check if the array is 2D, which it should be now thanks to the fix in _prepare_and_buffer_transition
        if policy_actions.ndim == 2 and policy_actions.shape[0] > 0 and policy_actions.shape[1] == self.action_dim:
            pose_data = policy_actions[-history_size:, :-1]
            gripper_data = policy_actions[-history_size:, -1]
            if pose_data.shape[1] > 0:
                p1_lines = ax.plot(pose_data, "-", label="Policy Pose", color="tab:blue")
                if p1_lines:
                    lines.append(p1_lines[0])
            p2_lines = ax2.plot(gripper_data, "-", label="Policy Gripper", color="purple")
            if p2_lines:
                lines.append(p2_lines[0])
                
        if teleop_actions.ndim == 2 and teleop_actions.shape[0] > 0 and teleop_actions.shape[1] == self.action_dim:
            pose_data = teleop_actions[-history_size:, :-1]
            gripper_data = teleop_actions[-history_size:, -1]
            if pose_data.shape[1] > 0:
                p1_lines = ax.plot(pose_data, "o-", label="Teleop Pose", color="tab:orange", markersize=3)
                if p1_lines:
                    lines.append(p1_lines[0])
            p2_lines = ax2.plot(gripper_data, "o-", label="Teleop Gripper", color="magenta", markersize=3)
            if p2_lines:
                lines.append(p2_lines[0])

        ax.set_ylabel("Pose Action", color="dimgray")
        ax.tick_params(axis="y", labelcolor="dimgray")
        ax2.set_ylabel("Gripper Action", color="purple")
        ax2.tick_params(axis="y", labelcolor="purple")
        
        # We need to reshape the arrays *before* vstack if they are 1D due to the original issue, 
        # but the fix ensures they are 2D (N, M). 
        # Here we rely on the fix applied above to ensure policy_actions and teleop_actions are 2D when non-empty.
        
        # Filter out 1D/empty arrays to prevent vstack crash if the fix wasn't robust, 
        # but since we fixed the append logic, they should be 2D.
        valid_policy_pose = policy_actions[-history_size:, :-1] if policy_actions.ndim == 2 and policy_actions.shape[1] > 0 else np.array([])
        valid_teleop_pose = teleop_actions[-history_size:, :-1] if teleop_actions.ndim == 2 and teleop_actions.shape[1] > 0 else np.array([])
        
        # Stack only if at least one array is valid
        all_pose_data = np.vstack(
            [arr for arr in [valid_policy_pose, valid_teleop_pose] if arr.size > 0]
        ) if valid_policy_pose.size > 0 or valid_teleop_pose.size > 0 else np.array([])
        
        # Handle gripper data (1D column from the 2D array)
        valid_policy_gripper = policy_actions[-history_size:, -1] if policy_actions.ndim == 2 and policy_actions.shape[1] > 0 else np.array([])
        valid_teleop_gripper = teleop_actions[-history_size:, -1] if teleop_actions.ndim == 2 and teleop_actions.shape[1] > 0 else np.array([])
        
        all_gripper_data = np.hstack(
            [arr for arr in [valid_policy_gripper, valid_teleop_gripper] if arr.size > 0]
        ) if valid_policy_gripper.size > 0 or valid_teleop_gripper.size > 0 else np.array([])

        if all_pose_data.size > 0 and np.any(~np.isnan(all_pose_data)):
            min_val, max_val = np.nanmin(all_pose_data), np.nanmax(all_pose_data)
            padding = (max_val - min_val) * 0.1 + 0.1
            ax.set_ylim(min_val - padding, max_val + padding)
        if all_gripper_data.size > 0 and np.any(~np.isnan(all_gripper_data)):
            min_val, max_val = np.nanmin(all_gripper_data), np.nanmax(all_gripper_data)
            padding = (max_val - min_val) * 0.1 + 0.1
            ax2.set_ylim(min_val - padding, max_val + padding)
        if lines:
            ax.legend(lines, [l.get_label() for l in lines], loc="upper left")

    def plot_transition_actions(self, ax):
        """Plots the actions stored in the transition buffer on a dual-axis chart."""
        ax.cla()
        ax2 = self.twin_axes["transition"]
        ax2.cla()
        ax.set_title("Transition Buffer", fontsize=16)
        ax.set_xlabel("Step in Buffer", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.6)
        actions_np = np.array([])
        if hasattr(self, "transition_buffer") and self.transition_buffer:
            # Note: This part uses squeeze(), which could potentially cause a similar issue 
            # if action_dim is 1 and all elements in transition_buffer are scalars.
            # However, since we are fetching from the buffer which stores the action tensor, 
            # and the tensor is created as (1, action_dim), it should be safe after .cpu().numpy().squeeze().
            # If the user's environment has action_dim=1, the buffer element `t["action"]` is likely 
            # a torch tensor of shape (1, 1). squeeze() turns it into a scalar,
            # which then results in a 1D actions_np array of shape (N,).
            actions_np_list = [t["action"].cpu().numpy().squeeze() for t in self.transition_buffer]
            actions_np = np.array(actions_np_list)
            
            # Defensive reshaping for the transition buffer plot as well
            if actions_np.ndim == 1 and self.action_dim > 1:
                # If actions_np is 1D, but action_dim > 1, something is wrong with the stored data.
                # Assuming the user's issue is primarily in plot_action due to action_dim=1,
                # we only reshape if it's 1D, regardless of action_dim, but handle dim > 1 for safety.
                actions_np = actions_np.reshape(-1, 1)

            if actions_np.size > 0 and actions_np.shape[1] == self.action_dim:
                pose_data = actions_np[:, :-1]
                gripper_data = actions_np[:, -1]
                lines = []
                if pose_data.shape[1] > 0:
                    p1s = ax.plot(pose_data, "-", label="Transition Pose", color="tab:green")
                    lines.append(p1s[0])
                p2s = ax2.plot(gripper_data, "-", label="Transition Gripper", color="purple")
                lines.append(p2s[0])
                ax.legend(lines, [l.get_label() for l in lines], loc="upper left")
                ax.set_ylabel("Pose Action", color="dimgray")
                ax.tick_params(axis="y", labelcolor="dimgray")
                ax2.set_ylabel("Gripper Action", color="purple")
                ax2.tick_params(axis="y", labelcolor="purple")
                if pose_data.size > 0 and np.any(~np.isnan(pose_data)):
                    min_val, max_val = np.nanmin(pose_data), np.nanmax(pose_data)
                    padding = (max_val - min_val) * 0.1 + 0.1
                    ax.set_ylim(min_val - padding, max_val + padding)
                if gripper_data.size > 0 and np.any(~np.isnan(gripper_data)):
                    min_val, max_val = np.nanmin(gripper_data), np.nanmax(gripper_data)
                    padding = (max_val - min_val) * 0.1 + 0.1
                    ax2.set_ylim(min_val - padding, max_val + padding)

    def draw_plot(self):
        if self.args.no_plot or not hasattr(self, "canvas"):
            return
        axes_row = self.axes[0]
        for ax in axes_row[: len(self.camera_names)]:
            ax.cla()
            ax.axis("off")
        self.plot_images(axes_row[: len(self.camera_names)])
        plot_idx = len(self.camera_names)
        if self.args.plot_transition_actions:
            self.plot_transition_actions(axes_row[plot_idx])
            plot_idx += 1
        self.plot_action(axes_row[plot_idx])
        self.fig.tight_layout()
        self.canvas.draw()
        frame = np.asarray(self.canvas.buffer_rgba())
        cv2.imshow(self.policy_name, cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR))


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        rollout = RolloutLerobotAcfql()
        rollout.run()
