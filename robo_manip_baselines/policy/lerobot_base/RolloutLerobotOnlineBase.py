# ruff: noqa: E501
import logging
import time
from threading import Thread
import numpy as np
    
import cv2
import torch
from termcolor import colored
import contextlib

# --- LeRobot Imports ---
from lerobot.processor import TransitionKey
from lerobot.rl.actor import (
    establish_learner_connection,
    learner_service_client,
    push_transitions_to_transport_queue,
    receive_policy,
    send_interactions,
    send_transitions,
    use_threads,
)
from lerobot.rl.process import ProcessSignalHandler
from lerobot.rl.queue import get_last_item_from_queue
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.transport.utils import bytes_to_state_dict, python_object_to_bytes
from lerobot.utils.transition import Transition, move_state_dict_to_device

logger = logging.getLogger(__name__)


class RolloutLerobotOnlineBase:
    """A base class for policies that support online training with a remote learner."""

    def __init__(self):
        """Initializes common attributes for online rollouts."""
        super().__init__()
        # This list is used by TeleopRolloutBase to track performance.
        self.inference_duration_list = []

    def set_additional_args(self, parser):
        super().set_additional_args(parser)
        parser.add_argument("--online", action="store_true", help="Enable online learning mode.")
        parser.add_argument(
            "--stale_policy_warning_s",
            type=int,
            default=60,
            help="Warn if policy parameters haven't been updated in this many seconds.",
        )
        parser.add_argument(
            "--time_step_delay", type=float, default=0.0, help="Artificial delay between steps (in seconds)."
        )
        parser.add_argument(
            "--ask_before_sending",
            action="store_true",
            help="Prompt user before sending an episode's data to the learner.",
        )
        parser.add_argument(
            "--debug_inference", action="store_true", help="Enable detailed logging for policy inference."
        )
        parser.add_argument("--skip_inference", type=int, default=1, help="Run policy inference every N steps.")

    def infer_policy(self, use_processors=True):
        """Performs a single step of policy inference with online updates."""
        if self.args.online:
            self.update_policy_parameters()

        if self.args.time_step_delay > 0.0:
            time.sleep(self.args.time_step_delay)
        if self.prev_raw_batch is None:
            self.prev_raw_batch = self._get_raw_observation_batch(debug=self.args.debug_inference)

        if use_processors:
            # --- START WORKAROUND: Use TransitionKey enum for processor input ---
            # The `batch_processor` step expects specific enum keys, even if the values are None.
            transition_in = {
                TransitionKey.OBSERVATION: self.prev_raw_batch,
                TransitionKey.ACTION: None,
                TransitionKey.COMPLEMENTARY_DATA: None,
            }
            transition_out = self.preprocessor._forward(transition_in)
            
            # The output from `_forward` uses string keys, so we can extract "observation" directly.
            self.processed_batch = transition_out["observation"]
            # --- END WORKAROUND ---
        else:
            self.processed_batch = self.prev_raw_batch
        self.processed_batch = {
            k: v for k, v in self.processed_batch.items() if k in self.policy.config.input_features
        }
        is_skip = self.rollout_time_idx % self.args.skip_inference != 0
        if not is_skip:
            timer = self.policy_timer if self.args.online else contextlib.nullcontext()
            with timer, torch.inference_mode():
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


    def setup_online_components(self):
        """Sets up queues, events, and communication processes for online learning."""
        if not self.args.online:
            return

        logger.info("Setting up components for online communication.")
        self.use_threads = use_threads(self.cfg)
        self.parameters_queue = torch.multiprocessing.Queue()
        self.transitions_queue = torch.multiprocessing.Queue()
        self.interactions_queue = torch.multiprocessing.Queue()
        self.shutdown_event = ProcessSignalHandler(use_threads=self.use_threads).shutdown_event
        self.processes = []
        self.transition_buffer = []
        self.last_policy_update_time = time.time()

        self.start_communication_workers()

    def start_communication_workers(self):
        """Establishes connection with the learner and starts communication threads/processes."""
        learner_client, grpc_channel = learner_service_client(
            host=self.cfg.policy.actor_learner_config.learner_host,
            port=self.cfg.policy.actor_learner_config.learner_port,
        )
        if not establish_learner_connection(learner_client, self.shutdown_event):
            raise ConnectionError("[ACTOR] Failed to establish connection with Learner")
        if not self.use_threads:
            grpc_channel.close()

        concurrency_entity = Thread if self.use_threads else torch.multiprocessing.Process
        self.processes = [
            concurrency_entity(
                target=receive_policy,
                args=(self.cfg, self.parameters_queue, self.shutdown_event),
                daemon=True,
            ),
            concurrency_entity(
                target=send_transitions,
                args=(self.cfg, self.transitions_queue, self.shutdown_event),
                daemon=True,
            ),
            concurrency_entity(
                target=send_interactions,
                args=(self.cfg, self.interactions_queue, self.shutdown_event),
                daemon=True,
            ),
        ]
        for p in self.processes:
            p.start()
        logging.info(colored("Communicator started.", "green"))

    def stop_communication_workers(self):
        """Signals all communication workers to shut down."""
        if not hasattr(self, "shutdown_event") or not self.args.online:
            return

        if not self.shutdown_event.is_set():
            self.shutdown_event.set()
            for process in self.processes:
                if process and hasattr(process, "is_alive") and process.is_alive():
                    process.join(timeout=2)
            logging.info(colored("[ACTOR] Communication processes joined.", "yellow"))

    def update_policy_parameters(self):
        """
        Pulls the latest policy parameters from the queue and updates the local policy.
        Returns True if an update occurred, False otherwise.
        """
        if time.time() - self.last_policy_update_time > self.args.stale_policy_warning_s:
            logger.warning(
                f"Policy parameters have not been updated in over {self.args.stale_policy_warning_s} seconds. "
                "Check learner status and network connection."
            )
            # Reset timer to avoid spamming warnings
            self.last_policy_update_time = time.time()

        bytes_state_dict = get_last_item_from_queue(self.parameters_queue, block=False)
        if bytes_state_dict is not None:
            state_dicts = bytes_to_state_dict(bytes_state_dict)

            if "policy" in state_dicts:
                policy_state_dict = move_state_dict_to_device(state_dicts["policy"], device=self.device)

                # Determine the correct target model to load into
                target_model = None
                if hasattr(self.policy, "actor"):  # For HIL-SERL/SAC
                    target_model = self.policy.actor
                    logger.debug("Target model for state_dict loading is `policy.actor` (HIL-SERL/SAC).")
                elif hasattr(self.policy, "actor_onestep_flow"):  # For ACFQL
                    target_model = self.policy.actor_onestep_flow
                    logger.debug(
                        "Target model for state_dict loading is `policy.actor_onestep_flow` (ACFQL)."
                    )
                else:
                    logger.error("Could not determine the target model to load parameters into.")
                    return False

                # --- START FIX: Handle torch.compile prefix mismatch ---
                local_keys = set(target_model.state_dict().keys())

                has_prefix_in_local = any(k.startswith("_orig_mod.") for k in local_keys)
                has_prefix_in_incoming = any(k.startswith("_orig_mod.") for k in policy_state_dict.keys())

                if has_prefix_in_local and not has_prefix_in_incoming:
                    logger.debug(
                        "Local model is compiled but incoming state_dict is not. Adding '_orig_mod.' prefix to keys."
                    )
                    policy_state_dict = {f"_orig_mod.{k}": v for k, v in policy_state_dict.items()}
                elif not has_prefix_in_local and has_prefix_in_incoming:
                    logger.debug(
                        "Incoming state_dict is compiled but local model is not. Stripping '_orig_mod.' prefix from keys."
                    )
                    policy_state_dict = {
                        k.replace("_orig_mod.", ""): v for k, v in policy_state_dict.items()
                    }
                # --- END FIX ---

                try:
                    target_model.load_state_dict(policy_state_dict)
                    log_time = time.strftime("%H:%M:%S", time.localtime())
                    logger.info(
                        f"[{log_time} | Step {getattr(self, 'episode_total_steps', 0)}] [ACTOR] Loaded new parameters from Learner."
                    )
                    self.last_policy_update_time = time.time()
                except RuntimeError as e:
                    logger.error(f"Failed to load state_dict into {type(target_model).__name__}: {e}")
                    logger.error(
                        "This often means there is an architecture mismatch between the trained model "
                        "(from the learner) and the policy in this rollout script."
                    )
                    logger.error(
                        "CRITICAL: Ensure you are using the correct Rollout script for your policy type "
                        "(e.g., 'RolloutLerobotAcfql' for an ACFQL policy, 'RolloutLerobotHilSerl' for HIL-SERL)."
                    )
                    return False  # Stop further processing on error

            # Handle discrete critic separately if it exists
            if hasattr(self.policy, "discrete_critic") and "discrete_critic" in state_dicts:
                if self.policy.discrete_critic is not None:
                    discrete_critic_state_dict = move_state_dict_to_device(
                        state_dicts["discrete_critic"], device=self.device
                    )
                    # Also handle torch.compile for the discrete critic
                    if any(key.startswith("_orig_mod.") for key in discrete_critic_state_dict.keys()):
                        discrete_critic_state_dict = {
                            k.replace("_orig_mod.", ""): v for k, v in discrete_critic_state_dict.items()
                        }
                    self.policy.discrete_critic.load_state_dict(discrete_critic_state_dict)
                    logger.info("Loaded new discrete_critic parameters from Learner.")

            return True
        return False

    def post_step_hook(self):
        """Generic post-step hook for online learning. Can be overridden by subclasses for more complex logic."""
        if not self.args.online or self.prev_raw_batch is None:
            return

        if not (
            self.phase_manager.is_phase("TeleopOverridePhase") or self.phase_manager.is_phase("RolloutPhase")
        ):
            return

        is_success = self.reward >= 1.0
        if not is_success and self._check_teleop_event(TeleopEvents.SUCCESS):
            is_success = True

        if is_success:
            self.episode_is_success = True

        self.reward = 0.0  # Sparse reward for all steps

        done = self.done
        if is_success and getattr(self, "reward_terminate_on_success", False):
            done = True
        if self._check_teleop_event(TeleopEvents.TERMINATE_EPISODE):
            done = True

        self.done = done
        truncated = self.info.get("TimeLimit.truncated", False)
        if (
            hasattr(self, "episode_total_steps")
            and hasattr(self.cfg, "env")
            and self.episode_total_steps + 1 >= self.cfg.env.max_episode_steps
        ):
            self.done = True
            truncated = True

        if hasattr(self, "episode_total_steps"):
            if self._check_teleop_event(TeleopEvents.IS_INTERVENTION):
                self.episode_intervention = True
                self.episode_intervention_steps += 1
            self.episode_total_steps += 1

        self._prepare_and_buffer_transition(truncated)

    def _prepare_and_buffer_transition(self, truncated):
        """Prepares and buffers the transition object using raw state and action data."""
        is_intervention = self._check_teleop_event(TeleopEvents.IS_INTERVENTION)
        if is_intervention:
            action_parts = [self.motion_manager.get_command_data(key) for key in self.action_keys]
            action_np = np.concatenate(action_parts)
            action_tensor = torch.from_numpy(action_np).float().unsqueeze(0).to(self.device)
        else:
            action_tensor = self.final_action_tensor

        # --- START FIX: Ensure state and next_state only contain tensors ---
        # The `move_transition_to_device` utility expects all values in state dictionaries to be tensors.
        # We filter the raw observation batches to remove any non-tensor data (like lists).
        state = {k: v for k, v in self.prev_raw_batch.items() if isinstance(v, torch.Tensor)}

        next_raw_batch = self._get_raw_observation_batch(debug=self.args.debug_inference) if not self.done else None
        if next_raw_batch:
            next_state = {k: v for k, v in next_raw_batch.items() if isinstance(v, torch.Tensor)}
        else:
            # If done, the next state is a clone of the current state's tensors.
            next_state = {k: v.clone() for k, v in state.items()}
        # --- END FIX ---

        transition = Transition(
            state=state,
            action=action_tensor,
            reward=float(self.reward),
            next_state=next_state,
            done=bool(self.done),
            truncated=bool(truncated),
            complementary_info={
                k.value: v
                for k, v in self.teleop_events.items()
                if isinstance(v, (int, float, bool)) and isinstance(k, TeleopEvents)
            },
        )
        self.transition_buffer.append(transition)
        self.prev_raw_batch = next_raw_batch

    def _handle_end_of_episode(self):
        """Manages sending buffered data to the learner at the end of an episode."""
        if not self.args.online or getattr(self, "summary_sent", False):
            return
        self.summary_sent = True

        final_reward = 1.0 if getattr(self, "episode_is_success", False) else 0.0
        if self.transition_buffer:
            self.transition_buffer[-1]["reward"] = final_reward

            send_episode = True
            if self.args.ask_before_sending:
                print("Send this episode to the learner? (y/n)")
                send_episode = (cv2.waitKey(0) & 0xFF) == ord("y")

            if send_episode:
                push_transitions_to_transport_queue(self.transition_buffer, self.transitions_queue)
            self.transition_buffer.clear()

        if hasattr(self, "episode_total_steps"):
            intervention_rate = (
                self.episode_intervention_steps / self.episode_total_steps
                if self.episode_total_steps > 0
                else 0
            )
            stats = (
                {"Policy frequency [Hz]": self.policy_timer.fps_avg}
                if hasattr(self, "policy_timer") and self.policy_timer.count > 1
                else {}
            )

            summary = {
                "Episodic reward": float(final_reward),
                "Episode length": self.episode_total_steps,
                "Episode intervention": int(getattr(self, "episode_intervention", False)),
                "Intervention rate": intervention_rate,
                "Interaction step": self.interaction_step,
                **stats,
            }
            self.interactions_queue.put(python_object_to_bytes(summary))
            logger.info(f"Sent episode summary: {summary}")

    def _check_teleop_event(self, event):
        """Safely checks for a teleoperation event."""
        return self.teleop_events.get(event, False) or self.teleop_events.get(event.value, False)

    def cleanup(self):
        super().cleanup()
        if self.args.online:
            self.stop_communication_workers()

