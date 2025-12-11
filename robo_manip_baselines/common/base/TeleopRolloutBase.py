import time
import os
import cv2
import numpy as np
import torch
import yaml
import sys
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pinocchio as pin
import re
import pickle

# Import base classes from the project structure
from ..manager.PhaseManager import PhaseManager
from .PhaseBase import PhaseBase
from .RolloutBase import EndRolloutPhase, InitialRolloutPhase, RolloutBase, RolloutPhase
from ..data.DataKey import DataKey
from ..data.TeleopOperationDataMixin import TeleopOperationDataMixin
from enum import Enum
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

# --- Path Setup ---
# Ensures the script can find modules from RoboManipBaselines and vendored LeRobot.
try:
    ROBOMANIP_PATH = Path(__file__).resolve().parents[3]
    if str(ROBOMANIP_PATH) not in sys.path:
        sys.path.append(str(ROBOMANIP_PATH))
    LEROBOT_PATH = ROBOMANIP_PATH / "third_party" / "lerobot" / "src"
    if str(LEROBOT_PATH) not in sys.path:
        sys.path.append(str(LEROBOT_PATH))
except NameError:
    # Handles cases where __file__ is not defined (e.g., in interactive notebooks).
    print("Warning: Could not determine ROBOMANIP_PATH. Assuming paths are correctly set.")


try:
    # Attempt to import TeleopEvents directly from lerobot
    from lerobot.teleoperators.utils import TeleopEvents
except ImportError:
    print("Warning: Using local TeleopEvents.")
    # If the import fails (e.g., lerobot is not installed),
    # define the class locally as a fallback.
    class TeleopEvents(str, Enum):
        """
        Shared constants for teleoperator events, inspired by lerobot.
        These events are used to signal human interventions and episode status.
        """

        SUCCESS = "success"
        FAILURE = "failure"
        RERECORD_EPISODE = "rerecord_episode"
        IS_INTERVENTION = "is_intervention"
        TERMINATE_EPISODE = "terminate_episode"

class TeleopOverridePhase(RolloutPhase):
    """
    Overrides the standard RolloutPhase to handle generic teleoperation input,
    allowing a human to override the policy's actions in real-time.
    """

    def start(self):
        super().start()
        self.was_overriding = False #for managing the policy's state across the human-to-agent transition
        

    def pre_update(self):
        # HIL-SERL CHANGE: Initialize HIL-related flags for the current step.
        self.op.is_intervention = False
        # Get events from keyboard first
        keyboard_events = {}

        # New Space Key Logic
        if self.op.key == ord(" "):
            self.op.force_intervention = not self.op.force_intervention
            if self.op.force_intervention:
                print("INFO: Space key pressed - Forcing intervention.")
            else:
                print("INFO: Space key pressed - Releasing intervention.")

        if self.op.key == ord("g"):
            keyboard_events[TeleopEvents.SUCCESS] = True
            self.op.terminated_by_teleop = True
            print("INFO: 'g' key pressed - Firing SUCCESS event.")
        elif self.op.key == ord("f"):
            keyboard_events[TeleopEvents.TERMINATE_EPISODE] = True
            print("INFO: 'f' key pressed - Firing TERMINATE_EPISODE event.")
            self.op.terminated_by_teleop = True
        else:
            if self.op.terminated_by_teleop:
                print('reset self.terminated_by_teleop')
                self.op.terminated_by_teleop = False
                  
        self.op.teleop_events = keyboard_events

        is_overriding_now = False
        if self.op.args.teleop_device:
            for device in self.op.input_device_list:
                device.read()  # Read latest state from the device
                if device.is_active():
                    device.set_command_data()
                    is_overriding_now = True
                # HIL-SERL CHANGE: Get intervention events from the teleop device.
                if hasattr(device, "get_teleop_events"):
                    device_events = device.get_teleop_events()
                    # Careful merge: give precedence to True values to avoid overwriting keyboard events
                    for key, value in device_events.items():
                        # If key exists, OR the values. Otherwise, just set it.
                        self.op.teleop_events[key] = self.op.teleop_events.get(key, False) or value

                    self.op.is_intervention = self.op.teleop_events.get(
                        TeleopEvents.IS_INTERVENTION, False
                    )
        
        # HIL-SERL CHANGE: If there's an intervention, use the teleop action.
        # The device.set_command_data() already sets the motion manager's command.
        if is_overriding_now or self.op.is_intervention or self.op.force_intervention:
             self.op.teleop_events[TeleopEvents.IS_INTERVENTION] = True
        else:
            # The policy is in control.
            if self.was_overriding:
                if (
                    hasattr(self.op, "policy_action_buf")
                    and self.op.policy_action_buf is not None
                ):
                    self.op.policy_action_buf = None
                # MODIFICATION: This line was clearing the action history. It has been removed.
                # self.op.reset_variables()

            # Standard policy inference logic.
            if self.op.rollout_time_idx % self.op.args.skip == 0:
                inference_start_time = time.time()
                self.op.infer_policy()
                self.op.inference_duration_list.append(
                    time.time() - inference_start_time
                )
            self.op.set_command_data()

        self.was_overriding = is_overriding_now or self.op.is_intervention or self.op.force_intervention

    def check_transition(self):
        elapsed_duration = self.get_elapsed_duration()
        terminate = self.op.teleop_events.get(TeleopEvents.TERMINATE_EPISODE, False)
        success_from_teleop = self.op.teleop_events.get(TeleopEvents.SUCCESS, False)
        is_success = success_from_teleop or (self.op.reward >= 1.0)

        # Check for teleop or environment-based termination
        if terminate or success_from_teleop or self.op.done:
            self.op.done = True
            # Command the robot to hold its current position to stop gracefully before reset.
            current_joint_pos = self.op.motion_manager.get_data("measured_joint_pos", self.op.obs)
            self.op.motion_manager.set_command_data("command_joint_pos", current_joint_pos)
            print(f"INFO: Episode ended via teleop/env. Done: {self.op.done}, Success: {is_success}, Terminate: {terminate}, 'op.reward: {self.op.reward}")

            # Correctly populate results before transitioning
            self.op.result["success"].append(is_success)
            self.op.result["reward"].append(float(self.op.reward))
            self.op.result["duration"].append(elapsed_duration)
            if self.op.args.save_last_image:
                self.op.save_rgb_image()
            return True

        # Fall back to the original RolloutPhase logic for manual termination ('n' key) or timeout.
        # This will also populate the results dictionary if it transitions.
        return super().check_transition()


class EndTeleopRolloutPhase(EndRolloutPhase):
    """
    Overrides the standard EndRolloutPhase to allow continuous rollouts
    without automatically quitting and to display the final status on the plot.
    """

    def start(self):
        PhaseBase.start(self)
        if not self.op.args.auto_reset:
            if self.op.args.save_rollout:
                msg = f"[{self.op.__class__.__name__}] Policy rollout is finished. Press 's' to save, 'f' to discard. Then press 'n' to start the next trial."
            else:
                msg = f"[{self.op.__class__.__name__}] Policy rollout is finished."
            if not self.op.args.auto_exit:
                    msg += " Press the 'n' key to reset."
            print(msg)
        # Draw the plot once when entering the phase
        self.op.draw_plot()

    def post_update(self):
        # Continuously redraw the plot to keep the window responsive and the status visible
        self.op.draw_plot()

        if not self.op.args.save_rollout:
            return

        if self.op.key == ord("s"):
            self.op.save_data()
            print(f"[{self.op.__class__.__name__}] Data saved for episode. Press 'n' to continue.")
        elif self.op.key == ord("f"):
            print(f"[{self.op.__class__.__name__}] Data for episode discarded. Press 'n' to continue.")
            self.op.data_manager.episode_idx += 1


    def check_transition(self):
        # HIL-SERL CHANGE: This logic now ensures the rollout loops continuously instead of exiting.
        if self.op.key == ord("n") or self.op.args.auto_exit or self.op.args.auto_reset:
            if not self.op.args.save_rollout:
                # If not saving, 'n' is the only action, so we increment the episode index here.
                self.op.data_manager.episode_idx += 1
            self.op.reset_flag = True

        # Never automatically transition out of this phase.
        # Reset is handled by setting the reset_flag. Quit is handled by ESC key in the main loop.
        return False


# --- Main Teleop Rollout Base Class ---
class TeleopRolloutBase(TeleopOperationDataMixin, RolloutBase):
    """
    An extension of RolloutBase that adds Human-in-the-Loop (HiL) capabilities.
    """

    def __init__(self):
        super().__init__()

        if self.args.eef_bound_min and self.args.eef_bound_max:
            print(f"INFO: Applying safety bounds to ArmManagers.")
            for manager in self.motion_manager.body_manager_list:
                if hasattr(manager, 'pin_model'):  # A simple check for ArmManager
                    manager.eef_bound_min = np.array(self.args.eef_bound_min)
                    manager.eef_bound_max = np.array(self.args.eef_bound_max)
                    print(f"  - Positional Bounds: min={manager.eef_bound_min}, max={manager.eef_bound_max}")

        if self.args.eef_orientation_target and self.args.eef_orientation_max_dev is not None:
            print(f"INFO: Applying orientation constraints to ArmManagers.")
            for manager in self.motion_manager.body_manager_list:
                if hasattr(manager, 'pin_model'):
                    manager.eef_orientation_target = np.array(self.args.eef_orientation_target)
                    manager.eef_orientation_max_dev_rad = np.deg2rad(self.args.eef_orientation_max_dev)
                    # Convert target quaternion to rotation matrix for easier calculations
                    manager.target_orientation_rot = pin.Quaternion(*manager.eef_orientation_target).toRotationMatrix()
                    print(f"  - Orientation Target (quat): {manager.eef_orientation_target}")
                    print(f"  - Max Deviation (deg): {self.args.eef_orientation_max_dev}")

        # MODIFICATION: Find the next available episode index to avoid overwriting files.
        output_dir = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "dataset",
                self.demo_name,
            )
        )
        
        start_episode_idx = 0
        if os.path.exists(output_dir):
            max_episode_idx = -1
            # Regex to find the episode number from filenames like 'MujocoUR5eSimplePick_world0_000.rmb'
            pattern = re.compile(rf"^{self.demo_name}_world\d+_(\d+)\.rmb$")
            for item in os.listdir(output_dir):
                # We only check for directories, since .rmb files are now directories
                if os.path.isdir(os.path.join(output_dir, item)):
                    match = pattern.match(item)
                    if match:
                        episode_num = int(match.group(1))
                        if episode_num > max_episode_idx:
                            max_episode_idx = episode_num
            
            if max_episode_idx > -1:
                start_episode_idx = max_episode_idx + 1
        
        self.data_manager.episode_idx = start_episode_idx
        if start_episode_idx > 0:
            print(f"INFO: Found existing rollouts in {output_dir}. Starting new rollouts from episode index {start_episode_idx}.")

        self.input_device_list = []
        if self.args.teleop_device:
            if self.args.input_device_config is None:
                input_device_kwargs = {}
            else:
                with open(self.args.input_device_config, "r") as f:
                    input_device_kwargs = yaml.safe_load(f)
            
            if self.args.pos_scale is not None:
                input_device_kwargs['pos_scale'] = self.args.pos_scale
            if self.args.rpy_scale is not None:
                input_device_kwargs['rpy_scale'] = self.args.rpy_scale
            if self.args.gripper_scale is not None:
                input_device_kwargs['gripper_scale'] = self.args.gripper_scale

            self.input_device_list = self.env.unwrapped.setup_input_device(
                self.args.teleop_device, self.motion_manager, input_device_kwargs
            )
            for device in self.input_device_list:
                device.connect()

        phase_order = [
            InitialRolloutPhase(self),
            *self.get_pre_motion_phases(),
            TeleopOverridePhase(self),
            EndTeleopRolloutPhase(self),
        ]
        self.phase_manager = PhaseManager(phase_order)
        # HIL-SERL CHANGE: Initialize attributes to store intervention data.
        self.is_intervention = False
        self.teleop_events = {}
        self.terminated_by_teleop = False
        self.force_intervention = False

    def set_additional_args(self, parser):
        # HIL-SERL CHANGE: This method is restored to fix the argument parsing error.
        super().set_additional_args(parser)
        parser.add_argument(
            "--teleop_device",
            type=str,
            default=None,
            choices=["spacemouse", "keyboard", "gello", "keyboard_azerty"],
            help="Enable teleoperation override with the specified device.",
        )
        parser.add_argument(
            "--input_device_config",
            type=str,
            help=
            "Configuration file of the input device (e.g., for spacemouse path).",
        )
        parser.add_argument(
            "--pos_scale",
            type=float,
            default=None,
            help="Scaling factor for positional control (translation). Affects spacemouse and keyboard.",
        )
        parser.add_argument(
            "--rpy_scale",
            type=float,
            default=None,
            help="Scaling factor for rotational control (roll, pitch, yaw). Affects spacemouse and keyboard.",
        )
        parser.add_argument(
            "--gripper_scale",
            type=float,
            default=None,
            help="Scaling factor for gripper control. Affects spacemouse and keyboard.",
        )
        parser.add_argument(
            "--eef-bound-min",
            type=float,
            nargs=3,
            default=None,
            help="Minimum [X, Y, Z] boundary for the end-effector.",
        )
        parser.add_argument(
            "--eef-bound-max",
            type=float,
            nargs=3,
            default=None,
            help="Maximum [X, Y, Z] boundary for the end-effector.",
        )
        parser.add_argument(
            "--eef-orientation-target",
            type=float,
            nargs=4,
            default=None,
            help="Target orientation as a quaternion [qw, qx, qy, qz].",
        )
        parser.add_argument(
            "--eef-orientation-max-dev",
            type=float,
            default=None,
            help="Maximum allowed deviation in degrees from the target orientation.",
        )
        parser.add_argument(
            "--save_camera_feed",
            action="store_true",
            help="Save camera feeds (rgb and depth images) during rollout. Only active if --save_rollout is also enabled.",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose logging for debugging purposes.",
        )
        parser.add_argument(
            "--camera_crops",
            nargs="*",
            type=str,
            default=None,
            help="Define cropping regions for cameras. Format: camera_name1:x,y,w,h camera_name2:x,y,w,h",
        )
        parser.add_argument(
            "--target_camera_resolution",
            type=str,
            default=None,
            help=(
                "Define target camera resolution as height,width "
                "(applied to all cameras after cropping, e.g., 480,640)."
            ),
        )
        parser.add_argument(
            "--auto_reset",
            action="store_true",
            help="Automatically reset the environment after an episode finishes.",
        )
    
    def setup_model_meta_info(self):
        # Call the parent's implementation (which now safely loads the pickle)
        super().setup_model_meta_info()

        # The parent (RolloutBase) has already set these from the pickle file.
        # We re-set them here to ensure they are correct, using .get for safety.
        self.state_keys = self.model_meta_info["state"]["keys"]
        self.action_keys = self.model_meta_info["action"]["keys"]
        self.camera_names = self.model_meta_info["image"]["camera_names"]
        self.state_dim = len(self.model_meta_info["state"].get("example", self.state_keys))
        self.action_dim = len(self.model_meta_info["action"].get("example", self.action_keys))

        # Set skip if not specified (already in base, but safe to repeat)
        if self.args.skip is None:
            self.args.skip = self.model_meta_info["data"]["skip"]
        if self.args.skip_draw is None:
            self.args.skip_draw = self.args.skip

        # --- This is the key logic this class adds ---
        # Parse camera crops from command-line arguments, overriding pickle/defaults
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


        # Parse target camera resolution (height,width) from command-line arguments
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
        logger.info('model_meta_info: %s', self.model_meta_info)

    def reset(self):
        # This method is an override of RolloutBase.reset() to handle
        # world index looping correctly for continuous rollouts.
        # Reset plot
        if not self.args.no_plot:
            for _ax in np.ravel(self.ax):
                _ax.cla()
                _ax.axis("off")

            self.canvas = FigureCanvasAgg(self.fig)
            self.canvas.draw()
            cv2.imshow(
                self.policy_name,
                cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
            )

        # Reset motion manager
        self.motion_manager.reset()

        # Reset data manager (clears buffers for the new episode)
        self.data_manager.reset()

        # Reset environment
        self.env.unwrapped.world_random_scale = self.args.world_random_scale

        # MODIFICATION: Use modulo to loop through worlds. This allows `episode_idx`
        # to increment indefinitely, preventing filename collisions, while still
        # cycling through the specified `world_idx_list`.
        world_idx_list = self.args.world_idx_list
        if len(world_idx_list) > 0:
            world_idx = world_idx_list[self.data_manager.episode_idx % len(world_idx_list)]
        else:
            world_idx = 0  # Fallback if the list is empty

        self.data_manager.setup_env_world(world_idx)
        self.obs, self.info = self.env.reset(seed=self.args.seed)
        self.reward = 0
        msg = f"[{self.__class__.__name__}] Reset environment. demo_name: {self.demo_name}, world_idx: {self.data_manager.world_idx}, episode_idx: {self.data_manager.episode_idx}"
        if self.require_task_desc:
            msg += f", task desc: {self.args.task_desc}"
        print(msg)

        # Reset phase manager
        self.phase_manager.reset()

        # Reset variables
        self.reset_variables()


    def reset_variables(self):
        super().reset_variables()
        self.policy_action = np.zeros(self.action_dim)
        self.force_intervention = False

    def get_images(self, debug: bool = False):
        processed_images = []

        for name in self.camera_names:
            img_hwc = self.info["rgb_images"][name]

            # Apply crop if specified
            if self.parsed_camera_crops and name in self.parsed_camera_crops:
                x, y, w, h = self.parsed_camera_crops[name]
                img_hwc = img_hwc[y : y + h, x : x + w]
                if debug:
                    logger.info(
                        f"Step {self.rollout_time_idx}: Cropped image for camera {name} | "
                        f"Crop: (x={x}, y={y}, w={w}, h={h}) | New shape: {img_hwc.shape}"
                    )

            # Apply resize if specified
            if self.target_camera_resolution:
                target_h, target_w = self.target_camera_resolution
                img_hwc = cv2.resize(img_hwc, (target_w, target_h), interpolation=cv2.INTER_AREA)
                if debug:
                    logger.info(
                        f"Step {self.rollout_time_idx}: Resized image for camera {name} | "
                        f"Target resolution: ({target_w}, {target_h}) | New shape: {img_hwc.shape}"
                    )

            # Ensure uint8 type (no normalization)
            if img_hwc.dtype != np.uint8:
                img_hwc = img_hwc.astype(np.uint8)

            processed_images.append(img_hwc)

        # Stack images [Cameras, Height, Width, Channels]
        images = np.stack(processed_images, axis=0)

        # Move channel dimension to [Cameras, Channels, Height, Width]
        images = np.moveaxis(images, -1, -3)

        # Convert to torch tensor
        images = torch.tensor(images, dtype=torch.uint8)

        # Apply defined transforms (may include normalization later)
        images = self.image_transforms(images)[torch.newaxis].to(self.device)

        return images

    def plot_images(self, axes):
        for camera_idx, camera_name in enumerate(self.camera_names):
            image = self.info["rgb_images"][camera_name].copy()

            if camera_name in self.parsed_camera_crops:
                x, y, w, h = self.parsed_camera_crops[camera_name]
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            axes[camera_idx].imshow(image)
            axes[camera_idx].set_title(camera_name, fontsize=20)
            axes[camera_idx].axis("off")

    def draw_plot(self):
        # First, call the parent's draw_plot to render the base graphs (e.g., action plot)
        super().draw_plot()

        # Add a dedicated status box for real-time reward and done status.
        reward = getattr(self, "reward", 0.0)
        done = getattr(self, "done", False)
        intervention_status = "FORCED" if self.force_intervention else ("ACTIVE" if self.is_intervention else "INACTIVE")
        status_text = f"Reward: {reward:.2f}\nDone: {done}\nIntervention: {intervention_status}"


        # Display the status in the top-right corner with a styled box.
        self.fig.text(0.98, 0.98, status_text,
                      ha='right', va='top',
                      fontsize=12, color='black', weight='bold',
                      transform=self.fig.transFigure,
                      bbox=dict(boxstyle="round,pad=0.5", fc="aliceblue", ec="grey", lw=1))

        # Then, if we are in the end phase, overlay the final success/failure text.
        if self.phase_manager.is_phase("EndTeleopRolloutPhase"):
            if self.result["success"] and self.result["reward"]:
                is_success = self.result["success"][-1]
                final_reward = self.result["reward"][-1]
                end_status_text = f"{'SUCCESS' if is_success else 'FAILURE'}\nFinal Reward: {final_reward:.2f}"
                color = "green" if is_success else "red"

                # Add text to the figure's center.
                self.fig.text(0.5, 0.5, end_status_text,
                              ha='center', va='center',
                              fontsize=40, color=color, weight='bold',
                              transform=self.fig.transFigure,
                              bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))
        
        # Redraw the canvas after all text has been added.
        self.canvas.draw()
        plot_image_bgr = cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        cv2.imshow(self.policy_name, plot_image_bgr)
        cv2.waitKey(1)

    def run(self):
        self.reset_flag = True
        self.quit_flag = False
        self.inference_duration_list = []

        while True:
            if self.reset_flag:
                self.reset()
                self.reset_flag = False

            self.phase_manager.pre_update()

            env_action = np.concatenate(
                [
                    self.motion_manager.get_command_data(key)
                    for key in self.env.unwrapped.command_keys_for_step
                ]
            )
            
            # Step the environment
            if not np.all(env_action == 0):
                self.obs, self.reward, self.done, _, self.info = self.env.step(
                    env_action
                )

            # Record data after stepping the environment to get the latest info
            if self.args.save_rollout and isinstance(self.phase_manager.phase, RolloutPhase):
                self.record_data()

            self.phase_manager.post_update()

            self.key = cv2.waitKey(1)
            # HIL-SERL CHANGE: The check_transition call now handles episode termination signals
            # from both the environment (self.done) and human input.
            if self.phase_manager.check_transition():
                # If a phase transition happens (e.g., episode ends), `done` might be updated.
                pass

            if self.key == 27:  # escape key
                self.quit_flag = True
            if self.quit_flag:
                break

        if self.args.result_filename is not None:
            print(
                f"[{self.__class__.__name__}] Save the rollout results: {self.args.result_filename}"
            )
            with open(self.args.result_filename, "w") as result_file:
                yaml.dump(self.result, result_file)

        self.print_statistics()

        if self.args.teleop_device:
            for device in self.input_device_list:
                device.close()

    def get_data_filename(self):
        # MODIFICATION: Changed the directory structure to match the user's request.
        # This removes the timestamped parent folder and saves rollouts into a
        # directory named after the demo/task.
        output_dir = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "dataset",
                self.demo_name,
            )
        )
        
        # The filename now uses the continuously incrementing `episode_idx`,
        # which prevents collisions and matches the desired flat file structure.
        filename = os.path.join(
            output_dir,
            f"{self.demo_name}_world{self.data_manager.world_idx:0>1}_{self.data_manager.episode_idx:0>3}.rmb",
        )
        return filename
        
    def save_data(self):
        filename = self.get_data_filename()
        # DataManager.save_data() increments `episode_idx` by default after saving.
        self.data_manager.save_data(filename)
        
    def print_statistics(self):
        print(f"[{self.__class__.__name__}] Statistics")
        if not hasattr(self, "policy") or self.policy is None:
            print("  - Policy not available for statistics.")
            return

        policy_model_size = self.calc_model_size()
        print(f"  - Policy model size [MB] | {policy_model_size / 1024**2:.2f}")
        gpu_memory_usage = torch.cuda.max_memory_reserved()
        print(f"  - GPU memory usage [GB] | {gpu_memory_usage / 1024**3:.3f}")

        if not self.inference_duration_list:
            print("  - No policy inferences were recorded to calculate frequency.")
            return

        print("  ---")
        print("  - Policy Inference:")
        inference_duration_arr = np.array(self.inference_duration_list)
        min_duration = inference_duration_arr.min()
        max_duration = inference_duration_arr.max()

        print(
            "    - Duration [s] | "
            f"mean: {inference_duration_arr.mean():.2e}, std: {inference_duration_arr.std():.2e}, "
            f"min: {min_duration:.2e}, max: {max_duration:.2e}"
        )
        print(
            "    - Frequency [Hz] | "
            f"mean: {1 / inference_duration_arr.mean():.2f}, "
            f"min: {1 / max_duration:.2f}, max: {1 / min_duration:.2f}"
        )