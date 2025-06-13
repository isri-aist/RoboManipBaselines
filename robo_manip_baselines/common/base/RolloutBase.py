import argparse
import datetime
import os
import pickle
import socket
import sys
import time
from abc import ABC, abstractmethod

import cv2
import matplotlib
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg
from torchvision.transforms import v2

from ..data.DataKey import DataKey
from ..manager.MotionManager import MotionManager
from ..manager.PhaseManager import PhaseManager
from ..utils.DataUtils import normalize_data
from ..utils.MathUtils import set_random_seed
from ..utils.MiscUtils import remove_suffix
from .PhaseBase import PhaseBase


class InitialRolloutPhase(PhaseBase):
    def start(self):
        super().start()

        if self.op.args.wait_before_start:
            print(f"[{self.op.__class__.__name__}] Press the 'n' key to proceed.")

    def check_transition(self):
        if self.op.args.wait_before_start:
            return self.op.key == ord("n")
        else:
            duration = 1.0  # [s]
            return self.get_elapsed_duration() > duration


class RolloutPhase(PhaseBase):
    def start(self):
        super().start()

        self.op.rollout_time_idx = 0
        self.termination_time = None
        print(
            f"[{self.op.__class__.__name__}] Start policy rollout (world_idx: {self.op.world_idx}). "
            "Press the 'n' key to finish policy rollout."
        )

    def pre_update(self):
        if self.op.rollout_time_idx % self.op.args.skip == 0:
            inference_start_time = time.time()
            self.op.infer_policy()
            self.op.inference_duration_list.append(time.time() - inference_start_time)

        self.op.set_command_data()

    def post_update(self):
        if (not self.op.args.no_plot) and (
            self.op.rollout_time_idx % self.op.args.skip_draw == 0
        ):
            self.op.draw_plot()

        self.op.rollout_time_idx += 1

    def check_transition(self):
        elapsed_duration = self.get_elapsed_duration()

        transition_flag = False
        if self.op.key == ord("n"):
            transition_flag = True
        elif self.op.args.auto_exit:
            if self.op.terminated and (self.termination_time is None):
                self.termination_time = elapsed_duration

            post_termination_duration = 3.0  # [s]
            if self.termination_time is not None:
                if elapsed_duration > self.termination_time + post_termination_duration:
                    print(
                        f"[{self.op.__class__.__name__}] Terminate the rollout phase because the environment has been terminated."
                    )
                    transition_flag = True
            else:
                if elapsed_duration > self.op.args.max_duration:
                    print(
                        f"[{self.op.__class__.__name__}] Terminate the rollout phase because the maximum duration has elapsed."
                    )
                    transition_flag = True

        if transition_flag:
            success_str = "success" if self.op.reward > 0.0 else "failure"
            print(
                # Do not change the following print description, as it will be used
                # to automatically obtain the task success/failure result
                f"Rollout result: {success_str}",
                flush=True,
            )

            self.op.result["success"].append(bool(self.op.reward > 0.0))
            self.op.result["reward"].append(self.op.reward)
            self.op.result["duration"].append(elapsed_duration)

            if self.op.args.save_last_image:
                self.op.save_rgb_image()

            return True
        else:
            return False


class EndRolloutPhase(PhaseBase):
    def start(self):
        super().start()

        if not self.op.args.auto_exit:
            print(f"[{self.op.__class__.__name__}] Press the 'n' key to exit.")

    def check_transition(self):
        if (self.op.key == ord("n")) or self.op.args.auto_exit:
            self.op.episode_idx += 1
            if False:  # self.op.episode_idx == len(self.op.args.world_idx_list):
                self.op.quit_flag = True
            else:
                self.op.reset_flag = True

        return False


class RolloutBase(ABC):
    SOCKET_PATH = "/tmp/rmb.sock"

    def __init__(self):
        self.setup_args()

        set_random_seed(self.args.seed)

        self.setup_model_meta_info()

        self.setup_policy()

        render_mode = None if self.args.no_render else "human"
        self.setup_env(render_mode=render_mode)

        if not self.args.no_plot:
            self.setup_plot()

        self.motion_manager = MotionManager(self.env)

        phase_order = [
            InitialRolloutPhase(self),
            *self.get_pre_motion_phases(),
            RolloutPhase(self),
            EndRolloutPhase(self),
        ]
        self.phase_manager = PhaseManager(phase_order)

        self.setup_variables()

        if os.path.exists(self.SOCKET_PATH):
            os.remove(self.SOCKET_PATH)

    def setup_args(self, parser=None, argv=None):
        if parser is None:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )

        parser.add_argument(
            "--checkpoint", type=str, required=True, help="checkpoint file"
        )

        parser.add_argument(
            "--world_idx",
            type=int,
            default=0,
            help="world index (if '--world_idx_list' option is specified, it takes precedence)",
        )
        parser.add_argument(
            "--world_idx_list",
            type=int,
            nargs="*",
            default=None,
            help="list of world indexes",
        )
        parser.add_argument(
            "--world_random_scale",
            nargs="+",
            type=float,
            default=None,
            help="random scale of simulation world (no randomness by default)",
        )

        parser.add_argument(
            "--skip",
            type=int,
            help="step interval to infer policy",
        )
        parser.add_argument(
            "--skip_draw",
            type=int,
            help="step interval to draw the plot",
        )

        parser.add_argument("--seed", type=int, default=-1, help="random seed")

        parser.add_argument(
            "--no_render",
            action="store_true",
            help="whether to disable simulation rendering",
        )
        parser.add_argument(
            "--no_plot", action="store_true", help="whether to disable policy plot"
        )
        parser.add_argument(
            "--win_xy_plot",
            type=int,
            nargs=2,
            help="xy position of window to plot policy information",
        )

        parser.add_argument(
            "--wait_before_start",
            action="store_true",
            help="whether to wait a key input before starting motion",
        )
        parser.add_argument(
            "--auto_exit",
            action="store_true",
            help="whether to automatically exit from rollout",
        )
        parser.add_argument(
            "--max_duration",
            type=float,
            default=30.0,
            help=(
                "maximum rollout duration for automatic exit [s] "
                "(used only when '--auto_exit' option is enabled)"
            ),
        )

        parser.add_argument(
            "--result_filename",
            type=str,
            default=None,
            help="File path (*.yaml) to save rollout results (default: do not save)",
        )

        parser.add_argument(
            "--save_last_image",
            action="store_true",
            help="whether to save the observation image of the last frame",
        )
        parser.add_argument(
            "--output_image_dir",
            type=str,
            default=".",
            help=(
                "directory to save the output image (default: current directory, "
                "used only when '--output_image_dir' option is enabled)."
            ),
        )

        self.set_additional_args(parser)

        if argv is None:
            argv = sys.argv
        self.args = parser.parse_args(argv[1:])

        if self.args.world_idx_list is None:
            self.args.world_idx_list = [self.args.world_idx]

        if self.args.world_random_scale is not None:
            self.args.world_random_scale = np.array(self.args.world_random_scale)

        if self.args.seed < 0:
            self.args.seed = int(time.time()) % (2**32)

    def set_additional_args(self, parser):
        pass

    def setup_model_meta_info(self):
        checkpoint_dir = os.path.split(self.args.checkpoint)[0]
        model_meta_info_path = os.path.join(checkpoint_dir, "model_meta_info.pkl")
        with open(model_meta_info_path, "rb") as f:
            self.model_meta_info = pickle.load(f)
        print(
            f"[{self.__class__.__name__}] Load model meta info: {model_meta_info_path}"
        )

        # Set state and action information
        self.state_keys = self.model_meta_info["state"]["keys"]
        self.action_keys = self.model_meta_info["action"]["keys"]
        self.camera_names = self.model_meta_info["image"]["camera_names"]
        self.state_dim = len(self.model_meta_info["state"]["example"])
        self.action_dim = len(self.model_meta_info["action"]["example"])

        # Set skip if not specified
        if self.args.skip is None:
            self.args.skip = self.model_meta_info["data"]["skip"]
        if self.args.skip_draw is None:
            self.args.skip_draw = self.args.skip

    @abstractmethod
    def setup_policy(self):
        pass

    def setup_env(self):
        raise NotImplementedError(
            f"[{self.__class__.__name__}] This method should be defined in the Operation class and inherited from it."
        )

    def setup_plot(self, fig_ax=None):
        matplotlib.use("agg")

        if fig_ax is None:
            self.fig, self.ax = plt.subplots(
                1, 1, figsize=(13.5, 6.0), dpi=60, squeeze=False
            )
        else:
            self.fig, self.ax = fig_ax

        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        self.canvas = FigureCanvasAgg(self.fig)
        self.canvas.draw()
        cv2.imshow(
            self.policy_name,
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )

        if self.args.win_xy_plot is not None:
            cv2.moveWindow(self.policy_name, *self.args.win_xy_plot)
        cv2.waitKey(1)

        if len(self.action_keys) > 0:
            self.action_plot_scale = np.concatenate(
                [DataKey.get_plot_scale(key, self.env) for key in self.action_keys]
            )
        else:
            self.action_plot_scale = np.zeros(0)

    def setup_variables(self):
        self.image_transforms = v2.ToDtype(torch.float32, scale=True)

        self.episode_idx = 0
        self.datetime_now = datetime.datetime.now()

        self.result = {key: [] for key in ("success", "reward", "duration")}

    def reset_variables(self):
        self.policy_action_list = np.empty((0, self.action_dim))

    def get_pre_motion_phases(self):
        return []

    def print_policy_info(self):
        print(
            f"[{self.__class__.__name__}] Construct {self.policy_name} policy.\n"
            f"  - state dim: {self.state_dim}, action dim: {self.action_dim}, camera num: {len(self.camera_names)}\n"
            f"  - state keys: {self.state_keys}\n"
            f"  - action keys: {self.action_keys}\n"
            f"  - camera names: {self.camera_names}\n"
            f"  - skip: {self.args.skip}"
        )

    def load_ckpt(self, device="cuda"):
        print(f"[{self.__class__.__name__}] Load {self.args.checkpoint}")
        self.device = torch.device(device)
        self.policy.load_state_dict(
            torch.load(
                self.args.checkpoint, map_location=self.device, weights_only=True
            )
        )
        self.policy.to(self.device)
        self.policy.eval()

    def run(self):
        self.reset_flag = True
        self.quit_flag = False
        self.inference_duration_list = []

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(self.SOCKET_PATH)
            server.listen()
            print("[Robot] Robot control process started...")

            conn = None
            while True:
                if self.reset_flag:
                    self.reset()
                    self.reset_flag = False
                    if conn:
                        conn.close()
                    conn, _ = server.accept()
                    print("[Robot] New connection accepted.")

                    # Blocking receive for task description
                    self.args.task_desc = conn.recv(1024).decode("utf-8")
                    print(f'[Robot] Received task description: "{self.args.task_desc}"')

                    # Set non-blocking mode for done signal
                    conn.setblocking(False)

                self.phase_manager.pre_update()

                env_action = np.concatenate(
                    [
                        self.motion_manager.get_command_data(key)
                        for key in self.env.unwrapped.command_keys_for_step
                    ]
                )
                self.obs, self.reward, self.terminated, _, self.info = self.env.step(
                    env_action
                )

                self.phase_manager.post_update()

                self.key = cv2.waitKey(1)

                # Non-blocking check for done signal
                try:
                    done_signal = conn.recv(1024).decode("utf-8")
                    if done_signal:
                        print(f"\n[Robot] Received done signal: {done_signal}")
                        self.key = ord("n")  # force transition
                except BlockingIOError:
                    pass  # nothing received

                self.phase_manager.check_transition()

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

            if conn:
                conn.close()

        self.print_statistics()

        # self.env.close()

    def reset(self):
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

        # Reset environment
        self.env.unwrapped.world_random_scale = self.args.world_random_scale
        world_idx = self.args.world_idx_list[
            self.episode_idx % len(self.args.world_idx_list)
        ]
        self.world_idx = self.env.unwrapped.modify_world(world_idx=world_idx)
        self.obs, self.info = self.env.reset(seed=self.args.seed)

        # Reset phase manager
        self.phase_manager.reset()

        # Reset variables
        self.reset_variables()

    @abstractmethod
    def infer_policy(self):
        pass

    def get_state(self):
        if len(self.state_keys) == 0:
            state = np.zeros(0, dtype=np.float32)
        else:
            state = np.concatenate(
                [
                    self.motion_manager.get_data(state_key, self.obs)
                    for state_key in self.state_keys
                ]
            )

        state = normalize_data(state, self.model_meta_info["state"])
        state = torch.tensor(state[np.newaxis], dtype=torch.float32).to(self.device)

        return state

    def get_images(self):
        # Assume all images are the same size
        images = np.stack(
            [self.info["rgb_images"][camera_name] for camera_name in self.camera_names],
            axis=0,
        )

        images = np.moveaxis(images, -1, -3)
        images = torch.tensor(images, dtype=torch.uint8)
        images = self.image_transforms(images)[torch.newaxis].to(self.device)

        return images

    @abstractmethod
    def draw_plot(self):
        pass

    def set_command_data(self, action_keys=None):
        if action_keys is None:
            action_keys = self.action_keys

        is_skip = self.rollout_time_idx % self.args.skip != 0
        action_idx = 0
        for key in action_keys:
            action_dim = DataKey.get_dim(key, self.env)
            self.motion_manager.set_command_data(
                key,
                self.policy_action[action_idx : action_idx + action_dim],
                is_skip,
            )
            action_idx += action_dim

    def plot_images(self, axes):
        for camera_idx, camera_name in enumerate(self.camera_names):
            axes[camera_idx].imshow(self.info["rgb_images"][camera_name])
            axes[camera_idx].set_title(camera_name, fontsize=20)

    def plot_action(self, ax):
        history_size = 100
        ax.plot(self.policy_action_list[-1 * history_size :] * self.action_plot_scale)
        ax.set_title("action", fontsize=20)
        ax.set_xlabel("step", fontsize=16)
        ax.set_xlim(0, history_size - 1)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.axis("on")

    def print_statistics(self):
        print(f"[{self.__class__.__name__}] Statistics on policy inference")
        policy_model_size = self.calc_model_size()
        print(f"  - Policy model size [MB] | {policy_model_size / 1024**2:.2f}")
        gpu_memory_usage = torch.cuda.max_memory_reserved()
        print(f"  - GPU memory usage [GB] | {gpu_memory_usage / 1024**3:.3f}")
        inference_duration_arr = np.array(self.inference_duration_list)
        print(
            "  - Inference duration [s] | "
            f"mean: {inference_duration_arr.mean():.2e}, std: {inference_duration_arr.std():.2e} "
            f"min: {inference_duration_arr.min():.2e}, max: {inference_duration_arr.max():.2e}"
        )

    def save_rgb_image(self):
        image = cv2.hconcat(list(self.info["rgb_images"].values()))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        demo_name = remove_suffix(self.env.spec.name, "Env")
        success_str = "success" if self.reward > 0.0 else "failure"
        image_path = os.path.abspath(
            os.path.join(
                self.args.output_image_dir,
                (
                    f"Rollout{self.policy_name}_{demo_name}_world{self.world_idx:0>1}_"
                    f"{success_str}_{self.datetime_now:%Y%m%d_%H%M%S}.png"
                ),
            )
        )

        print(
            f"[{self.__class__.__name__}] Save the observation image of the last frame: {image_path}"
        )
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        cv2.imwrite(image_path, image)

    def calc_model_size(self):
        # https://discuss.pytorch.org/t/finding-model-size/130275/2
        param_size = 0
        for param in self.policy.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.policy.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return param_size + buffer_size
