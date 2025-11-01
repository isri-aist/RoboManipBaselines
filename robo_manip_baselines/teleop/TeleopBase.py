import argparse
import datetime
import os
import sys
import time
from abc import ABC

import cv2
import matplotlib.pylab as plt
import numpy as np
import yaml

from robo_manip_baselines.common import (
    DataKey,
    DataManager,
    MotionManager,
    OperationDataMixin,
    PhaseBase,
    PhaseManager,
    convert_depth_image_to_color_image,
    convert_depth_image_to_pointcloud,
    find_rmb_files,
    remove_suffix,
    set_random_seed,
)


class InitialTeleopPhase(PhaseBase):
    def start(self):
        super().start()

        if not self.op.auto_mode:
            print(f"[{self.op.__class__.__name__}] Press the 'n' key to proceed.")

    def check_transition(self):
        return self.op.auto_mode or (self.op.key == ord("n"))


class StandbyTeleopPhase(PhaseBase):
    def start(self):
        super().start()

        for input_device in self.op.input_device_list:
            input_device.connect()
        print(
            f"[{self.op.__class__.__name__}] Press the 'n' key to start teleoperation."
        )

    def post_update(self):
        for input_device in self.op.input_device_list:
            input_device.read()

    def check_transition(self):
        is_ready = all(
            [input_device.is_ready() for input_device in self.op.input_device_list]
        )
        return is_ready and self.op.key == ord("n")


class SyncPhase(PhaseBase):
    def start(self):
        super().start()

        print(
            f"[{self.op.__class__.__name__}] Press the 'n' key to start teleoperation with recording."
        )

    def pre_update(self):
        for input_device in self.op.input_device_list:
            input_device.read()
            input_device.set_command_data()

    def check_transition(self):
        return self.op.key == ord("n")


class TeleopPhase(PhaseBase):
    def start(self):
        super().start()

        self.op.teleop_time_idx = 0
        print(
            f"[{self.op.__class__.__name__}] Press the 'n' key to finish teleoperation."
        )

    def pre_update(self):
        for input_device in self.op.input_device_list:
            input_device.read()
            input_device.set_command_data()

    def post_update(self):
        self.op.teleop_time_idx += 1

    def check_transition(self):
        if self.op.key == ord("n"):
            print(
                f"[{self.op.__class__.__name__}] Finish teleoperation. duration: {self.get_elapsed_duration():.1f} [s]"
            )
            self.op.episode_duration = self.get_elapsed_duration()
            return True
        else:
            return False


class EndTeleopPhase(PhaseBase):
    def start(self):
        super().start()

        if (not self.op.args.save_success_only) or (self.op.reward >= 1.0):
            print(
                f"[{self.op.__class__.__name__}] Press the 's' key if the teleoperation succeeded, or the 'f' key if it failed."
            )
        else:
            print(
                f"[{self.op.__class__.__name__}] Press the 'f' key. (Data cannot be saved in success-only mode.)"
            )

    def post_update(self):
        if ((not self.op.args.save_success_only) or (self.op.reward >= 1.0)) and (
            self.op.key == ord("s")
        ):
            self.op.result["success"].append(bool(self.op.reward >= 1.0))
            self.op.result["reward"].append(float(self.op.reward))
            self.op.result["duration"].append(self.op.episode_duration)
            self.op.save_data()
            self.op.reset_flag = True
        elif self.op.key == ord("f"):
            print(f"[{self.op.__class__.__name__}] Reset without saving the data.")
            self.op.reset_flag = True


class ReplayPhase(PhaseBase):
    def start(self):
        super().start()

        self.op.init_for_relative_command()

        self.op.teleop_time_idx = 0
        print(
            f"[{self.op.__class__.__name__}] Start to replay the log motion. Press the 'h' key to stop replay."
        )

    def pre_update(self):
        for replay_key in self.op.args.replay_keys:
            self.op.motion_manager.set_command_data(
                replay_key,
                self.op.replay_data_manager.get_single_data(
                    replay_key, self.op.teleop_time_idx
                ),
            )

    def post_update(self):
        self.op.teleop_time_idx += 1

    def check_transition(self):
        if self.op.key == ord("h") or self.op.teleop_time_idx == len(
            self.op.replay_data_manager.get_data_seq(DataKey.TIME)
        ):
            self.op.episode_duration = self.get_elapsed_duration()
            return True
        else:
            return False


class EndReplayPhase(PhaseBase):
    def start(self):
        super().start()

        msg = f"[{self.op.__class__.__name__}] Replay of the log is finished."
        if not self.op.auto_mode:
            msg += " Press the 'n' key to reset."
        print(msg)

    def post_update(self):
        if self.op.auto_mode or (self.op.key == ord("n")):
            self.op.result["success"].append(bool(self.op.reward >= 1.0))
            self.op.result["reward"].append(float(self.op.reward))
            self.op.result["duration"].append(self.op.episode_duration)
            if self.op.args.save_replay:
                self.op.save_data()
            self.op.replay_file_idx += 1
            if self.op.replay_file_idx == len(self.op.replay_filenames):
                self.op.quit_flag = True
            else:
                self.op.reset_flag = True


class TeleopBase(OperationDataMixin, ABC):
    MotionManagerClass = MotionManager
    DataManagerClass = DataManager

    def __init__(self):
        # Setup arguments
        self.setup_args()

        set_random_seed(self.args.seed)

        # Setup gym environment
        self.setup_env()
        self.demo_name = self.args.demo_name or remove_suffix(self.env.spec.name, "Env")
        self.env.reset(seed=self.args.seed)
        if self.args.target_task is not None:
            self.env.unwrapped.target_task = self.args.target_task
        if self.args.world_random_factors is not None:
            self.env.unwrapped.world_random_factors = self.args.world_random_factors

        # Setup motion manager
        self.motion_manager = self.MotionManagerClass(self.env)

        # Setup data manager
        self.data_manager = self.DataManagerClass(
            self.env, demo_name=self.demo_name, task_desc=self.args.task_desc
        )
        self.data_manager.setup_camera_info()
        self.datetime_now = datetime.datetime.now()
        self.result = {key: [] for key in ("success", "reward", "duration")}

        if self.args.replay_log is not None:
            # Setup data manager for replay
            self.replay_data_manager = DataManager(self.env, demo_name=self.demo_name)
            self.replay_data_manager.setup_camera_info()

            # Set log files for replay
            self.replay_filenames = find_rmb_files(self.args.replay_log)
            self.replay_filenames *= self.args.replay_repeat_count
            self.replay_file_idx = 0

        # Setup phase manager
        self.setup_phase_manager()

        # Setup plot
        if self.args.plot_pointcloud:
            plt.rcParams["keymap.quit"] = ["q", "escape"]
            fig, self.ax_3d = plt.subplots(
                len(self.env.unwrapped.camera_names),
                1,
                subplot_kw=dict(projection="3d"),
                constrained_layout=True,
            )
            fig.tight_layout()
            self.pointcloud_scatter_list = [None] * len(self.env.unwrapped.camera_names)

        if self.args.plot_tactile:
            if len(self.env.unwrapped.intensity_tactile_names) > 0:
                plt.rcParams["keymap.quit"] = ["q", "escape"]
                fig, self.ax_tactile = plt.subplots(
                    len(self.env.unwrapped.intensity_tactile_names),
                    1,
                    constrained_layout=True,
                )
            else:
                raise RuntimeError(
                    f"[{self.__class__.__name__}] The '--plot_tactile' option was specified "
                    "but no tactile sensor with intensity output was found."
                )

        # Setup input device
        if self.args.input_device_config is None:
            input_device_kwargs = {}
        else:
            with open(self.args.input_device_config, "r") as f:
                input_device_kwargs = yaml.safe_load(f)
        if self.args.replay_log is None:
            self.input_device_list = self.env.unwrapped.setup_input_device(
                self.args.input_device, self.motion_manager, input_device_kwargs
            )

    def setup_args(self, parser=None, argv=None):
        if parser is None:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )

        parser.add_argument(
            "--demo_name", type=str, default="", help="demonstration name"
        )
        parser.add_argument(
            "--target_task", type=str, default=None, help="target task name"
        )
        parser.add_argument(
            "--task_desc", type=str, default="", help="task description"
        )

        parser.add_argument(
            "--file_format",
            type=str,
            default="rmb",
            choices=["rmb", "hdf5"],
            help="file format to save ('rmb' or 'hdf5')",
        )
        parser.add_argument(
            "--save_success_only",
            action="store_true",
            help="whether to save data only when the task succeeds",
        )

        parser.add_argument(
            "--result_filename",
            type=str,
            default=None,
            help="File path (*.yaml) to save rollout results (default: do not save)",
        )

        parser.add_argument(
            "--input_device",
            type=str,
            default="spacemouse",
            choices=["spacemouse", "gello", "keyboard"],
            help="input device for teleoperation",
        )
        parser.add_argument(
            "--input_device_config", type=str, help="configuration file of input device"
        )

        parser.add_argument(
            "--sync_before_record",
            action="store_true",
            help="whether to synchronize with input device before starting record",
        )

        parser.add_argument(
            "--plot_pointcloud", action="store_true", help="whether to plot point cloud"
        )
        parser.add_argument(
            "--plot_tactile",
            action="store_true",
            help="whether to plot tactile sensor measurements",
        )

        parser.add_argument(
            "--world_idx_list",
            type=int,
            nargs="*",
            help="list of world indexes (if not given, loop through all world indicies)",
        )
        parser.add_argument(
            "--world_random_scale",
            nargs="+",
            type=float,
            default=None,
            help="random scale of simulation world (no randomness by default)",
        )
        parser.add_argument(
            "--world_random_factors",
            nargs="*",
            type=str,
            default=None,
            help="list of randomization factors applied to simulation world (no randomness by default)",
        )

        parser.add_argument(
            "--replay_log",
            type=str,
            default=None,
            help="log file path when replaying log motion",
        )
        parser.add_argument(
            "--replay_repeat_count",
            type=int,
            default=1,
            help="number of times to repeat replay per file",
        )
        parser.add_argument(
            "--auto_replay",
            action="store_true",
            help="whether replay proceeds automatically",
        )
        parser.add_argument(
            "--save_replay",
            action="store_true",
            help="whether to save replayed data",
        )
        parser.add_argument(
            "--replay_keys",
            nargs="+",
            choices=DataKey.COMMAND_DATA_KEYS,
            default=None,
            help="command data keys when replaying log motion",
        )

        parser.add_argument("--seed", type=int, default=-1, help="random seed")

        self.set_additional_args(parser)

        if argv is None:
            argv = sys.argv
        self.args = parser.parse_args(argv[1:])

        if self.args.world_random_scale is not None:
            self.args.world_random_scale = np.array(self.args.world_random_scale)

        self.auto_mode = (self.args.replay_log is not None) and self.args.auto_replay

        if self.args.seed < 0:
            self.args.seed = int(time.time()) % (2**32)

    def set_additional_args(self, parser):
        pass

    def setup_env(self):
        raise NotImplementedError(
            f"[{self.__class__.__name__}] This method should be defined in the Operation class and inherited from it."
        )

    def setup_phase_manager(self):
        if self.args.replay_log is None:
            operation_phases = [
                StandbyTeleopPhase(self),
                TeleopPhase(self),
                EndTeleopPhase(self),
            ]
            if self.args.sync_before_record:
                operation_phases.insert(1, SyncPhase(self))
        else:
            operation_phases = [ReplayPhase(self), EndReplayPhase(self)]
        phase_order = [
            InitialTeleopPhase(self),
            *self.get_pre_motion_phases(),
            *operation_phases,
        ]
        self.phase_manager = PhaseManager(phase_order)

        def get_text_func(phase):
            text = remove_suffix(phase.name, "Phase")
            if self.reward >= 1.0:
                text += " (success)"
            return text

        def get_color_func(phase):
            if phase.name in ("InitialTeleopPhase", "StandbyTeleopPhase"):
                return np.array([200, 200, 255])
            elif phase.name in ("SyncPhase"):
                return np.array([255, 255, 200])
            elif phase.name in ("TeleopPhase", "ReplayPhase"):
                return np.array([255, 200, 200])
            elif phase.name in ("EndTeleopPhase", "EndReplayPhase"):
                return np.array([200, 200, 200])
            else:
                return np.array([200, 255, 200])

        self.phase_manager.get_text_func = get_text_func
        self.phase_manager.get_color_func = get_color_func

    def get_pre_motion_phases(self):
        return []

    def run(self):
        self.reset_flag = True
        self.quit_flag = False
        self.iteration_duration_list = []

        while True:
            iteration_start_time = time.time()

            if self.reset_flag:
                self.reset()
                self.reset_flag = False

            self.phase_manager.pre_update()
            self.motion_manager.draw_markers()

            action = np.concatenate(
                [
                    self.motion_manager.get_command_data(key)
                    for key in self.env.unwrapped.command_keys_for_step
                ]
            )

            if self.phase_manager.is_phases(["TeleopPhase", "ReplayPhase"]):
                self.record_data()

            self.obs, self.reward, _, _, self.info = self.env.step(action)

            self.draw_image()

            if self.args.plot_pointcloud:
                self.draw_pointcloud()

            if self.args.plot_tactile:
                self.draw_tactile()

            self.phase_manager.post_update()

            self.key = cv2.waitKey(1)
            self.phase_manager.check_transition()

            if self.key == 27:  # escape key
                self.quit_flag = True
            if self.quit_flag:
                break

            iteration_duration = time.time() - iteration_start_time
            if self.phase_manager.is_phases(["TeleopPhase", "ReplayPhase"]) and (
                self.teleop_time_idx > 0
            ):
                self.iteration_duration_list.append(iteration_duration)

            if (not self.auto_mode) and (iteration_duration < self.env.unwrapped.dt):
                time.sleep(self.env.unwrapped.dt - iteration_duration)

        if self.args.result_filename is not None:
            print(
                f"[{self.__class__.__name__}] Save the teleoperation results: {self.args.result_filename}"
            )
            with open(self.args.result_filename, "w") as result_file:
                yaml.dump(self.result, result_file)

        self.print_statistics()

        if self.args.replay_log is None:
            for input_device in self.input_device_list:
                input_device.close()

        # self.env.close()

    def reset(self):
        # Reset motion manager
        self.motion_manager.reset()

        # Reset data manager
        self.data_manager.reset()
        if self.args.world_idx_list is None:
            world_idx = None
        else:
            world_idx = self.args.world_idx_list[
                self.data_manager.episode_idx % len(self.args.world_idx_list)
            ]

        # Load replay data
        if self.args.replay_log is not None:
            self.replay_data_manager.reset()
            if self.args.replay_keys is None:
                self.args.replay_keys = self.env.unwrapped.command_keys_for_step
            replay_file = self.replay_filenames[self.replay_file_idx]
            self.replay_data_manager.load_data(replay_file, skip_image=True)
            print(
                f"[{self.__class__.__name__}] Load teleoperation data "
                f"({self.replay_file_idx+1}/{len(self.replay_filenames)}): {replay_file}\n"
                f"  - replay keys: {self.args.replay_keys}"
            )
            world_idx = self.replay_data_manager.get_meta_data("world_idx")

        # Reset environment
        self.env.unwrapped.world_random_scale = self.args.world_random_scale
        self.data_manager.setup_env_world(world_idx)
        self.env.reset(seed=self.args.seed)
        print(
            f"[{self.__class__.__name__}] Reset environment. demo_name: {self.demo_name}, world_idx: {self.data_manager.world_idx}, episode_idx: {self.data_manager.episode_idx}"
        )

        # Reset phase manager
        self.phase_manager.reset()

    def init_for_relative_command(self):
        for key in [
            DataKey.COMMAND_JOINT_POS_REL,
            DataKey.COMMAND_GRIPPER_JOINT_POS_REL,
            DataKey.COMMAND_EEF_POSE_REL,
        ]:
            if key not in self.args.replay_keys:
                continue

            abs_key = DataKey.get_abs_key(key)
            self.motion_manager.set_command_data(
                abs_key,
                self.replay_data_manager.get_single_data(abs_key, 0),
            )

    def save_data(self):
        filename = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "dataset",
                f"{self.demo_name}_{self.datetime_now:%Y%m%d_%H%M%S}",
                f"{self.demo_name}_world{self.data_manager.world_idx:0>1}_{self.data_manager.episode_idx:0>3}.{self.args.file_format}",
            )
        )
        self.data_manager.save_data(filename)
        print(f"[{self.__class__.__name__}] Save the data as {filename}")

    def draw_image(self):
        phase_image = self.phase_manager.get_phase_image(
            get_text_func=self.phase_manager.get_text_func,
            get_color_func=self.phase_manager.get_color_func,
        )
        rgb_images = []
        depth_images = []
        for camera_name in (
            self.env.unwrapped.camera_names
            + self.env.unwrapped.rgb_tactile_names
            + self.env.unwrapped.pointcloud_camera_names
        ):
            rgb_image = self.info["rgb_images"][camera_name]
            image_ratio = rgb_image.shape[1] / rgb_image.shape[0]
            resized_image_width = phase_image.shape[1] / 2
            resized_image_size = (
                int(resized_image_width),
                int(resized_image_width / image_ratio),
            )
            rgb_images.append(cv2.resize(rgb_image, resized_image_size))
            if camera_name in self.env.unwrapped.rgb_tactile_names:
                depth_images.append(
                    np.full(resized_image_size[::-1] + (3,), 255, dtype=np.uint8)
                )
            else:
                depth_image = convert_depth_image_to_color_image(
                    self.info["depth_images"][camera_name]
                )
                depth_images.append(cv2.resize(depth_image, resized_image_size))
        window_image = cv2.vconcat(
            (
                cv2.hconcat((cv2.vconcat(rgb_images), cv2.vconcat(depth_images))),
                phase_image,
            )
        )
        cv2.namedWindow(
            "image",
            flags=(cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL),
        )
        cv2.imshow("image", cv2.cvtColor(window_image, cv2.COLOR_RGB2BGR))

    def draw_pointcloud(self):
        far_clip_list = (3.0, 3.0, 0.8)  # [m]
        for camera_idx, (camera_name, ax) in enumerate(
            zip(self.env.unwrapped.camera_names, self.ax_3d)
        ):
            pointcloud_skip = 10
            small_depth_image = self.info["depth_images"][camera_name][
                ::pointcloud_skip, ::pointcloud_skip
            ]
            small_rgb_image = self.info["rgb_images"][camera_name][
                ::pointcloud_skip, ::pointcloud_skip
            ]
            fovy = self.data_manager.get_meta_data(
                DataKey.get_depth_image_key(camera_name) + "_fovy"
            )
            xyz_array, rgb_array = convert_depth_image_to_pointcloud(
                small_depth_image,
                fovy=fovy,
                rgb_image=small_rgb_image,
                far_clip=far_clip_list[camera_idx],
            )
            if self.pointcloud_scatter_list[camera_idx] is None:

                def get_min_max(v_min, v_max):
                    return (
                        0.75 * v_min + 0.25 * v_max,
                        0.25 * v_min + 0.75 * v_max,
                    )

                ax.view_init(elev=-90, azim=-90)
                ax.set_xlim(*get_min_max(xyz_array[:, 0].min(), xyz_array[:, 0].max()))
                ax.set_ylim(*get_min_max(xyz_array[:, 1].min(), xyz_array[:, 1].max()))
                ax.set_zlim(*get_min_max(xyz_array[:, 2].min(), xyz_array[:, 2].max()))
            else:
                self.pointcloud_scatter_list[camera_idx].remove()
            ax.axis("off")
            ax.set_box_aspect(np.ptp(xyz_array, axis=0))
            self.pointcloud_scatter_list[camera_idx] = ax.scatter(
                xyz_array[:, 0], xyz_array[:, 1], xyz_array[:, 2], c=rgb_array
            )
        plt.draw()
        plt.pause(0.001)

    def draw_tactile(self, vmin=-50.0, vmax=50.0):
        for tactile_name, ax in zip(
            self.env.unwrapped.intensity_tactile_names, self.ax_tactile
        ):
            tactile_data = self.info["intensity_tactile"][tactile_name]
            ax.clear()
            ax.axis("off")
            ax.imshow(
                np.clip(tactile_data, vmin, vmax),
                cmap="coolwarm",
                interpolation="none",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(tactile_name)
        plt.draw()
        plt.pause(0.001)

    def print_statistics(self):
        print(f"[{self.__class__.__name__}] Statistics on teleoperation")
        if len(self.iteration_duration_list) > 0:
            iteration_duration_arr = np.array(self.iteration_duration_list)
            print(
                f"  - Real-time factor | {self.env.unwrapped.dt / iteration_duration_arr.mean():.2f}"
            )
            print(
                "  - Iteration duration [s] | "
                f"mean: {iteration_duration_arr.mean():.3f}, std: {iteration_duration_arr.std():.3f} "
                f"min: {iteration_duration_arr.min():.3f}, max: {iteration_duration_arr.max():.3f}"
            )
