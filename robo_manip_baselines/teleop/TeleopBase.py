import argparse
import datetime
import glob
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
    PhaseBase,
    PhaseManager,
    convert_depth_image_to_color_image,
    convert_depth_image_to_point_cloud,
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
            return True
        else:
            return False


class EndTeleopPhase(PhaseBase):
    def start(self):
        super().start()

        if (not self.op.args.save_success_only) or (self.op.reward > 0.0):
            print(
                f"[{self.op.__class__.__name__}] Press the 's' key if the teleoperation succeeded, or the 'f' key if it failed."
            )
        else:
            print(
                f"[{self.op.__class__.__name__}] Press the 'f' key. (Data cannot be saved in success-only mode.)"
            )

    def post_update(self):
        if ((not self.op.args.save_success_only) or (self.op.reward > 0.0)) and (
            self.op.key == ord("s")
        ):
            self.op.save_data()
            self.op.reset_flag = True
        elif self.op.key == ord("f"):
            print(
                f"[{self.op.__class__.__name__}] Teleoperation has failed. Reset without saving."
            )
            self.op.reset_flag = True


class ReplayPhase(PhaseBase):
    def start(self):
        super().start()

        self.op.teleop_time_idx = 0
        print(f"[{self.op.__class__.__name__}] Start to replay the log motion.")

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
        return self.op.teleop_time_idx == len(
            self.op.replay_data_manager.get_data_seq(DataKey.TIME)
        )


class EndReplayPhase(PhaseBase):
    def start(self):
        super().start()

        msg = f"[{self.op.__class__.__name__}] Replay of the log is finished."
        if not self.op.auto_mode:
            msg += " Press the 'n' key to reset."
        print(msg)

    def post_update(self):
        if self.op.auto_mode or (self.op.key == ord("n")):
            if self.op.args.save_replay:
                self.op.save_data()
            self.op.replay_file_idx += 1
            if self.op.replay_file_idx == len(self.op.replay_filenames):
                self.op.quit_flag = True
            else:
                self.op.reset_flag = True

        return False


class TeleopBase(ABC):
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

        # Setup motion manager
        self.motion_manager = self.MotionManagerClass(self.env)

        # Setup data manager
        self.data_manager = self.DataManagerClass(
            self.env, demo_name=self.demo_name, task_desc=self.args.task_desc
        )
        self.data_manager.setup_camera_info()
        self.datetime_now = datetime.datetime.now()

        if self.args.replay_log is not None:
            # Setup data manager for replay
            self.replay_data_manager = DataManager(self.env, demo_name=self.demo_name)
            self.replay_data_manager.setup_camera_info()

            # Set log files for replay
            if self.args.replay_log.rstrip("/").endswith((".rmb", ".hdf5")):
                self.replay_filenames = [self.args.replay_log]
            elif os.path.isdir(self.args.replay_log):
                self.replay_filenames = sorted(
                    [
                        f
                        for f in glob.glob(
                            f"{self.args.replay_log}/**/*.*", recursive=True
                        )
                        if f.endswith(".rmb")
                        or (f.endswith(".hdf5") and not f.endswith(".rmb.hdf5"))
                    ]
                )
            else:
                raise ValueError(
                    f"[{self.__class__.__name__}] Invalid path for replaying log: {self.args.replay_log}"
                )
            self.replay_file_idx = 0

        # Setup phase manager
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

        # Setup 3D plot
        if self.args.enable_3d_plot:
            plt.rcParams["keymap.quit"] = ["q", "escape"]
            self.fig, self.ax = plt.subplots(
                len(self.env.unwrapped.camera_names),
                1,
                subplot_kw=dict(projection="3d"),
            )
            self.fig.tight_layout()
            self.point_cloud_scatter_list = [None] * len(
                self.env.unwrapped.camera_names
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
            "--task_desc", type=str, default="", help="task_description"
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
            "--enable_3d_plot", action="store_true", help="whether to enable 3d plot"
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
            "--replay_log",
            type=str,
            default=None,
            help="log file path when replaying log motion",
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

        if argv is None:
            argv = sys.argv
        self.args = parser.parse_args(argv[1:])

        if self.args.world_random_scale is not None:
            self.args.world_random_scale = np.array(self.args.world_random_scale)

        self.auto_mode = (self.args.replay_log is not None) and self.args.auto_replay

        if self.args.seed < 0:
            self.args.seed = int(time.time()) % (2**32)

    def setup_env(self):
        raise NotImplementedError(
            f"[{self.__class__.__name__}] This method should be defined in the Operation class and inherited from it."
        )

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

            if self.args.enable_3d_plot:
                self.draw_point_cloud()

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

            # Set initial joint position for relative command
            if not set(self.args.replay_keys).isdisjoint(
                [DataKey.COMMAND_JOINT_POS_REL, DataKey.COMMAND_EEF_POSE_REL]
            ):
                self.motion_manager.set_command_data(
                    DataKey.COMMAND_JOINT_POS,
                    self.replay_data_manager.get_single_data(
                        DataKey.COMMAND_JOINT_POS, 0
                    ),
                )

        # Reset environment
        self.env.unwrapped.world_random_scale = self.args.world_random_scale
        self.data_manager.setup_env_world(world_idx)
        self.env.reset(seed=self.args.seed)
        print(
            f"[{self.__class__.__name__}] Reset environment. demo_name: {self.demo_name}, world_idx: {self.data_manager.world_idx}, episode_idx: {self.data_manager.episode_idx}"
        )

        # Reset phase manager
        self.phase_manager.reset()

    def record_data(self):
        # Add time
        self.data_manager.append_single_data(
            DataKey.TIME, self.phase_manager.phase.get_elapsed_duration()
        )

        # Add reward
        self.data_manager.append_single_data(DataKey.REWARD, self.reward)

        # Add measured data
        for key in self.env.unwrapped.measured_keys_to_save:
            self.data_manager.append_single_data(
                key, self.motion_manager.get_measured_data(key, self.obs)
            )

        # Add command data
        for key in self.env.unwrapped.command_keys_to_save:
            self.data_manager.append_single_data(
                key, self.motion_manager.get_command_data(key)
            )

        # Add relative data
        for key in (
            DataKey.MEASURED_JOINT_POS_REL,
            DataKey.COMMAND_JOINT_POS_REL,
            DataKey.MEASURED_EEF_POSE_REL,
            DataKey.COMMAND_EEF_POSE_REL,
        ):
            self.data_manager.append_single_data(
                key, self.data_manager.calc_rel_data(key)
            )

        # Add image
        for camera_name in self.env.unwrapped.camera_names:
            self.data_manager.append_single_data(
                DataKey.get_rgb_image_key(camera_name),
                self.info["rgb_images"][camera_name],
            )
            self.data_manager.append_single_data(
                DataKey.get_depth_image_key(camera_name),
                self.info["depth_images"][camera_name],
            )
        for tactile_name in self.env.unwrapped.tactile_names:
            self.data_manager.append_single_data(
                DataKey.get_rgb_image_key(tactile_name),
                self.info["rgb_images"][tactile_name],
            )

    def draw_image(self):
        def get_text_func(phase):
            text = remove_suffix(phase.name, "Phase")
            if self.reward > 0.0:
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

        phase_image = self.phase_manager.get_phase_image(
            get_text_func=get_text_func, get_color_func=get_color_func
        )
        rgb_images = []
        depth_images = []
        for camera_name in (
            self.env.unwrapped.camera_names + self.env.unwrapped.tactile_names
        ):
            rgb_image = self.info["rgb_images"][camera_name]
            image_ratio = rgb_image.shape[1] / rgb_image.shape[0]
            resized_image_width = phase_image.shape[1] / 2
            resized_image_size = (
                int(resized_image_width),
                int(resized_image_width / image_ratio),
            )
            rgb_images.append(cv2.resize(rgb_image, resized_image_size))
            if camera_name in self.env.unwrapped.tactile_names:
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

    def draw_point_cloud(self):
        far_clip_list = (3.0, 3.0, 0.8)  # [m]
        for camera_idx, camera_name in enumerate(self.env.unwrapped.camera_names):
            point_cloud_skip = 10
            small_depth_image = self.info["depth_images"][camera_name][
                ::point_cloud_skip, ::point_cloud_skip
            ]
            small_rgb_image = self.info["rgb_images"][camera_name][
                ::point_cloud_skip, ::point_cloud_skip
            ]
            fovy = self.data_manager.get_meta_data(
                DataKey.get_depth_image_key(camera_name) + "_fovy"
            )
            xyz_array, rgb_array = convert_depth_image_to_point_cloud(
                small_depth_image,
                fovy=fovy,
                rgb_image=small_rgb_image,
                far_clip=far_clip_list[camera_idx],
            )
            if self.point_cloud_scatter_list[camera_idx] is None:

                def get_min_max(v_min, v_max):
                    return (
                        0.75 * v_min + 0.25 * v_max,
                        0.25 * v_min + 0.75 * v_max,
                    )

                self.ax[camera_idx].view_init(elev=-90, azim=-90)
                self.ax[camera_idx].set_xlim(
                    *get_min_max(xyz_array[:, 0].min(), xyz_array[:, 0].max())
                )
                self.ax[camera_idx].set_ylim(
                    *get_min_max(xyz_array[:, 1].min(), xyz_array[:, 1].max())
                )
                self.ax[camera_idx].set_zlim(
                    *get_min_max(xyz_array[:, 2].min(), xyz_array[:, 2].max())
                )
            else:
                self.point_cloud_scatter_list[camera_idx].remove()
            self.ax[camera_idx].axis("off")
            self.ax[camera_idx].set_box_aspect(np.ptp(xyz_array, axis=0))
            self.point_cloud_scatter_list[camera_idx] = self.ax[camera_idx].scatter(
                xyz_array[:, 0], xyz_array[:, 1], xyz_array[:, 2], c=rgb_array
            )
        plt.draw()
        plt.pause(0.001)

    def save_data(self, filename=None):
        if filename is None:
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
        print(
            f"[{self.__class__.__name__}] Teleoperation succeeded: Save the data as {filename}"
        )

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
