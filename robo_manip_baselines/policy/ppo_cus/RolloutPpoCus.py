import argparse
import csv
import json
import os
import sys
import time
import types
from collections import defaultdict
from pathlib import Path
import queue
import threading
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.pylab as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from robo_manip_baselines.common import (
    DataKey,
    RolloutBase,
    denormalize_data,
    normalize_data,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_APRILTAG_SRC = _REPO_ROOT / "external" / "check_AprilTag" / "src"
if _APRILTAG_SRC.exists():
    apriltag_src_str = str(_APRILTAG_SRC)
    if apriltag_src_str not in sys.path:
        sys.path.append(apriltag_src_str)

try:
    from pose_viewer import (
        build_detector,
        build_homogeneous_transform,
        solve_tag_poses,
    )
except ImportError:  # pragma: no cover - optional dependency
    build_detector = None  # type: ignore[assignment]
    build_homogeneous_transform = None  # type: ignore[assignment]
    solve_tag_poses = None  # type: ignore[assignment]


_DEFAULT_T_BASE_TO_CAMERA_PATH = (
    _REPO_ROOT / "robo_manip_baselines" / "calib" / "T_base_to_camera.csv"
)


def _load_base_to_camera_transform(path: Path) -> Optional[np.ndarray]:
    try:
        matrix = np.loadtxt(path, delimiter=",", dtype=np.float64)
    except FileNotFoundError:
        print(
            f"[RolloutPpoCus] T_base→camera transform not found at {path}. "
            "Marker detection will be disabled.",
            flush=True,
        )
        return None
    except Exception as exc:  # pragma: no cover - I/O error
        print(
            f"[RolloutPpoCus] Failed to load T_base→camera transform from {path}: {exc}",
            flush=True,
        )
        return None

    matrix = matrix.reshape(4, 4)
    return matrix.astype(np.float64)


_GLOBAL_T_BASE_TO_CAMERA = _load_base_to_camera_transform(_DEFAULT_T_BASE_TO_CAMERA_PATH)

DEFAULT_TAG_SIZE_M = 0.0309
DEFAULT_DETECTOR_THREADS = 4
DEFAULT_DETECTOR_DECIMATE = 1.0
DEFAULT_DETECTOR_SIGMA = 0.3
DEFAULT_DETECTOR_SHARPENING = 0.1


class FrontCameraFrameWorker:
    """Maintain the latest grayscale frame and AprilTag transforms from the front camera."""

    def __init__(
        self,
        camera,
        base_to_camera: Optional[np.ndarray],
        frame_size: Tuple[int, int] = (640, 480),
        roi: Optional[Tuple[int, int, int, int]] = None,
        tag_size_m: float = DEFAULT_TAG_SIZE_M,
    ):
        self._camera = camera
        self._base_to_camera = base_to_camera
        self._frame_size = frame_size
        self._roi = roi
        self._tag_size_m = float(tag_size_m)
        self._frame_queue = queue.Queue(maxsize=1)
        self._result_queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._processing_thread: Optional[threading.Thread] = None
        self._latest_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: Optional[float] = None
        self._latest_transforms: Dict[int, np.ndarray] = {}
        self._latest_transforms_timestamp: Optional[float] = None
        self._detector = None
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs: Optional[np.ndarray] = None
        self._detection_available = (
            build_detector is not None
            and build_homogeneous_transform is not None
            and solve_tag_poses is not None
            and base_to_camera is not None
        )

    def start(self):
        if self._capture_thread and self._capture_thread.is_alive():
            return

        if not self._detection_available:
            print(
                "[FrontCameraFrameWorker] Marker detection disabled (missing dependencies or calibration).",
                flush=True,
            )

        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="front_camera_capture",
            daemon=True,
        )
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            name="front_camera_process",
            daemon=True,
        )
        self._capture_thread.start()
        self._processing_thread.start()

    def stop(self):
        self._stop_event.set()
        # Unblock the processing thread if it is waiting on the queue.
        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(None)
            except queue.Full:
                pass

        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)
            self._capture_thread = None
        if self._processing_thread:
            self._processing_thread.join(timeout=1.0)
            self._processing_thread = None

        # Drain result queue to avoid stale entries.
        while True:
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        with self._latest_lock:
            if self._latest_frame is None:
                return None, None
            return self._latest_frame.copy(), self._latest_timestamp

    def get_latest_transforms(self) -> Tuple[Dict[int, np.ndarray], Optional[float]]:
        with self._latest_lock:
            return (
                {tag_id: matrix.copy() for tag_id, matrix in self._latest_transforms.items()},
                self._latest_transforms_timestamp,
            )

    def poll_transforms(self) -> Tuple[Optional[Dict[int, np.ndarray]], Optional[float]]:
        try:
            timestamp, transforms = self._result_queue.get_nowait()
        except queue.Empty:
            return None, None

        copied = {tag_id: matrix.copy() for tag_id, matrix in transforms.items()}
        with self._latest_lock:
            self._latest_transforms = copied
            self._latest_transforms_timestamp = timestamp
        return copied, timestamp

    def _capture_loop(self):
        while not self._stop_event.is_set():
            try:
                rgb_image, _ = self._camera.read(self._frame_size)
            except Exception:
                continue

            if rgb_image is None:
                continue

            if self._roi is not None:
                x, y, w, h = self._roi
                x = max(0, int(x))
                y = max(0, int(y))
                w = max(1, int(w))
                h = max(1, int(h))
                max_y = min(y + h, rgb_image.shape[0])
                max_x = min(x + w, rgb_image.shape[1])
                rgb_image = rgb_image[y:max_y, x:max_x]
                if rgb_image.size == 0:
                    continue

            gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

            try:
                self._frame_queue.put_nowait(gray_image)
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._frame_queue.put_nowait(gray_image)
                except queue.Full:
                    continue

    def _processing_loop(self):
        while not self._stop_event.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if frame is None:
                break

            timestamp = time.time()
            transforms: Dict[int, np.ndarray] = {}

            if self._detection_available:
                detector = self._ensure_detector()
                camera_mats = self._ensure_camera_parameters(frame)
                if detector is not None and camera_mats is not None:
                    K, dist_coeffs = camera_mats
                    try:
                        poses = solve_tag_poses(
                            detector,
                            frame,
                            K,
                            dist_coeffs,
                            self._tag_size_m,
                        )
                    except Exception as exc:  # pragma: no cover - detector failure
                        print(f"[FrontCameraFrameWorker] solve_tag_poses failed: {exc}", flush=True)
                        poses = []

                    for pose in poses:
                        try:
                            T_cam_to_tag = build_homogeneous_transform(pose.rvec, pose.tvec)
                        except Exception as exc:  # pragma: no cover
                            print(
                                f"[FrontCameraFrameWorker] Failed to build transform for tag {pose.tag_id}: {exc}",
                                flush=True,
                            )
                            continue
                        T_base_to_tag = self._base_to_camera @ T_cam_to_tag
                        transforms[int(pose.tag_id)] = T_base_to_tag

            with self._latest_lock:
                self._latest_frame = frame.copy()
                self._latest_timestamp = timestamp
                self._latest_transforms = {
                    tag_id: matrix.copy() for tag_id, matrix in transforms.items()
                }
                self._latest_transforms_timestamp = timestamp

            payload = (
                timestamp,
                {tag_id: matrix.copy() for tag_id, matrix in transforms.items()},
            )
            try:
                self._result_queue.put_nowait(payload)
            except queue.Full:
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._result_queue.put_nowait(payload)
                except queue.Full:
                    pass

    def _ensure_detector(self):
        if self._detector is not None:
            return self._detector
        if build_detector is None:
            return None
        try:
            self._detector = build_detector(
                "tag36h11",
                nthreads=DEFAULT_DETECTOR_THREADS,
                quad_decimate=DEFAULT_DETECTOR_DECIMATE,
                quad_sigma=DEFAULT_DETECTOR_SIGMA,
                refine_edges=True,
                decode_sharpening=DEFAULT_DETECTOR_SHARPENING,
            )
        except Exception as exc:  # pragma: no cover - detector creation failure
            print(f"[FrontCameraFrameWorker] Failed to build detector: {exc}", flush=True)
            self._detection_available = False
            return None
        return self._detector

    def _ensure_camera_parameters(
        self, frame: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._camera_matrix is not None and self._dist_coeffs is not None:
            return self._camera_matrix, self._dist_coeffs

        camera_params = self._extract_camera_parameters(frame)
        if camera_params is None:
            print(
                "[FrontCameraFrameWorker] Unable to determine camera intrinsics; disabling detection.",
                flush=True,
            )
            self._detection_available = False
            return None

        self._camera_matrix, self._dist_coeffs = camera_params
        return camera_params

    def _extract_camera_parameters(
        self, frame: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        height, width = frame.shape[:2]
        intr_candidate_attrs = (
            "color_intrinsics",
            "intrinsics",
            "color_intrinsic",
            "intrinsic",
        )
        for attr in intr_candidate_attrs:
            intr = getattr(self._camera, attr, None)
            if intr is None:
                continue

            def _get_intr_value(name: str):
                if hasattr(intr, name):
                    return getattr(intr, name)
                if isinstance(intr, dict):
                    return intr.get(name)
                if hasattr(intr, "__getitem__"):
                    try:
                        return intr[name]  # type: ignore[index]
                    except Exception:
                        return None
                return None

            fx = _get_intr_value("fx")
            fy = _get_intr_value("fy")
            ppx = _get_intr_value("ppx")
            ppy = _get_intr_value("ppy")
            coeffs = _get_intr_value("coeffs")
            if fx is None or fy is None or ppx is None or ppy is None:
                continue
            dist_coeffs = (
                np.array(coeffs[:5], dtype=np.float32)
                if coeffs is not None
                else np.zeros(5, dtype=np.float32)
            )
            K = np.array(
                [
                    [float(fx), 0.0, float(ppx)],
                    [0.0, float(fy), float(ppy)],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            return K, dist_coeffs

        fovy_deg = getattr(self._camera, "color_fovy", None)
        if fovy_deg is not None:
            fovy_rad = np.deg2rad(float(fovy_deg))
            fy = (height / 2.0) / np.tan(max(1e-6, fovy_rad / 2.0))
            fx = fy * (width / max(height, 1))
            cx = width / 2.0
            cy = height / 2.0
            K = np.array(
                [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            return K, np.zeros(5, dtype=np.float32)

        # Fallback to a generic pinhole approximation.
        fx = fy = max(width, height)
        cx = width / 2.0
        cy = height / 2.0
        K = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return K, np.zeros(5, dtype=np.float32)


def gripper_q_robomanip_to_maniskill(q_robomanip):
    """Convert RoboManip gripper position scalar to ManiSkill scale."""

    return (q_robomanip - 840.0) / (-1000.0)


def gripper_qvel_robomanip_to_maniskill(qvel_robomanip):
    """Convert RoboManip gripper velocity scalar to ManiSkill scale."""

    return qvel_robomanip / (-1000.0)


def gripper_q_maniskill_to_robomanip(q_maniskill):
    """Convert ManiSkill gripper position scalar to RoboManip scale."""

    return q_maniskill * (-1000.0) + 840.0


def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ManiSkillPpoAgent(nn.Module):
    """Reproduction of ManiSkill PPO agent architecture for rollout."""

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        self.critic = nn.Sequential(
            _layer_init(nn.Linear(self.obs_dim, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            _layer_init(nn.Linear(self.obs_dim, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, self.action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, self.action_dim) * -0.5)

    def get_value(self, obs):
        return self.critic(obs)

    def get_action(self, obs, deterministic=False):
        action_mean = self.actor_mean(obs)
        if deterministic:
            return action_mean

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()


_NORMALIZED_ACTION_LOW = torch.tensor(-1.0, dtype=torch.float32)
_NORMALIZED_ACTION_HIGH = torch.tensor(1.0, dtype=torch.float32)

_DELTA_PHYSICAL_LOW = torch.tensor(
    [
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
    ],
    dtype=torch.float32,
)

_DELTA_PHYSICAL_HIGH = torch.tensor(
    [
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
    ],
    dtype=torch.float32,
)

_JOINT_POSITION_LOW = torch.tensor(
    [
        -6.2831853,
        -2.059,
        -6.2831853,
        -0.19198,
        -6.2831853,
        -1.69297,
        -6.2831853,
        0.05,
    ],
    dtype=torch.float32,
)

_JOINT_POSITION_HIGH = torch.tensor(
    [
        6.2831853,
        2.0944,
        6.2831853,
        3.927,
        6.2831853,
        3.1415927,
        6.2831853,
        0.84,
    ],
    dtype=torch.float32,
)


class RolloutPpoCus(RolloutBase):
    def run(self):
        try:
            return super().run()
        finally:
            worker = getattr(self, "_marker_worker", None)
            if worker is not None:
                worker.stop()
                self._marker_worker = None

    #RolloutMlpにはない、引数関係などの定義(重要度の低い関数)
    def set_additional_args(self, parser):
        super().set_additional_args(parser)

        parser.add_argument(
            "--ppo-deterministic",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use deterministic ManiSkill PPO actions (default: True).",
        )
        parser.add_argument(
            "--ppo-use-cuda",
            action=argparse.BooleanOptionalAction,
            default=torch.cuda.is_available(),
            help="Enable CUDA for ManiSkill PPO if available (default: enabled when CUDA exists).",
        )
        parser.add_argument(
            "--ppo-log-tsv",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Log observations and actions to TSV each step (default: False).",
        )
        parser.add_argument(
            "--ppo-enable-vision",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Capture camera and tactile images during rollout (default: False).",
        )
        parser.add_argument(
            "--ppo-profile",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Measure per-step timings for debugging (default: False).",
        )
        parser.add_argument(
            "--ppo-marker-enable",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Enable background front-camera marker worker (default: False).",
        )
        parser.add_argument(
            "--ppo-marker-roi",
            type=int,
            nargs=4,
            metavar=("X", "Y", "W", "H"),
            default=None,
            help="Optional ROI (pixels) for front camera marker processing.",
        )

    def setup_model_meta_info(self):
        checkpoint_dir = os.path.split(self.args.checkpoint)[0]
        model_meta_info_path = os.path.join(checkpoint_dir, "model_meta_info.pkl")

        if os.path.isfile(model_meta_info_path):
            super().setup_model_meta_info()
            return

        self.model_meta_info = self._build_default_model_meta_info()
        print(
            f"[{self.__class__.__name__}] model_meta_info.pkl not found. Using default ManiSkill-compatible meta info."
        )

        self.state_keys = self.model_meta_info["state"]["keys"]
        self.action_keys = self.model_meta_info["action"]["keys"]
        self.camera_names = self.model_meta_info["image"]["camera_names"]
        self.state_dim = len(self.model_meta_info["state"]["example"])
        self.action_dim = len(self.model_meta_info["action"]["example"])

        if self.args.skip is None:
            self.args.skip = self.model_meta_info["data"]["skip"]
        if self.args.skip_draw is None:
            self.args.skip_draw = self.args.skip

    def _build_default_model_meta_info(self):
        default_state_keys = [
            DataKey.MEASURED_JOINT_POS,
            DataKey.MEASURED_JOINT_VEL,
            DataKey.MEASURED_EEF_WRENCH,
        ]
        default_action_keys = [DataKey.COMMAND_JOINT_POS]

        state_dim = sum(DataKey.get_dim(key, self.env) for key in default_state_keys)
        action_dim = sum(DataKey.get_dim(key, self.env) for key in default_action_keys)

        state_template = np.zeros(state_dim, dtype=np.float32)
        action_template = np.zeros(action_dim, dtype=np.float32)

        if hasattr(self.env, "camera_names"):
            camera_names = list(self.env.camera_names)
        else:
            camera_names = []

        return {
            "state": {
                "keys": default_state_keys,
                "example": state_template.copy(),
                "mean": state_template.copy(),
                "std": np.ones_like(state_template),
            },
            "action": {
                "keys": default_action_keys,
                "example": action_template.copy(),
                "mean": action_template.copy(),
                "std": np.ones_like(action_template),
            },
            "image": {"camera_names": camera_names},
            "data": {"skip": 1, "n_obs_steps": 1, "n_action_steps": 1},
            "policy": {},
        }

    def setup_policy(self):
        if not self.args.ppo_enable_vision:
            self._disable_env_vision()

        # Print policy information
        self.print_policy_info()
        print(
            f"  - obs steps: {self.model_meta_info['data']['n_obs_steps']}, action steps: {self.model_meta_info['data']['n_action_steps']}"
        )

        state_dict = torch.load(self.args.checkpoint, map_location="cpu")

        if "actor_mean.0.weight" not in state_dict or "actor_logstd" not in state_dict:
            raise KeyError(
                f"[{self.__class__.__name__}] ManiSkill PPO checkpoint does not contain expected keys."
            )

        obs_dim = state_dict["actor_mean.0.weight"].shape[1]
        action_dim_from_ckpt = int(state_dict["actor_logstd"].shape[-1])
        if action_dim_from_ckpt != self.action_dim:
            raise ValueError(
                f"[{self.__class__.__name__}] action dim mismatch: meta={self.action_dim}, checkpoint={action_dim_from_ckpt}"
            )

        self.policy = ManiSkillPpoAgent(obs_dim, self.action_dim)
        self.policy.load_state_dict(state_dict)

        use_cuda = self.args.ppo_use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.policy.to(self.device)
        self.policy.eval()

        self._normalized_action_low = torch.full(
            (self.action_dim,), float(_NORMALIZED_ACTION_LOW.item()), device=self.device
        )
        self._normalized_action_high = torch.full(
            (self.action_dim,), float(_NORMALIZED_ACTION_HIGH.item()), device=self.device
        )

        if self.action_dim != len(_DELTA_PHYSICAL_LOW):
            raise ValueError(
                f"[{self.__class__.__name__}] action dim mismatch for delta bounds: "
                f"meta={self.action_dim}, expected={len(_DELTA_PHYSICAL_LOW)}"
            )

        self._delta_physical_low = _DELTA_PHYSICAL_LOW.to(self.device)
        self._delta_physical_high = _DELTA_PHYSICAL_HIGH.to(self.device)
        self._joint_position_low = _JOINT_POSITION_LOW.to(self.device)
        self._joint_position_high = _JOINT_POSITION_HIGH.to(self.device)

        print(
            f"[{self.__class__.__name__}] Load ManiSkill PPO checkpoint on {self.device}"
        )

        self._log_path = None
        if self.args.ppo_log_tsv:
            checkpoint_dir = os.path.dirname(os.path.abspath(self.args.checkpoint))
            default_name = f"{self.__class__.__name__.lower()}_debug_log.tsv"
            self._log_path = os.path.join(checkpoint_dir, default_name)
            os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
            with open(self._log_path, "w", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["step_idx", "obs", "direct_joint_command"])
            print(
                f"[{self.__class__.__name__}] Logging observations and actions to {self._log_path}"
            )

    def setup_variables(self):
        super().setup_variables()

        self._marker_worker = None
        if getattr(self.args, "ppo_marker_enable", False):
            if _GLOBAL_T_BASE_TO_CAMERA is None:
                print(
                    f"[{self.__class__.__name__}] T_base→camera calibration not loaded; marker worker disabled.",
                    flush=True,
                )
            else:
                env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
                front_camera = getattr(env, "cameras", {}).get("front")
                if front_camera is None:
                    print(
                        f"[{self.__class__.__name__}] front camera not found; marker worker disabled.",
                        flush=True,
                    )
                else:
                    roi = (
                        tuple(int(v) for v in self.args.ppo_marker_roi)
                        if self.args.ppo_marker_roi
                        else None
                    )
                    self._marker_worker = FrontCameraFrameWorker(
                        front_camera,
                        base_to_camera=_GLOBAL_T_BASE_TO_CAMERA,
                        frame_size=(640, 480),
                        roi=roi,
                        tag_size_m=DEFAULT_TAG_SIZE_M,
                    )
                    self._marker_worker.start()
                    print(
                        f"[{self.__class__.__name__}] Started front camera marker worker with ROI={roi}.",
                        flush=True,
                    )

        self._profile_enabled = bool(getattr(self.args, "ppo_profile", False))
        if self._profile_enabled:
            self._profile_data = defaultdict(list)
            self._wrap_profile_hooks()

    def _disable_env_vision(self):
        env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env

        self._vision_backup = {
            "cameras": getattr(env, "cameras", None),
            "rgb_tactiles": getattr(env, "rgb_tactiles", None),
        }

        if hasattr(env, "cameras"):
            env.cameras = {}
        if hasattr(env, "rgb_tactiles"):
            env.rgb_tactiles = {}

        self.camera_names = []
        if "image" in self.model_meta_info:
            self.model_meta_info["image"]["camera_names"] = []

    def _wrap_profile_hooks(self):
        env = self.env

        original_step = env.step
        rollout_self = self

        def profiled_step(env_self, action):
            start = time.perf_counter()
            result = original_step(action)
            rollout_self._profile_data["env_step"].append(time.perf_counter() - start)
            return result

        env.step = types.MethodType(profiled_step, env)

        original_record_data = self.record_data

        def profiled_record_data():
            start = time.perf_counter()
            result = original_record_data()
            rollout_self._profile_data["record_data"].append(time.perf_counter() - start)
            return result

        self.record_data = profiled_record_data

    def setup_plot(self):
        num_cols = max(len(self.camera_names), 1)
        fig_ax = plt.subplots(
            2,
            num_cols,
            figsize=(13.5, 6.0),
            dpi=60,
            squeeze=False,
            constrained_layout=True,
        )
        super().setup_plot(fig_ax)

    def reset_variables(self):
        super().reset_variables()

        self.state_buf = None
        self.images_buf = None
        self.policy_action_buf = None

    def get_latest_marker_frame(self):
        if getattr(self, "_marker_worker", None) is None:
            return None, None
        return self._marker_worker.get_latest_frame()

    def get_latest_marker_transforms(self, poll: bool = False):
        worker = getattr(self, "_marker_worker", None)
        if worker is None:
            return None, None
        if poll:
            transforms, timestamp = worker.poll_transforms()
            if transforms is not None:
                return transforms, timestamp
        return worker.get_latest_transforms()

    def get_state(self):
        # Get latest value
        if len(self.state_keys) == 0:
            state = np.zeros(0, dtype=np.float32)
        else:
            state = np.concatenate(
                [
                    self.motion_manager.get_data(state_key, self.obs)
                    for state_key in self.state_keys
                ]
            )

        qpos = self.motion_manager.get_data(DataKey.MEASURED_JOINT_POS, self.obs)
        qvel = self.motion_manager.get_data(DataKey.MEASURED_JOINT_VEL, self.obs)
        target_qpos = np.array(
            [
                0.0,
                -0.477,
                0.0,
                0.8571976,
                0.0,
                1.2771976,
                -1.5707964,
                40,
            ],
         
            dtype=np.float32,
        )
        qpos_ms = qpos.astype(np.float32).copy()
        qpos_ms[-1] = gripper_q_robomanip_to_maniskill(qpos_ms[-1])
        qvel_ms = qvel.astype(np.float32).copy()
        if qvel_ms.size > 0:
            qvel_ms[-1] = gripper_qvel_robomanip_to_maniskill(qvel_ms[-1])
        target_qpos_ms = target_qpos.copy()
        target_qpos_ms[-1] = gripper_q_robomanip_to_maniskill(target_qpos_ms[-1])

        self.state_for_ppo = np.concatenate([qpos_ms, qvel_ms, target_qpos_ms]).astype(
            np.float32
        )

        norm_state = normalize_data(state, self.model_meta_info["state"])

        state = torch.tensor(norm_state, dtype=torch.float32)

        # Store and return
        if self.state_buf is None:
            self.state_buf = [
                state for _ in range(self.model_meta_info["data"]["n_obs_steps"])
            ]
        else:
            self.state_buf.pop(0)
            self.state_buf.append(state)

        state = torch.stack(self.state_buf, dim=0)[torch.newaxis].to(self.device)

        return state

    def get_images(self):
        # Get latest value
        if len(self.camera_names) == 0:
            return None

        images = []
        for camera_name in self.camera_names:
            image = self.info["rgb_images"][camera_name]

            image = np.moveaxis(image, -1, -3)
            image = torch.tensor(image.copy(), dtype=torch.uint8)
            image = self.image_transforms(image)

            images.append(image)

        # Store and return
        if self.images_buf is None:
            self.images_buf = [
                [image for _ in range(self.model_meta_info["data"]["n_obs_steps"])]
                for image in images
            ]
        else:
            for single_images_buf, image in zip(self.images_buf, images):
                single_images_buf.pop(0)
                single_images_buf.append(image)

        images = torch.stack(
            [
                torch.stack(single_images_buf, dim=0)[torch.newaxis].to(self.device)
                for single_images_buf in self.images_buf
            ]
        )

        return images

    def infer_policy(self):
        # Infer
        if self.policy_action_buf is None or len(self.policy_action_buf) == 0:
            profile_enabled = getattr(self, "_profile_enabled", False)
            if profile_enabled:
                timer = time.perf_counter
                total_start = timer()
                state_start = timer()

            self.get_state()  # update buffers and logs

            if profile_enabled:
                self._profile_data["state_fetch"].append(timer() - state_start)

            obs_tensor = torch.tensor(
                self.state_for_ppo, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            if profile_enabled:
                policy_start = timer()

            with torch.no_grad():
                raw_action = self.policy.get_action(
                    obs_tensor, deterministic=self.args.ppo_deterministic
                )

            if profile_enabled:
                self._profile_data["policy_forward"].append(timer() - policy_start)

            raw_action = raw_action.squeeze(0)
            clipped_action = torch.clamp(
                raw_action, self._normalized_action_low, self._normalized_action_high
            )

            normalized_span = self._normalized_action_high - self._normalized_action_low
            delta_scale = (clipped_action - self._normalized_action_low) / normalized_span
            denormalized_delta = self._delta_physical_low + delta_scale * (
                self._delta_physical_high - self._delta_physical_low
            )

            current_joint_pos = obs_tensor[..., : self.action_dim].squeeze(0)
            direct_joint_command = current_joint_pos + denormalized_delta
            direct_joint_command = torch.max(
                torch.min(direct_joint_command, self._joint_position_high),
                self._joint_position_low,
            )

            if direct_joint_command.numel() > 0:
                direct_joint_command = direct_joint_command.clone()
                direct_joint_command[-1] = gripper_q_maniskill_to_robomanip(
                    direct_joint_command[-1]
                )

            physical_np = direct_joint_command.detach().cpu().numpy().astype(np.float64)

            if hasattr(self, "_log_path") and self._log_path:
                obs_list = (
                    obs_tensor.squeeze(0).detach().cpu().numpy().astype(np.float64).tolist()
                )
                direct_list = physical_np.tolist()
                with open(self._log_path, "a", newline="") as f:
                    writer = csv.writer(f, delimiter="\t")
                    writer.writerow(
                        [
                            int(getattr(self, "rollout_time_idx", 0)),
                            json.dumps(obs_list),
                            json.dumps(direct_list),
                        ]
                    )

            self.policy_action_buf = [physical_np]

            if profile_enabled:
                self._profile_data["infer_total"].append(timer() - total_start)

        # Store action
        self.policy_action = denormalize_data(
            self.policy_action_buf.pop(0), self.model_meta_info["action"]
        )
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )

    def set_command_data(self, action_keys=None):
        if getattr(self, "_profile_enabled", False):
            start = time.perf_counter()
            super().set_command_data(action_keys)
            self._profile_data["set_command_data"].append(time.perf_counter() - start)
        else:
            super().set_command_data(action_keys)

    def draw_plot(self):
        # Clear plot
        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        # Plot images
        self.plot_images(self.ax[0, 0 : len(self.camera_names)])

        # Plot action
        self.plot_action(self.ax[1, 0])

        # Finalize plot
        self.canvas.draw()
        cv2.imshow(
            self.policy_name,
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )

    def print_statistics(self):
        super().print_statistics()
        if getattr(self, "_profile_enabled", False) and self._profile_data:
            print(f"[{self.__class__.__name__}] Profiling summary")
            for key, samples in self._profile_data.items():
                if not samples:
                    continue
                samples_arr = np.array(samples)
                print(
                    f"  - {key} [s] | mean: {samples_arr.mean():.2e}, "
                    f"std: {samples_arr.std():.2e}, min: {samples_arr.min():.2e}, max: {samples_arr.max():.2e}"
                )
