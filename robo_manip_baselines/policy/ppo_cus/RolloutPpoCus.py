import argparse
import csv
import importlib
import inspect
import json
import os
import sys
import time
import types
from collections import defaultdict
from pathlib import Path
import queue
import threading
from typing import Any, Dict, Optional, Tuple, List

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

DEFAULT_TARGET_JOINT_POS = np.array(
    [
        0.0,
        -0.477,
        0.0,
        0.8571976,
        0.0,
        1.2771976,
        -1.5707964,
        40.0,
    ],
    dtype=np.float32,
)



class FrontCameraDetectionWorker:
    """Process front-camera frames on a background thread to estimate AprilTag poses."""

    def __init__(
        self,
        base_to_camera: Optional[np.ndarray],
        intrinsic_info: Optional[Dict[str, Any]] = None,
        tag_size_m: float = DEFAULT_TAG_SIZE_M,
    ):
        self._base_to_camera = None if base_to_camera is None else base_to_camera.astype(np.float64)
        self._intrinsic_info: Dict[str, Any] = intrinsic_info or {}
        self._tag_size_m = float(tag_size_m)
        self._frame_queue: "queue.Queue[Optional[Tuple[np.ndarray, float]]]" = queue.Queue(maxsize=1)
        self._result_queue: "queue.Queue[Tuple[float, Dict[int, np.ndarray]]]" = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._processing_thread: Optional[threading.Thread] = None
        self._latest_lock = threading.Lock()
        self._latest_gray: Optional[np.ndarray] = None
        self._latest_timestamp: Optional[float] = None
        self._latest_transforms: Dict[int, np.ndarray] = {}
        self._latest_transform_times: Dict[int, float] = {}
        self._latest_transforms_timestamp: Optional[float] = None
        self._detector = None
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs: Optional[np.ndarray] = None
        self._detection_available = (
            build_detector is not None
            and build_homogeneous_transform is not None
            and solve_tag_poses is not None
            and self._base_to_camera is not None
        )
        self._frame_counter = 0
        self._last_detection_count = None

    def start(self):
        if self._processing_thread and self._processing_thread.is_alive():
            return

        if not self._detection_available:
            print(
                "[FrontCameraDetectionWorker] Marker detection disabled (missing dependencies or calibration).",
                flush=True,
            )

        detector = self._build_detector()
        if detector is None:
            self._detection_available = False
            print(
                "[FrontCameraDetectionWorker] Detector initialization failed; detection disabled.",
                flush=True,
            )

        self._stop_event.clear()
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            name="front_camera_detection",
            daemon=True,
        )
        self._processing_thread.start()

    def stop(self):
        self._stop_event.set()
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

        if self._processing_thread:
            self._processing_thread.join(timeout=1.0)
            self._processing_thread = None

        while True:
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break

    def submit_frame(self, rgb_image: np.ndarray) -> None:
        if rgb_image is None or self._stop_event.is_set():
            return

        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        timestamp = time.time()

        with self._latest_lock:
            self._latest_gray = gray_image
            self._latest_timestamp = timestamp

        payload = (gray_image, timestamp)
        try:
            self._frame_queue.put_nowait(payload)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(payload)
            except queue.Full:
                pass
        self._frame_counter += 1

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        with self._latest_lock:
            if self._latest_gray is None:
                return None, None
            return self._latest_gray.copy(), self._latest_timestamp

    def get_latest_transforms(self) -> Tuple[Dict[int, np.ndarray], Optional[float]]:
        with self._latest_lock:
            return (
                {tag_id: matrix.copy() for tag_id, matrix in self._latest_transforms.items()},
                self._latest_transforms_timestamp,
            )

    def get_latest_transform_times(self) -> Dict[int, float]:
        with self._latest_lock:
            return dict(self._latest_transform_times)

    def poll_transforms(self) -> Tuple[Optional[Dict[int, np.ndarray]], Optional[float]]:
        try:
            timestamp, transforms = self._result_queue.get_nowait()
        except queue.Empty:
            return None, None

        copied = {tag_id: matrix.copy() for tag_id, matrix in transforms.items()}
        with self._latest_lock:
            self._latest_transforms = {
                tag_id: matrix.copy() for tag_id, matrix in copied.items()
            }
            if timestamp is not None:
                self._latest_transforms_timestamp = timestamp
        return copied, timestamp

    def _processing_loop(self):
        while not self._stop_event.is_set():
            try:
                item = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                break

            gray_image, timestamp = item
            transforms: Dict[int, np.ndarray] = {}

            if self._detection_available:
                detector = self._build_detector()
                camera_mats = self._ensure_camera_parameters(gray_image)
                if detector is not None and camera_mats is not None:
                    K, dist_coeffs = camera_mats
                    try:
                        poses = solve_tag_poses(
                            detector,
                            gray_image,
                            K,
                            dist_coeffs,
                            self._tag_size_m,
                        )
                    except Exception as exc:  # pragma: no cover - detector failure
                        poses = []
                    if self._last_detection_count != len(poses):
                        print(
                            f"[FrontCameraDetectionWorker] Detected {len(poses)} tags in current frame.",
                            flush=True,
                        )
                        self._last_detection_count = len(poses)

                    for pose in poses:
                        if int(pose.tag_id) == 3:
                            continue
                        try:
                            T_cam_to_tag = build_homogeneous_transform(pose.rvec, pose.tvec)
                        except Exception as exc:  # pragma: no cover
                            print(
                                f"[FrontCameraDetectionWorker] Failed to build transform for tag {pose.tag_id}: {exc}",
                                flush=True,
                            )
                            continue
                        T_base_to_tag = self._base_to_camera @ T_cam_to_tag
                        transforms[int(pose.tag_id)] = T_base_to_tag
                        matrix_str = np.array2string(
                            T_base_to_tag,
                            formatter={"float_kind": lambda x: f"{x: .4f}"},
                        )
                        print(
                            f"[FrontCameraDetectionWorker] tag {pose.tag_id} transform:\n{matrix_str}",
                            flush=True,
                        )

            payload = None
            with self._latest_lock:
                if transforms:
                    for tag_id, matrix in transforms.items():
                        self._latest_transforms[tag_id] = matrix.copy()
                        self._latest_transform_times[tag_id] = timestamp
                    self._latest_transforms_timestamp = timestamp
                    combined = {
                        tag_id: matrix.copy()
                        for tag_id, matrix in self._latest_transforms.items()
                    }
                    payload = (timestamp, combined)

            if payload is not None:
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

    def _build_detector(self):
        if self._detector is not None:
            return self._detector
        if build_detector is None:
            return None

        detector_kwargs = {
            "nthreads": DEFAULT_DETECTOR_THREADS,
            "quad_decimate": DEFAULT_DETECTOR_DECIMATE,
            "quad_sigma": DEFAULT_DETECTOR_SIGMA,
            "refine_edges": True,
            "decode_sharpening": DEFAULT_DETECTOR_SHARPENING,
        }
        try:
            sig = inspect.signature(build_detector)
            accepted = {
                key: value for key, value in detector_kwargs.items() if key in sig.parameters
            }
        except (TypeError, ValueError):  # pragma: no cover - signature introspection failure
            accepted = detector_kwargs

        try:
            self._detector = build_detector("tag36h11", **accepted)
        except TypeError:
            try:
                self._detector = build_detector("tag36h11")
            except Exception as exc:
                self._detector = None
        except Exception as exc:  # pragma: no cover - detector creation failure
            self._detector = None
        else:
            if self._detector is not None:
                print(
                    "[FrontCameraDetectionWorker] AprilTag detector initialized (tag36h11).",
                    flush=True,
                )
        return self._detector

    def _ensure_camera_parameters(
        self, frame: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._camera_matrix is not None and self._dist_coeffs is not None:
            return self._camera_matrix, self._dist_coeffs

        height, width = frame.shape[:2]
        info = self._intrinsic_info
        fx = info.get("fx")
        fy = info.get("fy")
        ppx = info.get("ppx")
        ppy = info.get("ppy")
        coeffs = info.get("coeffs")

        if fx is not None and fy is not None and ppx is not None and ppy is not None:
            K = np.array(
                [
                    [float(fx), 0.0, float(ppx)],
                    [0.0, float(fy), float(ppy)],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            dist_coeffs = (
                np.array(coeffs[:5], dtype=np.float32)
                if isinstance(coeffs, (list, tuple, np.ndarray))
                else np.zeros(5, dtype=np.float32)
            )
            self._camera_matrix = K
            self._dist_coeffs = dist_coeffs
            return K, dist_coeffs

        fovy_deg = info.get("color_fovy")
        frame_w = int(info.get("frame_width", width))
        frame_h = int(info.get("frame_height", height))
        if fovy_deg is not None:
            fovy_rad = np.deg2rad(float(fovy_deg))
            fy = (frame_h / 2.0) / np.tan(max(1e-6, fovy_rad / 2.0))
            fy = float(fy)
            fx = fy * (frame_w / max(frame_h, 1))
            cx = frame_w / 2.0
            cy = frame_h / 2.0
            K = np.array(
                [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            self._camera_matrix = K
            self._dist_coeffs = np.zeros(5, dtype=np.float32)
            return self._camera_matrix, self._dist_coeffs

        fx = fy = max(frame_w, frame_h)
        cx = frame_w / 2.0
        cy = frame_h / 2.0
        self._camera_matrix = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self._dist_coeffs = np.zeros(5, dtype=np.float32)
        return self._camera_matrix, self._dist_coeffs
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
            "--ppo-profile",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Measure per-step timings for debugging (default: False).",
        )

    def setup_model_meta_info(self):
        checkpoint_dir = os.path.split(self.args.checkpoint)[0]
        model_meta_info_path = os.path.join(checkpoint_dir, "model_meta_info.pkl")

        if not os.path.isfile(model_meta_info_path):
            raise FileNotFoundError(
                f"[{self.__class__.__name__}] Required model_meta_info.pkl was not found "
                f"next to the checkpoint: {model_meta_info_path}. "
                "Generate the file (e.g., with CreatePpoCusMetaInfo.py) and re-run."
            )

        super().setup_model_meta_info()
        self.extra_state_keys: list[str] = []
        self.extra_state_dims: Dict[str, int] = {}
        self.ppo_task_handler = None
        self.ppo_task_params: Dict[str, Any] = {}
        self.marker_definitions: List[Dict[str, Any]] = []
        self.required_marker_ids: List[int] = []
        self.marker_name_map: Dict[int, str] = {}
        self.marker_size_map: Dict[int, float] = {}
        self.default_target_joint_pos = DEFAULT_TARGET_JOINT_POS.copy()
        self.marker_camera_names: List[str] = ["front"]
        self._marker_camera_active: Optional[str] = None
        self._setup_ppo_task_from_meta()

    def _setup_ppo_task_from_meta(self) -> None:
        self.standard_state_keys = list(self.state_keys)

        ppo_task_cfg = self.model_meta_info.get("ppo_task")
        if not ppo_task_cfg:
            return

        extra_keys_cfg = ppo_task_cfg.get("extra_keys", [])
        if not isinstance(extra_keys_cfg, list):
            raise TypeError(
                f"[{self.__class__.__name__}] 'ppo_task.extra_keys' must be a list."
            )

        extra_state_keys: list[str] = []
        extra_state_dims: Dict[str, int] = {}
        for entry in extra_keys_cfg:
            if not isinstance(entry, dict):
                raise TypeError(
                    f"[{self.__class__.__name__}] 'ppo_task.extra_keys' entries must be objects."
                )
            name = entry.get("name")
            dim = entry.get("dim")
            if not name or not isinstance(name, str):
                raise ValueError(
                    f"[{self.__class__.__name__}] Invalid extra key name: {entry}"
                )
            if name in extra_state_keys:
                raise ValueError(
                    f"[{self.__class__.__name__}] Duplicate extra key detected: {name}"
                )
            if dim is None:
                raise ValueError(
                    f"[{self.__class__.__name__}] Missing dimension for extra key '{name}'."
                )
            dim_int = int(dim)
            if dim_int <= 0:
                raise ValueError(
                    f"[{self.__class__.__name__}] Dimension for extra key '{name}' must be positive."
                )
            extra_state_keys.append(name)
            extra_state_dims[name] = dim_int

        missing_in_state = [key for key in extra_state_keys if key not in self.state_keys]
        if missing_in_state:
            raise ValueError(
                f"[{self.__class__.__name__}] Extra state keys {missing_in_state} "
                "are not present in model_meta_info['state']['keys']."
            )

        module_path = ppo_task_cfg.get("module")
        if not module_path or not isinstance(module_path, str):
            raise ValueError(
                f"[{self.__class__.__name__}] 'ppo_task.module' must be a non-empty string."
            )

        try:
            task_module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"[{self.__class__.__name__}] Failed to import PPO task module '{module_path}'."
            ) from exc

        builder = getattr(task_module, "build_ppo_task", None)
        if builder is None:
            raise AttributeError(
                f"[{self.__class__.__name__}] Module '{module_path}' does not expose a 'build_ppo_task' function."
            )

        marker_cameras = ppo_task_cfg.get("marker_cameras", [])
        if isinstance(marker_cameras, list):
            self.marker_camera_names = [str(name) for name in marker_cameras]
        elif marker_cameras:
            raise TypeError(
                f"[{self.__class__.__name__}] 'ppo_task.marker_cameras' must be a list of camera names."
            )

        marker_defs_cfg = ppo_task_cfg.get("markers", [])
        if marker_defs_cfg:
            if not isinstance(marker_defs_cfg, list):
                raise TypeError(
                    f"[{self.__class__.__name__}] 'ppo_task.markers' must be a list."
                )
            definitions: List[Dict[str, Any]] = []
            seen_ids: set[int] = set()
            for entry in marker_defs_cfg:
                if not isinstance(entry, dict):
                    raise TypeError(
                        f"[{self.__class__.__name__}] Invalid marker entry: {entry}"
                    )
                if "id" not in entry:
                    raise ValueError(
                        f"[{self.__class__.__name__}] Marker entry requires 'id': {entry}"
                    )
                tag_id = int(entry["id"])
                if tag_id in seen_ids:
                    raise ValueError(
                        f"[{self.__class__.__name__}] Duplicate marker id detected: {tag_id}"
                    )
                seen_ids.add(tag_id)
                name = str(entry.get("name", f"marker_{tag_id}"))
                size_m = float(entry.get("size_m", DEFAULT_TAG_SIZE_M))
                definitions.append({"id": tag_id, "name": name, "size_m": size_m})
            self.marker_definitions = definitions
            self.required_marker_ids = [item["id"] for item in definitions]
            self.marker_name_map = {item["id"]: item["name"] for item in definitions}
            self.marker_size_map = {item["id"]: item["size_m"] for item in definitions}

        params = ppo_task_cfg.get("params") or {}
        if not isinstance(params, dict):
            raise TypeError(
                f"[{self.__class__.__name__}] 'ppo_task.params' must be a dictionary."
            )
        self.ppo_task_params = dict(params)

        self.extra_state_keys = extra_state_keys
        self.extra_state_dims = extra_state_dims
        self.standard_state_keys = [
            key for key in self.state_keys if key not in self.extra_state_keys
        ]

        self.ppo_task_handler = builder(self, params)
        if self.ppo_task_handler is None:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Task builder '{module_path}.build_ppo_task' returned None."
            )

        task_name = ppo_task_cfg.get("name") or module_path
        print(
            f"[{self.__class__.__name__}] Loaded PPO task '{task_name}'. "
            f"Extra state keys: {self.extra_state_keys}"
        )

    def setup_policy(self):
        # Always keep vision enabled for marker detection

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
        self._marker_camera_active = None
        self.marker_transform_cache: Dict[int, np.ndarray] = {}
        self.marker_detection_verified = False
        if _GLOBAL_T_BASE_TO_CAMERA is None:
            print(
                f"[{self.__class__.__name__}] T_base→camera calibration not loaded; marker worker disabled.",
                flush=True,
            )
        else:
            env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
            primary_cameras = getattr(env, "cameras", {}) or {}
            backup_cameras: Dict[str, Any] = {}
            vision_backup = getattr(self, "_vision_backup", None)
            if isinstance(vision_backup, dict):
                backup_candidate = vision_backup.get("cameras")
                if isinstance(backup_candidate, dict):
                    backup_cameras = backup_candidate

            available_camera_names = set(primary_cameras.keys()) | set(
                backup_cameras.keys()
            )

            for camera_name in self.marker_camera_names:
                camera_obj = primary_cameras.get(camera_name)
                if camera_obj is None:
                    camera_obj = backup_cameras.get(camera_name)
                if camera_obj is None:
                    continue

                intrinsic_info = self._extract_camera_intrinsic_info(camera_obj)
                self._marker_worker = FrontCameraDetectionWorker(
                    base_to_camera=_GLOBAL_T_BASE_TO_CAMERA,
                    intrinsic_info=intrinsic_info,
                    tag_size_m=DEFAULT_TAG_SIZE_M,
                )
                self._marker_worker.start()
                self._marker_camera_active = camera_name
                print(
                    f"[{self.__class__.__name__}] Started marker worker for camera '{camera_name}'.",
                    flush=True,
                )
                break

            if self._marker_worker is None:
                if self.marker_camera_names:
                    raise ValueError(
                        f"[{self.__class__.__name__}] None of the requested marker cameras "
                        f"{self.marker_camera_names} were found. "
                        f"Available cameras: {sorted(available_camera_names)}"
                    )
                else:
                    print(
                        f"[{self.__class__.__name__}] No marker cameras specified; marker worker disabled.",
                        flush=True,
                    )

        self._profile_enabled = bool(getattr(self.args, "ppo_profile", False))
        if self._profile_enabled:
            self._profile_data = defaultdict(list)
            self._wrap_profile_hooks()

    def _submit_marker_frame(self) -> bool:
        if getattr(self, "_marker_worker", None) is None:
            return False
        if not isinstance(getattr(self, "info", None), dict):
            return False
        rgb_images = self.info.get("rgb_images")
        if not isinstance(rgb_images, dict):
            return False
        candidate_names = (
            [self._marker_camera_active]
            if self._marker_camera_active
            else self.marker_camera_names
        )
        for camera_name in candidate_names:
            if camera_name is None:
                continue
            frame = rgb_images.get(camera_name)
            if frame is None:
                continue
            self._marker_worker.submit_frame(frame.copy())
            return True
        return False

    def _ensure_initial_marker_detection(self) -> None:
        if not self.required_marker_ids:
            return
        if self._marker_worker is None:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Marker worker not initialized, but marker IDs were specified."
            )

        timeout = float(self.ppo_task_params.get("initial_detection_timeout", 5.0))
        poll_interval = float(self.ppo_task_params.get("initial_detection_interval", 0.05))
        start_time = time.time()
        last_report_time = start_time

        missing_ids = set(self.required_marker_ids)

        while time.time() - start_time <= timeout:
            self._submit_marker_frame()
            transforms, _ = self.get_latest_marker_transforms(poll=True)
            if not transforms:
                transforms, _ = self.get_latest_marker_transforms()

            if transforms:
                for marker_id, matrix in transforms.items():
                    self.marker_transform_cache[marker_id] = matrix.copy()
                missing_ids = {
                    marker_id
                    for marker_id in self.required_marker_ids
                    if marker_id not in self.marker_transform_cache
                }
                if not missing_ids:
                    self.marker_detection_verified = True
                    print(
                        f"[{self.__class__.__name__}] Confirmed initial detection of markers: "
                        f"{sorted(self.required_marker_ids)}",
                        flush=True,
                    )
                    return

            current_time = time.time()
            if current_time - last_report_time >= 1.0:
                print(
                    f"[{self.__class__.__name__}] Waiting for marker detection. "
                    f"Missing IDs: {sorted(missing_ids)}",
                    flush=True,
                )
                last_report_time = current_time

            time.sleep(poll_interval)

        raise RuntimeError(
            f"[{self.__class__.__name__}] Failed to detect required markers within {timeout:.1f}s. "
            f"Missing IDs: {sorted(missing_ids)}"
        )

    def _extract_camera_intrinsic_info(self, camera) -> Optional[Dict[str, Any]]:
        if camera is None:
            return None

        info: Dict[str, Any] = {}
        candidate_attrs = (
            "color_intrinsics",
            "intrinsics",
            "color_intrinsic",
            "intrinsic",
        )
        for attr in candidate_attrs:
            intr = getattr(camera, attr, None)
            if intr is None:
                continue

            def _get_value(name: str):
                if hasattr(intr, name):
                    return getattr(intr, name)
                if isinstance(intr, dict):
                    return intr.get(name)
                if hasattr(intr, "__getitem__"):
                    try:
                        return intr[name]
                    except Exception:
                        return None
                return None

            fx = _get_value("fx")
            fy = _get_value("fy")
            ppx = _get_value("ppx")
            ppy = _get_value("ppy")
            coeffs = _get_value("coeffs")

            if fx is not None:
                info["fx"] = float(fx)
            if fy is not None:
                info["fy"] = float(fy)
            if ppx is not None:
                info["ppx"] = float(ppx)
            if ppy is not None:
                info["ppy"] = float(ppy)
            if coeffs is not None:
                info["coeffs"] = list(coeffs) if not isinstance(coeffs, list) else coeffs

            if info:
                break

        color_fovy = getattr(camera, "color_fovy", None)
        if color_fovy is not None:
            info["color_fovy"] = float(color_fovy)

        frame_width = getattr(camera, "color_width", None) or getattr(camera, "width", None)
        frame_height = getattr(camera, "color_height", None) or getattr(camera, "height", None)
        if frame_width is not None:
            info["frame_width"] = int(frame_width)
        if frame_height is not None:
            info["frame_height"] = int(frame_height)

        return info if info else None

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
        extra_state_arrays: Dict[str, np.ndarray] = {}
        if len(self.state_keys) == 0:
            state_vector = np.zeros(0, dtype=np.float32)
        else:
            extra_state_values: Dict[str, np.ndarray] = {}
            if self.ppo_task_handler is not None:
                extra_state_raw = self.ppo_task_handler.get_extra_state() or {}
                if not isinstance(extra_state_raw, dict):
                    raise TypeError(
                        f"[{self.__class__.__name__}] Task handler must return a dict of extra states."
                    )
                extra_state_values = {
                    key: np.asarray(value, dtype=np.float32).reshape(-1)
                    for key, value in extra_state_raw.items()
                }

            # Validate extra state contributions
            for key in self.extra_state_keys:
                if key not in extra_state_values:
                    raise KeyError(
                        f"[{self.__class__.__name__}] Extra state '{key}' missing from PPO task handler output."
                    )
                arr = extra_state_values[key]
                expected_dim = self.extra_state_dims.get(key)
                if expected_dim is not None and arr.size != expected_dim:
                    raise ValueError(
                        f"[{self.__class__.__name__}] Extra state '{key}' has dimension {arr.size}, "
                        f"expected {expected_dim}."
                    )
                extra_state_arrays[key] = arr

            components = []
            for state_key in self.state_keys:
                if state_key in extra_state_arrays:
                    components.append(extra_state_arrays[state_key])
                else:
                    components.append(
                        self.motion_manager.get_data(state_key, self.obs)
                    )

            state_vector = (
                np.concatenate(components).astype(np.float32)
                if components
                else np.zeros(0, dtype=np.float32)
            )
            expected_state_dim = len(self.model_meta_info["state"]["example"])
            if state_vector.size != expected_state_dim:
                raise ValueError(
                    f"[{self.__class__.__name__}] State dimension mismatch. "
                    f"Constructed {state_vector.size} elements, "
                    f"but model_meta_info expects {expected_state_dim}."
                )

        marker_transforms, marker_timestamp = self.get_latest_marker_transforms(poll=True)
        if marker_transforms:
            for marker_id, matrix in marker_transforms.items():
                self.marker_transform_cache[marker_id] = matrix.copy()

        if self._marker_worker is not None:
            marker_times = self._marker_worker.get_latest_transform_times()
        else:
            marker_times = {}

        if self.marker_transform_cache:
            ts_str = (
                f"{marker_timestamp:.3f}"
                if marker_timestamp is not None
                else "unknown"
            )
            print(
                f"[{self.__class__.__name__}] Marker transforms cache (t={ts_str}):",
                flush=True,
            )
            for marker_id in sorted(self.marker_transform_cache.keys()):
                matrix = self.marker_transform_cache[marker_id]
                marker_name = self.marker_name_map.get(marker_id, f"id{marker_id}")
                matrix_str = np.array2string(
                    matrix,
                    formatter={"float_kind": lambda x: f"{x: .4f}"},
                )
                last_seen = marker_times.get(marker_id)
                last_seen_str = (
                    f"{last_seen:.3f}s" if last_seen is not None else "unknown"
                )
                print(
                    f"  [{marker_name}] id={marker_id}, last_seen={last_seen_str}\n{matrix_str}",
                    flush=True,
                )
        if self.required_marker_ids:
            missing_now = [
                marker_id
                for marker_id in self.required_marker_ids
                if marker_id not in self.marker_transform_cache
            ]
            if missing_now:
                print(
                    f"[{self.__class__.__name__}] Warning: Missing cached transforms for marker IDs {missing_now}.",
                    flush=True,
                )

        qpos = self.motion_manager.get_data(DataKey.MEASURED_JOINT_POS, self.obs)
        qvel = self.motion_manager.get_data(DataKey.MEASURED_JOINT_VEL, self.obs)

        target_qpos = extra_state_arrays.get(
            "target_joint_pos", self.default_target_joint_pos
        )
        target_qpos = target_qpos.astype(np.float32).copy()

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

        norm_state = normalize_data(state_vector, self.model_meta_info["state"])

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
        self._submit_marker_frame()

        if len(self.camera_names) == 0:
            return None

        images = []
        for camera_name in self.camera_names:
            rgb_image = self.info["rgb_images"][camera_name]
            if (
                self._marker_worker is not None
                and camera_name == "front"
            ):
                self._marker_worker.submit_frame(rgb_image.copy())

            image = np.moveaxis(rgb_image, -1, -3)
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

            self._submit_marker_frame()
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

    def reset(self):
        super().reset()
        if self.marker_transform_cache is not None:
            self.marker_transform_cache.clear()
        self.marker_detection_verified = False

        if self._marker_worker is not None:
            # Submit the first frame captured during reset, if available.
            self._submit_marker_frame()
            try:
                self._ensure_initial_marker_detection()
            except RuntimeError as exc:
                print(f"[{self.__class__.__name__}] {exc}", flush=True)
                raise

        if self.ppo_task_handler and hasattr(self.ppo_task_handler, "on_reset"):
            self.ppo_task_handler.on_reset()

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
