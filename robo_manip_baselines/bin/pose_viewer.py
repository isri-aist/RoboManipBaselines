"""AprilTag pose viewer that reports marker coordinates in the camera frame.

Run with a RealSense depth camera for production, or fall back to a webcam for
quick tests. The script highlights each detected marker, renders axes, and
shows XYZ position (in meters) together with roll/pitch/yaw derived from the
pose estimate.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from pupil_apriltags import Detector

try:  # Optional dependency when running with a webcam only.
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - handled dynamically in main()
    rs = None  # type: ignore[assignment]

@dataclass
class PoseEstimate:
    tag_id: int
    family: str
    position: np.ndarray  # (3,) meters in camera coordinates
    rpy_deg: np.ndarray   # (3,) roll, pitch, yaw in degrees
    distance_m: float
    corners: np.ndarray   # (4, 2) image coordinates
    center: Tuple[int, int]
    rvec: np.ndarray
    tvec: np.ndarray
    tag_size_m: float


BASE_TAG_ID = 3
BASE_TAG_SIZE_M = 0.09265  # 9.265 cm
TEST_TAG_ID = 2

T_BASE_TAG_TO_BASE_DEFAULT = np.array([
    [0.0, 0.0, 1.0, -0.066235],
    [0.0, 1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, 0.049494],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)
T_BASE_TO_BASE_TAG_DEFAULT = np.linalg.inv(T_BASE_TAG_TO_BASE_DEFAULT)

T_BASE_TAG_TO_BASE = T_BASE_TAG_TO_BASE_DEFAULT.copy()
T_BASE_TO_BASE_TAG = T_BASE_TO_BASE_TAG_DEFAULT.copy()


def build_detector(family: str) -> Detector:
    return Detector(
        families=family,
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=1.0,
        refine_edges=True,
        decode_sharpening=0.1,
    )


def rotmat_to_rpy_deg(R: np.ndarray) -> np.ndarray:
    sy = float(np.clip(-R[2, 0], -1.0, 1.0))
    pitch = np.arcsin(sy)
    if abs(sy) < 0.999999:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def put_text_safe(img: np.ndarray, text: str, x: int, y: int,
                  scale: float = 0.55, thick: int = 2) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = int(np.clip(x, 0, img.shape[1] - tw - 2))
    y = int(np.clip(y, th + 2, img.shape[0] - 2))
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_axes(img: np.ndarray, K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
              axis_len: float = 0.02, thickness: int = 2) -> None:
    origin = np.float32([[0, 0, 0]])
    axes = np.float32([[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]])
    pts3d = np.vstack([origin, axes]).reshape(-1, 1, 3)
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, None)
    p0, pX, pY, pZ = [tuple(p.ravel().astype(int)) for p in proj]
    cv2.line(img, p0, pX, (0, 0, 255), thickness)
    cv2.line(img, p0, pY, (0, 255, 0), thickness)
    cv2.line(img, p0, pZ, (255, 0, 0), thickness)


def solve_tag_poses(detector: Detector, gray: np.ndarray,
                    K: np.ndarray, dist_coeffs: np.ndarray,
                    tag_size_m: float, base_tag_id: int = BASE_TAG_ID,
                    base_tag_size_m: float = BASE_TAG_SIZE_M,
                    test_tag_id: int = TEST_TAG_ID) -> List[PoseEstimate]:
    detections = detector.detect(gray, estimate_tag_pose=False)
    estimates: List[PoseEstimate] = []
    objp_default = make_object_points(tag_size_m)
    objp_reference = make_object_points(base_tag_size_m)

    for det in detections:
        tag_id = int(det.tag_id)
        is_reference_tag = tag_id == base_tag_id or tag_id == test_tag_id
        tag_size = base_tag_size_m if is_reference_tag else tag_size_m
        objp = objp_reference if is_reference_tag else objp_default
        imgp = det.corners.astype(np.float32).reshape(-1, 1, 2)
        ok, rvec, tvec = cv2.solvePnP(
            objp,
            imgp,
            K,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            continue

        R, _ = cv2.Rodrigues(rvec)
        position = tvec.reshape(3)
        estimates.append(
            PoseEstimate(
                tag_id=tag_id,
                family=det.tag_family.decode() if isinstance(det.tag_family, bytes) else str(det.tag_family),
                position=position.astype(float),
                rpy_deg=rotmat_to_rpy_deg(R),
                distance_m=float(np.linalg.norm(position)),
                corners=det.corners.astype(int),
                center=tuple(np.round(det.center).astype(int)),
                rvec=rvec,
                tvec=tvec,
                tag_size_m=float(tag_size),
            )
        )

    return estimates


def build_homogeneous_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def report_base_tag_transform(poses: Sequence[PoseEstimate], base_tag_id: int = BASE_TAG_ID,
                              log: bool = True) -> Optional[np.ndarray]:
    for pose in poses:
        if pose.tag_id == base_tag_id:
            T = build_homogeneous_transform(pose.rvec, pose.tvec)
            if log:
                matrix_str = np.array2string(
                    T,
                    formatter={
                        "float_kind": lambda x: f"{x: .4f}",
                    },
                )
                print(f"T_cam→base_tag (ID={base_tag_id}):\n{matrix_str}\n", flush=True)
            return T
    return None


def report_test_tag_transform(poses: Sequence[PoseEstimate],
                              T_cam_to_base_tag: np.ndarray,
                              T_base_to_base_tag: np.ndarray,
                              test_tag_id: int = TEST_TAG_ID,
                              log: bool = True) -> Optional[np.ndarray]:
    test_pose: Optional[PoseEstimate] = None
    for pose in poses:
        if pose.tag_id == test_tag_id:
            test_pose = pose
            break
    if test_pose is None:
        return None

    T_base_tag_to_camera = np.linalg.inv(T_cam_to_base_tag)
    T_cam_to_test = build_homogeneous_transform(test_pose.rvec, test_pose.tvec)
    T_base_to_test = T_base_to_base_tag @ T_base_tag_to_camera @ T_cam_to_test
    if log:
        matrix_str = np.array2string(
            T_base_to_test,
            formatter={
                "float_kind": lambda x: f"{x: .4f}",
            },
        )
        print(f"T_base→test_tag (ID={test_tag_id}):\n{matrix_str}\n", flush=True)
    return T_base_to_test


def annotate_frame(img: np.ndarray, poses: Sequence[PoseEstimate], K: np.ndarray,
                   fps: float, header: str,
                   T_base_to_test: Optional[np.ndarray] = None) -> None:
    panel_w = max(280, min(420, img.shape[1] // 3))
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, img.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0.0, dst=img)

    cv2.putText(img, header, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"FPS: {fps:.1f}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                (220, 220, 220), 2, cv2.LINE_AA)

    sorted_poses = sorted(poses, key=lambda p: p.distance_m)
    y = 80
    for pose in sorted_poses[:18]:
        x_pos, y_pos, z_pos = pose.position
        size_cm = pose.tag_size_m * 100.0
        tag_label = pose.family
        if pose.tag_id == BASE_TAG_ID:
            tag_label = f"{tag_label} base_tag"
        elif pose.tag_id == TEST_TAG_ID:
            tag_label = f"{tag_label} test_tag"
        line1 = f"ID {pose.tag_id:3d} ({tag_label})  d={pose.distance_m:.3f} m"
        line2 = f"XYZ=({x_pos:+.3f},{y_pos:+.3f},{z_pos:+.3f}) m"
        r, p_val, y_val = pose.rpy_deg
        line3 = f"RPY=({r:+.1f},{p_val:+.1f},{y_val:+.1f}) deg"
        line4 = f"pix=({pose.center[0]},{pose.center[1]}) size={size_cm:.1f} cm"
        cv2.putText(img, line1, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                    (255, 255, 255), 2, cv2.LINE_AA); y += 20
        cv2.putText(img, line2, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (230, 230, 230), 2, cv2.LINE_AA); y += 20
        cv2.putText(img, line3, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (210, 210, 210), 2, cv2.LINE_AA); y += 20
        cv2.putText(img, line4, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (190, 190, 190), 2, cv2.LINE_AA); y += 26

        cv2.polylines(img, [pose.corners], True, (0, 255, 0), 2)
        cv2.circle(img, pose.center, 3, (0, 0, 255), -1)
        draw_axes(img, K, pose.rvec, pose.tvec, axis_len=pose.tag_size_m * 0.75, thickness=2)

    if T_base_to_test is not None:
        y_info = max(y, 80)
        tx, ty, tz = T_base_to_test[:3, 3]
        rpy = rotmat_to_rpy_deg(T_base_to_test[:3, :3])
        overlay_lines = [
            "T_base→test_tag:",
            f"  XYZ=({tx:+.3f},{ty:+.3f},{tz:+.3f}) m",
            f"  RPY=({rpy[0]:+.1f},{rpy[1]:+.1f},{rpy[2]:+.1f}) deg",
        ]
        for line in overlay_lines:
            cv2.putText(img, line, (12, y_info), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)
            y_info += 22

    put_text_safe(img, "Camera frame: X right, Y down, Z forward", 14, img.shape[0] - 18,
                  scale=0.50, thick=1)


def make_object_points(tag_size_m: float) -> np.ndarray:
    s = tag_size_m
    return np.float32([
        [-s / 2,  s / 2, 0.0],
        [ s / 2,  s / 2, 0.0],
        [ s / 2, -s / 2, 0.0],
        [-s / 2, -s / 2, 0.0],
    ]).reshape(-1, 1, 3)


def setup_realsense(args: argparse.Namespace):
    if rs is None:
        raise RuntimeError("pyrealsense2 is not available; install it to use the RealSense mode")

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)

    sensor = profile.get_device().first_color_sensor()
    try:
        if not args.auto_exposure:
            sensor.set_option(rs.option.enable_auto_exposure, 0)
            sensor.set_option(rs.option.exposure, float(args.exposure))
            sensor.set_option(rs.option.gain, float(args.gain))
    except Exception:
        pass

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    K = np.array([[intr.fx, 0.0, intr.ppx],
                  [0.0, intr.fy, intr.ppy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float32)
    header = f"RealSense {intr.width}x{intr.height}@{args.fps} serial={profile.get_device().get_info(rs.camera_info.serial_number)}"
    return pipeline, K, dist_coeffs, header


def setup_webcam(args: argparse.Namespace):
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open webcam index {args.camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read an initial frame from the webcam")

    height, width = frame.shape[:2]
    fov_rad = np.deg2rad(args.webcam_fov)
    fx = (width / 2.0) / np.tan(fov_rad / 2.0)
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    dist_coeffs = np.zeros(5, dtype=np.float32)
    header = f"Webcam {width}x{height}@{args.fps} FOV~{args.webcam_fov:.1f}"
    return cap, frame, K, dist_coeffs, header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize AprilTag pose relative to the camera.")
    parser.add_argument("--camera", choices=("realsense", "webcam"), default="realsense", help="Camera backend to use")
    parser.add_argument("--serial", default="314422070401", help="RealSense serial number (realsense mode)")
    parser.add_argument("--width", type=int, default=1920, help="Stream width")
    parser.add_argument("--height", type=int, default=1080, help="Stream height")
    parser.add_argument("--fps", type=int, default=30, help="Stream FPS")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index (webcam mode)")
    parser.add_argument("--webcam-fov", type=float, default=62.0, help="Estimated horizontal FOV in degrees for webcam mode")
    parser.add_argument("--family", default="tag36h11", help="AprilTag family")
    parser.add_argument("--tag-size", type=float, default=0.031, help="Tag edge length in meters")
    parser.add_argument("--disable-base-tag-log", action="store_true",
                        help="Do not print the T_cam→base_tag matrix (still used internally for computations)")
    parser.add_argument("--auto-exposure", action="store_true", help="Keep RealSense auto exposure enabled")
    parser.add_argument("--exposure", type=float, default=140.0, help="Manual RealSense exposure (when auto exposure is disabled)")
    parser.add_argument("--gain", type=float, default=64.0, help="Manual RealSense gain (when auto exposure is disabled)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = build_detector(args.family)
    log_base_tag = not args.disable_base_tag_log
    T_base_to_base_tag = T_BASE_TO_BASE_TAG

    fps = 0.0
    t_prev = time.time()

    if args.camera == "realsense":
        pipeline, K, dist_coeffs, header = setup_realsense(args)
        try:
            while True:
                frames = pipeline.wait_for_frames()
                cf = frames.get_color_frame()
                if not cf:
                    continue
                img = np.asanyarray(cf.get_data())
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                poses = solve_tag_poses(detector, gray, K, dist_coeffs, args.tag_size)
                T_base_to_test = None
                T_cam_to_base_tag = report_base_tag_transform(poses, log=log_base_tag)
                if T_cam_to_base_tag is not None:
                    T_base_to_test = report_test_tag_transform(
                        poses, T_cam_to_base_tag, T_base_to_base_tag
                    )
                annotate_frame(img, poses, K, fps, header, T_base_to_test=T_base_to_test)

                cv2.imshow("AprilTag Pose (RealSense)", img)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - t_prev))
                t_prev = now
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()
    else:
        cap, first_frame, K, dist_coeffs, header = setup_webcam(args)
        fps = 0.0
        t_prev = time.time()
        try:
            frame = first_frame
            while True:
                if frame is None:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                img = frame.copy()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                poses = solve_tag_poses(detector, gray, K, dist_coeffs, args.tag_size)
                T_base_to_test = None
                T_cam_to_base_tag = report_base_tag_transform(poses, log=log_base_tag)
                if T_cam_to_base_tag is not None:
                    T_base_to_test = report_test_tag_transform(
                        poses, T_cam_to_base_tag, T_base_to_base_tag
                    )
                annotate_frame(img, poses, K, fps, header, T_base_to_test=T_base_to_test)

                cv2.imshow("AprilTag Pose (Webcam)", img)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - t_prev))
                t_prev = now

                ok, frame = cap.read()
                if not ok:
                    frame = None
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
