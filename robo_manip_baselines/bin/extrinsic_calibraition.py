"""Capture T_base→camera using the calibrated base_tag pose and persist it."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pose_viewer import (
    T_BASE_TAG_TO_BASE_DEFAULT,
    build_detector,
    report_base_tag_transform,
    setup_realsense,
    setup_webcam,
    solve_tag_poses,
)

T_BASE_TO_BASE_TAG_DEFAULT = np.linalg.inv(T_BASE_TAG_TO_BASE_DEFAULT)
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "calib" / "T_base_to_camera.csv"


def save_transform(matrix: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, delimiter=",")
    print(f"Saved T_base→camera to {path.resolve()}", flush=True)


def load_transform(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        matrix = np.loadtxt(path, delimiter=",").reshape(4, 4)
    except Exception as exc:
        raise ValueError(f"Failed to load transform from {path}: {exc}") from exc
    return matrix


def make_status_text(T_base_to_camera: Optional[np.ndarray], is_saved: bool) -> str:
    if T_base_to_camera is None:
        return "base_tag not detected"
    tx, ty, tz = T_base_to_camera[:3, 3]
    rpy = rotmat_to_rpy_deg(T_base_to_camera[:3, :3])
    status = (
        f"T_base→camera XYZ=({tx:+.3f},{ty:+.3f},{tz:+.3f}) m "
        f"RPY=({rpy[0]:+.1f},{rpy[1]:+.1f},{rpy[2]:+.1f}) deg"
    )
    if is_saved:
        status += " [saved]"
    return status


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture and store the T_base→camera extrinsic transform.")
    parser.add_argument("--camera", choices=("realsense", "webcam"), default="realsense", help="Camera backend to use")
    parser.add_argument("--serial", default="314422070401", help="RealSense serial number (realsense mode)")
    parser.add_argument("--width", type=int, default=1920, help="Stream width")
    parser.add_argument("--height", type=int, default=1080, help="Stream height")
    parser.add_argument("--fps", type=int, default=30, help="Stream FPS")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index (webcam mode)")
    parser.add_argument("--webcam-fov", type=float, default=62.0, help="Estimated horizontal FOV in degrees for webcam mode")
    parser.add_argument("--tag-size", type=float, default=0.031, help="Default tag edge length in meters")
    parser.add_argument("--family", default="tag36h11", help="AprilTag family")
    parser.add_argument("--auto-exposure", action="store_true", help="Keep RealSense auto exposure enabled")
    parser.add_argument("--exposure", type=float, default=140.0, help="Manual RealSense exposure (when auto exposure is disabled)")
    parser.add_argument("--gain", type=float, default=64.0, help="Manual RealSense gain (when auto exposure is disabled)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH,
                        help="Destination file for the captured transform (CSV).")
    parser.add_argument("--quiet", action="store_true", help="Suppress console logs for T_cam→base_tag.")
    parser.add_argument("--load", action="store_true", help="Load and print the stored transform without capturing.")
    return parser.parse_args()


def draw_status(img: np.ndarray, text: str) -> None:
    if not text:
        return
    y = img.shape[0] - 30
    cv2.rectangle(img, (0, y - 30), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)


def run_capture(args: argparse.Namespace) -> None:
    detector = build_detector(args.family)
    log_base_tag = not args.quiet

    if args.camera == "realsense":
        pipeline, K, dist_coeffs, header = setup_realsense(args)
        cap_type = "RealSense"
        pipeline_started = True
    else:
        cap, first_frame, K, dist_coeffs, header = setup_webcam(args)
        cap_type = "Webcam"
        pipeline_started = False
        frame_buffer = first_frame.copy()

    print("Press 's' to save the current T_base→camera, 'q' to quit.", flush=True)
    T_base_to_camera: Optional[np.ndarray] = None
    saved_matrix: Optional[np.ndarray] = load_transform(args.output)
    if saved_matrix is not None:
        print(f"Loaded existing transform from {args.output}:\n{saved_matrix}", flush=True)

    fps = 0.0
    t_prev = time.time()
    try:
        while True:
            if pipeline_started:
                frames = pipeline.wait_for_frames()
                cf = frames.get_color_frame()
                if not cf:
                    continue
                img = np.asanyarray(cf.get_data())
            else:
                if frame_buffer is not None:
                    img = frame_buffer.copy()
                    frame_buffer = None
                else:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    img = frame.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            poses = solve_tag_poses(detector, gray, K, dist_coeffs, args.tag_size)
            T_cam_to_base_tag = report_base_tag_transform(poses, log=log_base_tag)
            if T_cam_to_base_tag is not None:
                T_base_tag_from_camera = np.linalg.inv(T_cam_to_base_tag)
                T_base_to_camera = T_BASE_TO_BASE_TAG_DEFAULT @ T_base_tag_from_camera
                if log_base_tag:
                    matrix_str = np.array2string(
                        T_base_to_camera,
                        formatter={"float_kind": lambda x: f"{x: .4f}"},
                    )
                    print(f"{cap_type} T_base→camera:\n{matrix_str}\n", flush=True)

            status_text = make_status_text(T_base_to_camera, saved_matrix is not None)
            draw_status(img, status_text)
            cv2.putText(img, header, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Extrinsic Calibration", img)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s") and T_base_to_camera is not None:
                save_transform(T_base_to_camera, args.output)
                saved_matrix = T_base_to_camera.copy()

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - t_prev))
            t_prev = now
    finally:
        if pipeline_started:
            pipeline.stop()
        else:
            cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    if args.load:
        matrix = load_transform(args.output)
        if matrix is None:
            print(f"No transform stored at {args.output}", flush=True)
        else:
            print(f"Loaded T_base→camera from {args.output}:\n{matrix}", flush=True)
        return
    run_capture(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
