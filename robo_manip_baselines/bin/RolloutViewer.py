import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import yaml

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pyrealsense2 is required for RolloutViewer. Please install Intel RealSense SDK."
    ) from exc


def parse_size(value: Optional[str], default: Tuple[int, int]) -> Tuple[int, int]:
    if value is None:
        return default
    try:
        width_str, height_str = value.lower().split("x")
        return int(width_str), int(height_str)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            f"Invalid size specification '{value}'. Use the form WIDTHxHEIGHT."
        ) from err


def load_yaml_camera_id(config_path: Optional[str], camera_name: str) -> Optional[str]:
    if config_path is None:
        return None
    path_obj = Path(config_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with path_obj.open("r") as file:
        data = yaml.safe_load(file) or {}
    camera_ids: Dict[str, str] = data.get("camera_ids", {})
    if camera_name not in camera_ids:
        available = ", ".join(camera_ids) or "<none>"
        raise ValueError(
            f"Camera '{camera_name}' not listed in config. Available cameras: {available}"
        )
    return camera_ids[camera_name]


class RealSenseStream:
    def __init__(
        self,
        serial: Optional[str],
        color_size: Tuple[int, int],
        depth_size: Tuple[int, int],
        color_fps: int,
        depth_fps: int,
        align_to_color: bool,
    ) -> None:
        self.serial = serial
        self.color_size = color_size
        self.depth_size = depth_size
        self.color_fps = color_fps
        self.depth_fps = depth_fps
        self.align_to_color = align_to_color
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color) if align_to_color else None
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        self.stop()

    def start(self):
        if self._started:
            return
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(
            rs.stream.color,
            self.color_size[0],
            self.color_size[1],
            rs.format.bgr8,
            self.color_fps,
        )
        config.enable_stream(
            rs.stream.depth,
            self.depth_size[0],
            self.depth_size[1],
            rs.format.z16,
            self.depth_fps,
        )
        try:
            self.pipeline.start(config)
            self._started = True
        except RuntimeError as err:
            if self.serial:
                raise RuntimeError(
                    f"Failed to start RealSense pipeline for serial {self.serial}: {err}"
                ) from err
            raise RuntimeError(f"Failed to start RealSense pipeline: {err}") from err

    def stop(self):
        if not self._started:
            return
        self.pipeline.stop()
        self._started = False

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        frames = self.pipeline.wait_for_frames()
        if self.align:
            frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame:
            raise RuntimeError("Failed to retrieve color frame from RealSense camera")
        if not depth_frame:
            raise RuntimeError("Failed to retrieve depth frame from RealSense camera")
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        return color, depth


class ImageDisplay:
    def __init__(
        self,
        window_name: str,
        scale: float,
        fullscreen: bool,
        keep_aspect: bool,
    ) -> None:
        self.window_name = window_name
        self.scale = scale
        self.fullscreen = fullscreen
        self.keep_aspect = keep_aspect
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(
                self.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )

    def show(self, image: np.ndarray):
        display_image = image
        if not self.fullscreen and self.scale != 1.0:
            interp = cv2.INTER_LINEAR if self.scale > 1.0 else cv2.INTER_AREA
            new_size = (
                int(round(image.shape[1] * self.scale)),
                int(round(image.shape[0] * self.scale)),
            )
            display_image = cv2.resize(image, new_size, interpolation=interp)
            if self.keep_aspect:
                cv2.resizeWindow(self.window_name, new_size[0], new_size[1])
        cv2.imshow(self.window_name, display_image)

    def close(self):
        cv2.destroyWindow(self.window_name)


def depth_to_colormap(depth: np.ndarray, clip_min: Optional[float], clip_max: Optional[float]) -> np.ndarray:
    depth_float = depth.astype(np.float32)
    if clip_min is not None or clip_max is not None:
        depth_float = np.clip(depth_float, clip_min or 0.0, clip_max or np.max(depth_float))
    valid = depth_float > 0
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    depth_norm = np.zeros_like(depth_float, dtype=np.float32)
    valid_values = depth_float[valid]
    min_val = valid_values.min()
    max_val = valid_values.max()
    if max_val - min_val < 1e-6:
        depth_norm[valid] = 0.5
    else:
        depth_norm[valid] = (valid_values - min_val) / (max_val - min_val)
    depth_uint8 = (255 * depth_norm).astype(np.uint8)
    colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    colored[~valid] = 0
    return colored


def list_devices() -> Dict[str, str]:
    context = rs.context()
    devices = {}
    for device in context.devices:  # type: ignore[attr-defined]
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        devices[serial] = name
    return devices


def setup_signal_handlers(stop_callback):
    def _handle(sig, _frame):
        stop_callback()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)


def parse_arguments():
    parser = argparse.ArgumentParser(description="High-quality RealSense live viewer")
    parser.add_argument("--serial", help="RealSense device serial number", default=None)
    parser.add_argument(
        "--config",
        help="YAML config with camera_ids (RealXarm7DemoEnv-style)",
        default=None,
    )
    parser.add_argument("--camera", help="Camera key inside the YAML config", default="front")
    parser.add_argument("--color-size", help="Color resolution (e.g. 1280x720)")
    parser.add_argument("--depth-size", help="Depth resolution (e.g. 640x480)")
    parser.add_argument("--color-fps", type=int, default=30)
    parser.add_argument("--depth-fps", type=int, default=30)
    parser.add_argument("--scale", type=float, default=1.0, help="Window scaling factor")
    parser.add_argument("--fullscreen", action="store_true", help="Open the color window in fullscreen")
    parser.add_argument("--show-depth", action="store_true", help="Show depth colormap in a second window")
    parser.add_argument("--depth-clip", nargs=2, type=float, metavar=("MIN", "MAX"), help="Optional depth clipping range in meters")
    parser.add_argument("--no-align", action="store_true", help="Do not align depth to color")
    parser.add_argument("--list-devices", action="store_true", help="Print available RealSense devices and exit")
    parser.add_argument(
        "--keep-aspect",
        action="store_true",
        help="Resize the window to match scaled output when not fullscreen",
    )
    args = parser.parse_args()
    if args.list_devices:
        devices = list_devices()
        if not devices:
            print("No RealSense devices detected.")
        else:
            print("Detected RealSense devices:")
            for serial, name in devices.items():
                print(f"  {serial}: {name}")
        sys.exit(0)
    if args.config and args.serial:
        raise ValueError("Specify either --serial or --config/--camera, not both.")
    depth_clip = (None, None)
    if args.depth_clip:
        depth_clip = (args.depth_clip[0], args.depth_clip[1])
    return args, depth_clip


def main():
    args, depth_clip = parse_arguments()
    serial = args.serial
    if serial is None:
        serial = load_yaml_camera_id(args.config, args.camera)

    color_size = parse_size(args.color_size, (1280, 720))
    depth_size = parse_size(args.depth_size, (640, 480))

    real_sense_stream = RealSenseStream(
        serial=serial,
        color_size=color_size,
        depth_size=depth_size,
        color_fps=args.color_fps,
        depth_fps=args.depth_fps,
        align_to_color=not args.no_align,
    )

    stop_requested = False

    def request_stop():
        nonlocal stop_requested
        stop_requested = True

    setup_signal_handlers(request_stop)

    color_display = ImageDisplay(
        window_name="RealSense Color",
        scale=args.scale,
        fullscreen=args.fullscreen,
        keep_aspect=args.keep_aspect,
    )
    depth_display = (
        ImageDisplay(
            window_name="RealSense Depth",
            scale=args.scale,
            fullscreen=False,
            keep_aspect=args.keep_aspect,
        )
        if args.show_depth
        else None
    )

    frame_counter = 0
    fps_timer = time.time()

    try:
        with real_sense_stream:
            while not stop_requested:
                color_frame, depth_frame = real_sense_stream.read()
                frame_counter += 1

                elapsed = time.time() - fps_timer
                if elapsed >= 1.0:
                    fps_text = f"FPS: {frame_counter / elapsed:.1f}"
                    fps_timer = time.time()
                    frame_counter = 0
                else:
                    fps_text = None

                if fps_text:
                    cv2.putText(
                        color_frame,
                        fps_text,
                        (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                color_display.show(color_frame)

                if depth_display is not None:
                    depth_vis = depth_to_colormap(
                        depth_frame.astype(np.float32) * 0.001,
                        depth_clip[0],
                        depth_clip[1],
                    )
                    depth_display.show(depth_vis)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
    finally:
        color_display.close()
        if depth_display is not None:
            depth_display.close()
        real_sense_stream.stop()


if __name__ == "__main__":
    main()
