#!/usr/bin/env python3
import argparse
import csv
import json
import signal
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from xarm.wrapper import XArmAPI
import xml.etree.ElementTree as ET


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect xArm7 joint motion data and controller limits for offline analysis."
    )
    parser.add_argument(
        "--robot-ip",
        required=True,
        help="IP address of the xArm7 controller.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Recording duration in seconds. Leave unset to run until interrupted.",
    )
    parser.add_argument(
        "--collect-stream",
        action="store_true",
        help="Collect a live joint-state stream. If false and duration is unset, only static parameters are queried.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=100.0,
        help="Target sampling rate (Hz) when polling joint states.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./xarm_measurements"),
        help="Directory where the log files will be written.",
    )
    parser.add_argument(
        "--urdf-path",
        type=Path,
        default=Path("robo_manip_baselines/envs/assets/common/robots/xarm7/xarm7.urdf"),
        help="URDF file used to extract joint effort limits.",
    )
    parser.add_argument(
        "--report-type",
        type=str,
        default="rich",
        choices=["normal", "rich"],
        help="xArm report type. Use 'rich' to make joint limits and jerk available.",
    )
    return parser.parse_args()


class GracefulShutdown:
    def __init__(self):
        self._stop = False
        signal.signal(signal.SIGINT, self._request_stop)
        signal.signal(signal.SIGTERM, self._request_stop)

    def _request_stop(self, *_):
        self._stop = True

    @property
    def stop_requested(self) -> bool:
        return self._stop


def connect_robot(robot_ip: str, report_type: str) -> XArmAPI:
    arm = XArmAPI(
        port=robot_ip,
        is_radian=True,
        enable_report=True,
        report_type=report_type,
    )
    arm.connect()
    arm.clean_error()
    arm.motion_enable(enable=True)
    arm.ft_sensor_enable(1)
    time.sleep(0.1)
    arm.ft_sensor_set_zero()
    arm.clean_gripper_error()
    arm.set_gripper_mode(0)
    arm.set_gripper_enable(True)
    arm.set_mode(6)
    arm.set_state(0)
    # Allow the controller to populate the rich report buffers.
    time.sleep(0.5)
    return arm


def read_urdf_effort_limits(urdf_path: Path) -> List[Dict[str, float]]:
    if not urdf_path.exists():
        return []

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    limits: List[Dict[str, float]] = []
    for joint_elem in root.findall("joint"):
        if joint_elem.get("type") != "revolute":
            continue
        limit_elem = joint_elem.find("limit")
        if limit_elem is None or "effort" not in limit_elem.attrib:
            continue
        try:
            effort_val = float(limit_elem.attrib["effort"])
        except ValueError:
            continue
        limits.append(
            {
                "joint": joint_elem.get("name", ""),
                "effort_limit_Nm": effort_val,
            }
        )
    return limits


def _init_extrema(num_joints: int) -> List[float]:
    return [0.0 for _ in range(num_joints)]


def _update_extrema(extrema: List[float], values: Iterable[float]) -> None:
    for idx, value in enumerate(values):
        extrema[idx] = max(extrema[idx], abs(float(value)))


def _sleep_remaining(start_ts: float, period: float) -> None:
    elapsed = time.perf_counter() - start_ts
    remaining = period - elapsed
    if remaining > 0:
        time.sleep(remaining)


def collect_joint_stream(
    arm: XArmAPI,
    sample_rate: float,
    duration: Optional[float],
    shutdown_flag: GracefulShutdown,
) -> Dict[str, object]:
    sample_period = 1.0 / max(sample_rate, 1.0)
    records: List[Dict[str, object]] = []

    max_velocity = _init_extrema(7)
    max_acceleration = _init_extrema(7)
    max_jerk = _init_extrema(7)

    last_timestamp = None
    last_velocity = None
    last_acceleration = None

    start_wall_clock = datetime.now(timezone.utc).isoformat()
    start_perf = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        code, joint_state_list = arm.get_joint_states(is_radian=True)
        if code != 0:
            raise RuntimeError(f"get_joint_states failed with code {code}")

        elapsed = loop_start - start_perf
        positions = [float(val) for val in joint_state_list[0]]
        velocities = [float(val) for val in joint_state_list[1]] if len(joint_state_list) > 1 else [0.0] * 7
        efforts = [float(val) for val in joint_state_list[2]] if len(joint_state_list) > 2 else [0.0] * 7

        records.append(
            {
                "elapsed_s": elapsed,
                "positions_rad": positions,
                "velocities_rad_s": velocities,
                "efforts": efforts,
            }
        )
        _update_extrema(max_velocity, velocities)

        if last_timestamp is not None and last_velocity is not None:
            dt = max(loop_start - last_timestamp, 1e-9)
            acceleration = [(cur - prev) / dt for cur, prev in zip(velocities, last_velocity)]
            _update_extrema(max_acceleration, acceleration)

            if last_acceleration is not None:
                jerk = [(cur - prev) / dt for cur, prev in zip(acceleration, last_acceleration)]
                _update_extrema(max_jerk, jerk)
            last_acceleration = acceleration

        last_velocity = velocities
        last_timestamp = loop_start

        if duration is not None and elapsed >= duration:
            break
        if shutdown_flag.stop_requested:
            break

        _sleep_remaining(loop_start, sample_period)

    end_wall_clock = datetime.now(timezone.utc).isoformat()
    total_time = records[-1]["elapsed_s"] if records else 0.0
    achieved_rate = (len(records) - 1) / total_time if len(records) > 1 and total_time > 0 else 0.0

    return {
        "records": records,
        "start_wall_time_utc": start_wall_clock,
        "end_wall_time_utc": end_wall_clock,
        "duration_s": total_time,
        "sample_count": len(records),
        "achieved_rate_hz": achieved_rate,
        "max_velocity_rad_s": max_velocity,
        "max_acceleration_rad_s2": max_acceleration,
        "max_jerk_rad_s3": max_jerk,
    }


def write_timeseries_csv(csv_path: Path, records: List[Dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        header = ["elapsed_s"]
        header += [f"joint{i+1}_pos_rad" for i in range(7)]
        header += [f"joint{i+1}_vel_rad_s" for i in range(7)]
        header += [f"joint{i+1}_effort" for i in range(7)]
        writer.writerow(header)
        for entry in records:
            row = [f"{entry['elapsed_s']:.9f}"]
            row.extend(f"{value:.9f}" for value in entry["positions_rad"])
            row.extend(f"{value:.9f}" for value in entry["velocities_rad_s"])
            row.extend(f"{value:.9f}" for value in entry["efforts"])
            writer.writerow(row)


def build_joint_summary(
    joint_labels: List[str],
    max_velocity: List[float],
    max_acceleration: List[float],
    max_jerk: List[float],
) -> List[Dict[str, float]]:
    summary: List[Dict[str, float]] = []
    for idx, label in enumerate(joint_labels):
        summary.append(
            {
                "joint": label,
                "max_velocity_rad_s": max_velocity[idx],
                "max_acceleration_rad_s2": max_acceleration[idx],
                "max_jerk_rad_s3": max_jerk[idx],
            }
        )
    return summary


def snapshot_controller_limits(arm: XArmAPI) -> Dict[str, object]:
    limit_snapshot: Dict[str, object] = OrderedDict()
    try:
        speed_limit = arm.joint_speed_limit
        if speed_limit:
            limit_snapshot["joint_speed_limit_rad_s"] = [float(val) for val in speed_limit]
    except Exception:  # noqa: BLE001
        pass

    try:
        acc_limit = arm.joint_acc_limit
        if acc_limit:
            limit_snapshot["joint_acc_limit_rad_s2"] = [float(val) for val in acc_limit]
    except Exception:  # noqa: BLE001
        pass

    try:
        jerk_limit = arm.joint_jerk
        if jerk_limit is not None:
            limit_snapshot["joint_jerk_rad_s3"] = float(jerk_limit)
    except Exception:  # noqa: BLE001
        pass

    try:
        last_speed = arm.last_used_joint_speed
        limit_snapshot["last_used_joint_speed_rad_s"] = float(last_speed)
    except Exception:  # noqa: BLE001
        pass

    try:
        last_acc = arm.last_used_joint_acc
        limit_snapshot["last_used_joint_acc_rad_s2"] = float(last_acc)
    except Exception:  # noqa: BLE001
        pass

    return limit_snapshot


def fetch_internal_params(arm: XArmAPI) -> Dict[str, object]:
    try:
        params = arm.arm._get_params(is_radian=True)
    except Exception:  # noqa: BLE001
        return {}

    formatted: Dict[str, object] = {}
    for key, value in params.items():
        if isinstance(value, (list, tuple)):
            formatted[key] = [float(v) for v in value]
        elif isinstance(value, (int, float)):
            formatted[key] = float(value)
        else:
            formatted[key] = value
    return formatted


def main() -> None:
    args = parse_arguments()
    shutdown_flag = GracefulShutdown()

    try:
        arm = connect_robot(args.robot_ip, args.report_type)
    except Exception as exc:  # noqa: BLE001
        print(f"[CollectXarm7Dynamics] Failed to connect to xArm7 at {args.robot_ip}: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        urdf_limits = read_urdf_effort_limits(args.urdf_path)
        joint_labels = [item["joint"] for item in urdf_limits] if urdf_limits else [f"joint{i+1}" for i in range(7)]

        collect_stream = args.collect_stream or args.duration is not None
        stream_result = None
        if collect_stream:
            stream_result = collect_joint_stream(
                arm=arm,
                sample_rate=args.sample_rate,
                duration=args.duration,
                shutdown_flag=shutdown_flag,
            )

        output_dir = args.output_dir.resolve()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        prefix = output_dir / f"xarm7_dynamics_{timestamp}"
        csv_path = prefix.with_suffix(".csv")
        json_path = prefix.with_suffix(".json")

        artifacts: Dict[str, Optional[str]] = {
            "timeseries_csv": None,
        }

        if stream_result is not None and stream_result["records"]:
            write_timeseries_csv(csv_path, stream_result["records"])
            artifacts["timeseries_csv"] = str(csv_path)

        controller_limits = snapshot_controller_limits(arm)
        joint_summary = build_joint_summary(
            joint_labels=joint_labels,
            max_velocity=stream_result["max_velocity_rad_s"] if stream_result else [0.0] * len(joint_labels),
            max_acceleration=stream_result["max_acceleration_rad_s2"] if stream_result else [0.0] * len(joint_labels),
            max_jerk=stream_result["max_jerk_rad_s3"] if stream_result else [0.0] * len(joint_labels),
        )
        internal_params = fetch_internal_params(arm)

        summary_payload = {
            "metadata": {
                "robot_ip": args.robot_ip,
                "start_wall_time_utc": stream_result["start_wall_time_utc"] if stream_result else datetime.now(timezone.utc).isoformat(),
                "end_wall_time_utc": stream_result["end_wall_time_utc"] if stream_result else datetime.now(timezone.utc).isoformat(),
                "duration_s": stream_result["duration_s"] if stream_result else 0.0,
                "sample_count": stream_result["sample_count"] if stream_result else 0,
                "sample_rate_requested_hz": args.sample_rate,
                "sample_rate_achieved_hz": stream_result["achieved_rate_hz"] if stream_result else 0.0,
                "report_type": args.report_type,
                "stream_collected": bool(stream_result and stream_result["records"]),
            },
            "observed_joint_limits": joint_summary,
            "controller_reported_limits": controller_limits,
            "urdf_effort_limits": urdf_limits,
            "internal_controller_params": internal_params,
            "artifacts": artifacts,
            "notes": [
                "Controller parameters are queried via the xArm SDK; PD gains still require manual identification.",
            ],
        }

        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w") as json_file:
            json.dump(summary_payload, json_file, indent=2)

        if artifacts["timeseries_csv"]:
            print(f"[CollectXarm7Dynamics] Wrote joint samples to {csv_path}")
        print(f"[CollectXarm7Dynamics] Wrote summary to {json_path}")

    finally:
        try:
            arm.disconnect()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
