#!/usr/bin/env python3
"""First-stage dual-FR3 ACT native RMB rollout (12 arm joints ONLY).

Uses RoboManipBaselines' ORIGINAL RolloutAct.infer_policy and temporal ensembling.
No modifications to RMB's RolloutAct, Fairino controller or ServoJ bridge.

Default = PREVIEW ONLY. --execute requires a hardware start confirmation.
This is experimental software, NOT a certified safety controller. The user's
physical E-stop and existing bridge limits remain essential.

Place in the RoboManipBaselines scripts/ directory and run from repository root.
Requires an existing session-aware ServoJ bridge that reports session_id and
servo_end_ok, as used by the user's prior verified STOP/RESUME rollout.
"""
import json
import time

import gymnasium as gym
import numpy as np
import rclpy
from gymnasium.spaces import Box, Dict
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, String

# Import to register existing RMB environments, then dynamically register ours.
import robo_manip_baselines.envs
from robo_manip_baselines.common import ArmConfig, DataKey
from robo_manip_baselines.envs.real.RealEnvBase import RealEnvBase
from robo_manip_baselines.policy.act.RolloutAct import RolloutAct

ENV_ID = "robo_manip_baselines/RealDualFR3ArmOnlyNative-v0"
ARM_KEYS = ("fr1", "fr2")
CAMERA_KEYS = ("front", "hand")
JOINT_TOPICS = {"fr1": "/fairino_1/joint_states", "fr2": "/fairino_2/joint_states"}
IMAGE_TOPICS = {
    "front": "/camera_d405_1/color/image_rect_raw",
    "hand": "/camera_d405_2/color/image_rect_raw",
}
BRIDGES = {"fr1": "/fairino_1/act_servoj", "fr2": "/fairino_2/act_servoj"}


def ordered_joints(msg):
    names = [f"joint_{i}" for i in range(1, 7)]
    if len(set(msg.name)) != len(msg.name) or len(msg.name) != len(msg.position):
        raise ValueError("Invalid JointState names/positions")
    mapping = dict(zip(msg.name, msg.position))
    if any(name not in mapping for name in names):
        raise ValueError("Missing joint_1...joint_6 in joint feedback")
    result = np.array([mapping[n] for n in names], dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite joint feedback")
    return result


def rgb_frame(msg):
    if msg.encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"Expected rgb8 or bgr8, got {msg.encoding}")
    if msg.step < msg.width * 3:
        raise ValueError("Invalid Image row stride")
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if data.size != msg.height * msg.step:
        raise ValueError("Invalid Image buffer length")
    out = data.reshape(msg.height, msg.step)[:, :3*msg.width]
    out = out.reshape(msg.height, msg.width, 3)
    if msg.encoding == "bgr8":
        out = out[..., ::-1]
    return out.copy()


class FR3Topics(Node):
    """One ROS client; does not open a Fairino SDK RPC connection."""

    def __init__(self):
        super().__init__("fr3_rmb_native_act_arm_only")
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        self.q = {}
        self.q_at = {}
        self.rgb = {}
        self.rgb_at = {}
        self.status = {a: None for a in ARM_KEYS}
        self.status_at = {a: 0.0 for a in ARM_KEYS}
        self.target_pub = {}
        self.arm_pub = {}
        self.stop_pub = {}
        self.subs = []
        for arm in ARM_KEYS:
            self.subs.append(self.create_subscription(
                JointState, JOINT_TOPICS[arm],
                lambda msg, a=arm: self._joint(a, msg), qos))
        for camera in CAMERA_KEYS:
            self.subs.append(self.create_subscription(
                Image, IMAGE_TOPICS[camera],
                lambda msg, c=camera: self._image(c, msg), qos))
        for arm in ARM_KEYS:
            prefix = BRIDGES[arm]
            self.target_pub[arm] = self.create_publisher(JointState, prefix + "/target", 10)
            self.arm_pub[arm] = self.create_publisher(Bool, prefix + "/arm", 10)
            self.stop_pub[arm] = self.create_publisher(Bool, prefix + "/stop", 10)
            self.subs.append(self.create_subscription(
                String, prefix + "/status", lambda msg, a=arm: self._status(a, msg),
                QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL)))

    def _joint(self, arm, msg):
        try:
            self.q[arm] = ordered_joints(msg)
            self.q_at[arm] = time.monotonic()
        except Exception as exc:
            self.get_logger().warning(f"{arm} invalid joint feedback: {exc}")

    def _image(self, camera, msg):
        try:
            self.rgb[camera] = rgb_frame(msg)
            self.rgb_at[camera] = time.monotonic()
        except Exception as exc:
            self.get_logger().warning(f"{camera} invalid RGB image: {exc}")

    def _status(self, arm, msg):
        try:
            value = json.loads(msg.data)
            if not isinstance(value, dict):
                raise ValueError("status is not JSON object")
            self.status[arm] = value
            self.status_at[arm] = time.monotonic()
        except Exception as exc:
            self.get_logger().warning(f"{arm} invalid bridge status: {exc}")

    def pump(self, seconds):
        deadline = time.monotonic() + max(0, seconds)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.01, max(0, deadline - time.monotonic())))

    def wait_sources(self, timeout_s=6.0):
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            if self.sources_fresh():
                return
        raise RuntimeError("No fresh dual-FR3 joints + both RGB cameras; "
                           f"joints={list(self.q)}, RGB={list(self.rgb)}")

    def sources_fresh(self, max_age=0.28, max_spread=0.13):
        if any(a not in self.q for a in ARM_KEYS) or any(c not in self.rgb for c in CAMERA_KEYS):
            return False
        stamps = [self.q_at[a] for a in ARM_KEYS] + [self.rgb_at[c] for c in CAMERA_KEYS]
        now = time.monotonic()
        return (now - min(stamps) <= max_age and max(stamps) - min(stamps) <= max_spread)

    def measured_q12(self):
        if any(a not in self.q for a in ARM_KEYS):
            raise RuntimeError("Joint data unavailable")
        return np.concatenate([self.q[a] for a in ARM_KEYS]).copy()

    def target(self, q12):
        q12 = np.asarray(q12, dtype=np.float64).reshape(-1)
        if q12.shape != (12,) or not np.isfinite(q12).all():
            raise ValueError("Target must be 12 finite joint positions (radians)")
        for i, arm in enumerate(ARM_KEYS):
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = [f"joint_{j}" for j in range(1, 7)]
            msg.position = q12[i*6:(i+1)*6].tolist()
            self.target_pub[arm].publish(msg)

    def arm_both(self):
        msg = Bool(data=True)
        for arm in ARM_KEYS:
            self.arm_pub[arm].publish(msg)

    def stop_both(self):
        msg = Bool(data=True)
        for arm in ARM_KEYS:
            self.stop_pub[arm].publish(msg)


class JointOnlyMotionManager:
    """RMB MotionManager-compatible 12D joint-only adapter; no fake URDF/FK.

    This stage intentionally does not instantiate ArmManager/Pinocchio: there
    is no gripper or EEF action, just calibrated arm joint angles in radians.
    """
    def __init__(self, env):
        self.env = env
        self.command = np.zeros(12, dtype=np.float64)

    def reset(self):
        env = self.env.unwrapped
        if env.topics.q and all(a in env.topics.q for a in ARM_KEYS):
            self.command = env.topics.measured_q12()
        else:
            self.command = np.zeros(12, dtype=np.float64)

    def get_data(self, key, obs=None):
        if key == DataKey.MEASURED_JOINT_POS:
            return self.get_measured_data(key, obs)
        if key == DataKey.COMMAND_JOINT_POS:
            return self.get_command_data(key)
        raise ValueError(f"Unsupported first-stage 12D key: {key}")

    def get_measured_data(self, key, obs):
        if key == DataKey.MEASURED_JOINT_POS:
            return np.asarray(obs["joint_pos"], dtype=np.float64).copy()
        raise ValueError(f"Unsupported measured key: {key}")

    def set_command_data(self, key, command, is_skip=False):
        if key != DataKey.COMMAND_JOINT_POS:
            raise ValueError(f"Unsupported command key: {key}")
        q = np.asarray(command, dtype=np.float64).reshape(-1)
        if q.shape != (12,) or not np.isfinite(q).all():
            raise ValueError("ACT joint action must be finite 12D radians")
        self.command = q.copy()

    def get_command_data(self, key):
        if key != DataKey.COMMAND_JOINT_POS:
            raise ValueError(f"Unsupported command key: {key}")
        return self.command.copy()

    def draw_markers(self):
        pass


class RealDualFR3ArmOnlyNative(RealEnvBase):
    """Joint-only environment. No gripper operation, reset motion or SDK connection."""
    action_space = Box(-np.pi*2, np.pi*2, shape=(12,), dtype=np.float64)
    observation_space = Dict({
        "joint_pos": Box(-np.inf, np.inf, shape=(12,), dtype=np.float64),
        "joint_vel": Box(-np.inf, np.inf, shape=(12,), dtype=np.float64),
    })

    def __init__(self, execute=False, arm_max_duration=2.5, max_target_error_deg=3.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.dt = 1.0 / 15.0
        self.camera_names = list(CAMERA_KEYS)
        self.pointcloud_camera_names = []
        self.rgb_tactile_names = []
        self.intensity_tactile_names = []
        self.target_task = None
        self.world_random_factors = None
        self.execute = bool(execute)
        self.arm_max_duration = float(arm_max_duration)
        self.max_target_error_deg = float(max_target_error_deg)
        self.active = False
        self.armed_at = None
        self.session_ids = {}
        self.closed = False
        # This is ONLY for DataKey's 12D indexing. No ArmManager is created and
        # arm_urdf_path is intentionally blank; do NOT use for FK/EEF planning.
        no_gripper = np.array([], dtype=int)
        self.body_config_list = [
            ArmConfig(arm_urdf_path="", arm_root_pose=None, ik_eef_joint_id=0,
                      arm_joint_idxes=np.arange(i*6, (i+1)*6),
                      gripper_joint_idxes=no_gripper,
                      gripper_joint_idxes_in_gripper_joint_pos=no_gripper,
                      eef_idx=None, init_arm_joint_pos=np.zeros(6),
                      init_gripper_joint_pos=np.zeros(0))
            for i in range(2)
        ]
        self.owns_ros = not rclpy.ok()
        if self.owns_ros:
            rclpy.init()
        self.topics = FR3Topics()
        self.topics.wait_sources(timeout_s=10.0)
        print("[FR3] ROS topics ready; SDK connection remains owned by fairino_ctrl")

    @property
    def measured_keys_to_save(self):
        return [DataKey.MEASURED_JOINT_POS]

    @property
    def command_keys_to_save(self):
        return [DataKey.COMMAND_JOINT_POS]

    def get_camera_fovy(self, camera_name):
        # Unused by this no-save first-stage test. No calibration is assumed.
        return float("nan")

    def modify_world(self, world_idx=None, cumulative_idx=None):
        return 0 if world_idx is None else int(world_idx)

    def _reset_robot(self):
        # NEVER move the hardware to a "home" pose automatically.
        if self.active:
            raise RuntimeError("Reset refused while ServoJ session active")
        self.topics.wait_sources(timeout_s=5.0)
        print("[FR3] Reset: read current robot pose only; no robot command")

    def _get_obs(self):
        self.topics.wait_sources(timeout_s=0.50)
        q = self.topics.measured_q12()
        return {"joint_pos": q, "joint_vel": np.zeros(12, dtype=np.float64)}

    def _get_info(self):
        if not self.topics.sources_fresh():
            self.topics.wait_sources(timeout_s=0.50)
        return {"rgb_images": {c: self.topics.rgb[c].copy() for c in CAMERA_KEYS}}

    def start_arms(self, first_policy_target):
        self.topics.wait_sources(timeout_s=1.0)
        measured = self.topics.measured_q12()
        delta_deg = np.rad2deg(first_policy_target - measured)
        print("[ACT FIRST TARGET] FR3 #1 deltas (deg):", np.round(delta_deg[:6], 2))
        print("[ACT FIRST TARGET] FR3 #2 deltas (deg):", np.round(delta_deg[6:], 2))
        if not self.execute:
            print("[PREVIEW] Original RMB temporal ensembling active; NO ServoJ ARM/targets")
            return
        if self.active:
            return
        initial_delta = np.max(np.abs(delta_deg))
        if initial_delta > 1.0:
            raise RuntimeError(f"Initial predicted target {initial_delta:.2f} deg away; "
                               "require <=1.00 deg before arming")
        if any(self.topics.status[a] is None for a in ARM_KEYS):
            raise RuntimeError("Missing ServoJ bridge status")
        for arm in ARM_KEYS:
            st = self.topics.status[arm]
            if st.get("state") != "idle" or not isinstance(st.get("session_id"), int):
                raise RuntimeError(f"{arm}: bridge must be idle and session-aware: {st}")
            if time.monotonic() - self.topics.status_at[arm] > 10.0:
                raise RuntimeError(f"{arm}: idle status too old; reverify controller")
        print("[HARDWARE] First ACT target <=1 degree from measured pose.")
        print("[HARDWARE] Verify both FR3 arms stationary, clear swept paths, "
              "no competing motion source, and physical independent E-stop.")
        if input("Type ARM FR3 ACT to start both ServoJ sessions: ").strip() != "ARM FR3 ACT":
            raise RuntimeError("Operator declined ServoJ ARM")
        # Re-observe after operator confirmation. Do not arm from old observations.
        self.topics.wait_sources(timeout_s=1.0)
        measured = self.topics.measured_q12()
        if np.max(np.abs(np.rad2deg(first_policy_target - measured))) > 1.0:
            raise RuntimeError("Robot moved since preview; refusing to arm")
        self.session_ids = {a: self.topics.status[a]["session_id"] + 1 for a in ARM_KEYS}
        requested_at = time.monotonic()
        # Set active BEFORE ARM publish so cleanup always requests STOP.
        self.active = True
        self.armed_at = requested_at
        self.topics.arm_both()
        deadline = requested_at + 2.0
        next_hold = 0.0
        while rclpy.ok() and time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_hold:
                self.topics.target(measured)  # Stationary startup target
                next_hold = now + self.dt
            rclpy.spin_once(self.topics, timeout_sec=0.01)
            states = []
            for arm in ARM_KEYS:
                if self.topics.status_at[arm] <= requested_at:
                    states.append(None)
                    continue
                st = self.topics.status[arm]
                if st.get("state") in ("fault", "disabled", "busy", "stopping", "stopped"):
                    raise RuntimeError(f"{arm} bridge rejected start: {st}")
                if st.get("session_id") == self.session_ids[arm]:
                    states.append(st.get("state"))
                else:
                    states.append(None)
            if states == ["running", "running"]:
                print("[FR3] Both ServoJ bridges running; beginning bounded arm-only phase")
                return
        raise RuntimeError("Timed out waiting for both ServoJ running statuses")

    def _set_action(self, action, duration=None, joint_vel_limit_scale=2.0, wait=True):
        t0 = time.monotonic()
        q = np.asarray(action, dtype=np.float64).reshape(-1)
        if q.shape != (12,) or not np.isfinite(q).all():
            raise ValueError("ACT action must contain 12 finite joint radians")
        if self.active:
            if time.monotonic() - self.armed_at > self.arm_max_duration:
                raise RuntimeError("Arm-only time budget reached; STOP required")
            if not self.topics.sources_fresh(max_age=0.30, max_spread=0.15):
                self.topics.wait_sources(timeout_s=0.25)
            for arm in ARM_KEYS:
                status = self.topics.status[arm] or {}
                if status.get("state") != "running" or status.get("session_id") != self.session_ids[arm]:
                    raise RuntimeError(f"Bridge no longer running as expected: {arm} {status}")
            actual = self.topics.measured_q12()
            delta = np.rad2deg(q - actual)
            max_gap = np.max(np.abs(delta))
            if max_gap > self.max_target_error_deg:
                idx = int(np.argmax(np.abs(delta)))
                raise RuntimeError(f"Native ACT target gap {max_gap:.2f} deg exceeds "
                                   f"{self.max_target_error_deg:.2f} deg guard "
                                   f"at FR3 #{idx//6+1} J{idx%6+1}")
            self.topics.target(q)
        if wait:
            self.topics.pump(max(0.0, (duration if duration is not None else self.dt)
                                 - (time.monotonic() - t0)))

    def stop_arms(self, timeout_s=5.0):
        if not self.active:
            return
        print("[FR3] Requesting STOP of both ServoJ sessions...")
        deadline = time.monotonic() + timeout_s
        confirmed = set()
        # Send STOP only once: some bridge versions can overwrite a final
        # 'stopped' report with a later 'stopping' report on repeated requests.
        self.topics.stop_both()
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self.topics, timeout_sec=0.02)
            for arm in ARM_KEYS:
                status = self.topics.status[arm] or {}
                if (status.get("session_id") == self.session_ids.get(arm)
                        and status.get("state") == "stopped"
                        and status.get("servo_end_ok") is True):
                    confirmed.add(arm)
                elif status.get("state") == "fault":
                    print(f"[FR3] {arm} bridge FAULT during STOP: {status}")
            if len(confirmed) == 2:
                self.active = False
                print("[FR3] STOP acknowledged on both bridges (ServoMoveEnd OK)")
                return
        raise RuntimeError(f"STOP not verified on both bridges; statuses={self.topics.status}. "
                           "Do NOT rearm. Verify physical stop with established procedure.")

    def close(self):
        if self.closed:
            return
        try:
            self.stop_arms()
        finally:
            self.closed = True
            self.topics.destroy_node()
            if self.owns_ros and rclpy.ok():
                rclpy.shutdown()


class RolloutNativeFR3Arms(RolloutAct):
    """Use the unmodified RMB RolloutAct.infer_policy()/temporal ensembling."""
    policy_name = "Act"
    MotionManagerClass = JointOnlyMotionManager

    def set_additional_args(self, parser):
        super().set_additional_args(parser)
        parser.add_argument("--execute", action="store_true",
                            help="Explicitly permit bounded hardware arm motion")
        parser.add_argument("--arm-max-duration", type=float, default=2.5,
                            help="Hard ServoJ active-time bound (seconds)")
        parser.add_argument("--max-target-error-deg", type=float, default=3.0,
                            help="Abort if any requested joint differs too much from measured")

    def setup_env(self, render_mode=None):
        if not (0 < self.args.arm_max_duration <= 5.0):
            raise ValueError("For initial experiments arm duration must be <=5s")
        if not (0.5 <= self.args.max_target_error_deg <= 3.0):
            raise ValueError("First-stage target-error guard cannot exceed 3 degrees")
        self.env = gym.make(ENV_ID, disable_env_checker=True,
                            execute=self.args.execute,
                            arm_max_duration=self.args.arm_max_duration,
                            max_target_error_deg=self.args.max_target_error_deg)
        self.entered_rollout = False

    def setup_model_meta_info(self):
        super().setup_model_meta_info()
        if (self.state_keys != ["measured_joint_pos"]
                or self.action_keys != ["command_joint_pos"]
                or self.state_dim != 12 or self.action_dim != 12
                or self.camera_names != ["front", "hand"]):
            raise ValueError("This first-stage launcher requires the OLD 12D checkpoint "
                             "(state=measured_joint_pos, action=command_joint_pos, cameras=front/hand). "
                             "Do NOT modify checkpoint metadata to bypass this.")
        if self.args.skip < 1:
            raise ValueError("Expected positive checkpoint skip")
        print(f"[NATIVE ACT] 12D joint-only policy, skip={self.args.skip}, "
              f"temporal ensembling={'OFF' if self.args.no_temp_ensem else 'ON'}")

    def set_command_data(self, action_keys=None):
        # Called by the ORIGINAL RMB RolloutPhase after its native ACT inference.
        super().set_command_data(action_keys=action_keys)
        if self.rollout_time_idx % self.args.skip == 0:
            q = self.motion_manager.get_command_data(DataKey.COMMAND_JOINT_POS)
            measured = self.env.unwrapped.topics.measured_q12()
            delta = np.rad2deg(q - measured)
            print(f"[NATIVE ENSEMBLE] base step={self.rollout_time_idx:03d} "
                  f"max_joint_gap={np.max(np.abs(delta)):.2f} deg, "
                  f"FR3#1={np.round(delta[:6], 2)}, FR3#2={np.round(delta[6:], 2)}")
        if not self.entered_rollout:
            self.entered_rollout = True
            target = self.motion_manager.get_command_data(DataKey.COMMAND_JOINT_POS)
            self.env.unwrapped.start_arms(target)

    def run(self):
        if not self.args.auto_exit:
            raise RuntimeError("Require --auto_exit; never run indefinite hardware experiments")
        if self.args.save_rollout:
            raise RuntimeError("Do not enable --save_rollout in this first-stage adapter")
        if self.args.max_duration > self.args.arm_max_duration:
            print("[FR3] Note: servo active hard bound is shorter than rollout phase duration")
        try:
            super().run()  # Unmodified original RMB rollout state machine
        finally:
            self.env.unwrapped.close()
            self.env.close()


def main():
    # Dynamic registration avoids edits to envs/__init__.py for this first stage.
    gym.register(id=ENV_ID, entry_point=RealDualFR3ArmOnlyNative)
    rollout = RolloutNativeFR3Arms()
    rollout.run()


if __name__ == "__main__":
    main()
