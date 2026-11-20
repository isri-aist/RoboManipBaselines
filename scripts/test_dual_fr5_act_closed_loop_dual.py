#!/usr/bin/env python3
"""Experimental dual-arm ACT -> ROS -> FAIRINO ServoJ.

Both existing fairino_ctrl nodes own their SDK connections. This process
reads live ROS observations and publishes 12 predicted joint targets in radians.

Default is PREVIEW ONLY. --execute and explicit terminal confirmation are
required to request real motion. This does NOT certify collision safety.
"""
# ============================================================================
# 1) DEPENDENCIES
# argparse: command-line flags. json: ServoJ status decoding.
# time: monotonic deadlines and latency. numpy: robot joint/image arrays.
# rclpy: ROS 2 pub/sub and callbacks. torch: ACT neural-network inference.
# IMPORTANT: This script is an *ACT client*, NOT the FAIRINO SDK controller.
# The existing fairino_ctrl and ServoJ bridge processes own the SDK connection.
# ============================================================================
# =============================================================================
# WARNING: the full-gripper execution path has previously caused ServoJ faults.
# --execute is NOT recommended until the controller/stop issue is resolved.
# For a motion-only baseline, use the separate *_MOTION_ONLY.py.
# =============================================================================
import argparse
import json
import sys
import time

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, String

# RMB training normalizes action vectors. At inference we must invert that
# normalization before interpreting values as joint radians or gripper 0/1.
# load_policy(), prepare_state(), and prepare_images() live in another file:
# test_dual_fr5_act.py. This script still depends on that file at runtime.
from robo_manip_baselines.common import denormalize_data
from test_dual_fr5_act import load_policy, prepare_state, prepare_images

# ============================================================================
# 2) ROS TOPIC MAPPING
# Bridge topics are separated per FR5. Do NOT swap FR1/FR2 ordering: the
# policy's first six joints belong to FR1, next six belong to FR2.
# SOURCES contains the four *continuously updated* streams that need fresh
# data: two arm joint arrays and two RGB images. Gripper states are tracked
# separately because their ROS update cadence differs.
# ============================================================================
PREFIXES = {'fr1': '/fairino_1/act_servoj', 'fr2': '/fairino_2/act_servoj'}
SOURCES = {
    'fr1': '/fairino_1/joint_states',
    'fr2': '/fairino_2/joint_states',
    'front': '/camera_d405_1/color/image_rect_raw',
    'hand': '/camera_d405_2/color/image_rect_raw',
}


# ============================================================================
# 3) INPUT VALIDATION / IMAGE DECODING
# JointState messages may use arbitrary name ordering, so reorder explicitly
# to [joint_1,...,joint_6] rather than trusting msg.position order.
# Inputs stay in RADIANS for ACT and the ROS ServoJ bridge.
# ============================================================================
def joint_rad(msg):
    names = [f'joint_{n}' for n in range(1, 7)]
    if len(msg.name) != len(set(msg.name)) or len(msg.position) != len(msg.name):
        raise ValueError('Duplicate or mismatched joint names/positions')
    # Map each incoming joint name to its measured position before sorting.
    mapping = dict(zip(msg.name, msg.position))
    if not all(name in mapping for name in names):
        raise ValueError(f'Joint names must include {names}')
    q = np.asarray([mapping[name] for name in names], dtype=np.float64)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError('Invalid joint positions')
    return q


# Converts a ROS sensor_msgs/Image byte buffer into a H x W x 3 RGB array.
# The step field may include row padding, so we trim it correctly. If the
# publisher sends BGR, reverse channels to obtain RGB for ACT.
def image_rgb(msg):
    if msg.encoding not in ('rgb8', 'bgr8'):
        raise ValueError(f'Expected rgb8/bgr8, got {msg.encoding!r}')
    if msg.width <= 0 or msg.height <= 0 or msg.step < msg.width * 3:
        raise ValueError('Invalid image dimensions/stride')
    # frombuffer avoids copying the raw ROS message before reshaping.
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    expected = msg.height * msg.step
    if raw.size != expected:
        raise ValueError(f'Image byte count {raw.size} != {expected}')
    rgb = raw.reshape(msg.height, msg.step)[:, :msg.width * 3]
    rgb = rgb.reshape(msg.height, msg.width, 3)
    if msg.encoding == 'bgr8':
        return rgb[:, :, ::-1].copy()
    return rgb.copy()


# ============================================================================
# 4) ROS NODE: OBSERVATIONS + COMMAND INTERFACES
# One ROS node subscribes to joint feedback/cameras/gripper feedback and
# publishes requested arm/gripper commands through the existing controllers.
# It does not instantiate a FAIRINO robot RPC client.
# ============================================================================
class Observations(Node):
    def __init__(self):
        super().__init__('act_dual_fr5_live_client')
        # Latest per-source data. arrivals stores monotonic *arrival times*, not
        # camera hardware timestamps. seq counts received messages per source;
        # wait_observation(since=...) can therefore require a truly new sample.
        self.data = {}
        self.arrivals = {}
        self.seq = {k: 0 for k in SOURCES}
        self.statuses = {'fr1': None, 'fr2': None}
        self.status_at = {'fr1': 0.0, 'fr2': 0.0}
        # BEST_EFFORT + depth 1 minimizes queued old sensor readings (common for
        # high-frequency cameras and joint streams). It can drop packets if busy.
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.subscribers = []
        # Register both six-joint state subscriptions; lambda k=key binds each
        # arm name now (otherwise late binding would point both to FR2).
        for key in ('fr1', 'fr2'):
            self.subscribers.append(self.create_subscription(
                JointState, SOURCES[key], lambda msg, k=key: self._joint(k, msg), qos))
        # Register the front and hand RGB camera streams independently.
        for key in ('front', 'hand'):
            self.subscribers.append(self.create_subscription(
                Image, SOURCES[key], lambda msg, k=key: self._image(k, msg), qos))
        # ServoJ bridge interface per arm: /target contains JointState positions
        # (radians), /arm begins the bounded session, /stop requests termination.
        # /status is retained by ROS (TRANSIENT_LOCAL), so it can sometimes be stale.
        self.target_pubs = {}
        self.arm_pubs = {}
        self.stop_pubs = {}
        self.status_subs = []
        for arm, prefix in PREFIXES.items():
            self.target_pubs[arm] = self.create_publisher(JointState, prefix + '/target', 10)
            self.arm_pubs[arm] = self.create_publisher(Bool, prefix + '/arm', 10)
            self.stop_pubs[arm] = self.create_publisher(Bool, prefix + '/stop', 10)
            self.status_subs.append(self.create_subscription(
                String, prefix + '/status',
                lambda msg, arm=arm: self._status(arm, msg),
                QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL)))
        # Gripper Bool command interface (True=OPEN, False=CLOSE). Controllers
        # translate these topics into digital-output SDK commands (SetDO/SetD0).
        # WARNING: in the FULL version, these commands previously caused faults
        # while ServoJ was running. Do not treat this as a validated live path.
        self.gripper_cmd_pubs = {
            "fr1": self.create_publisher(
                Bool, "/fairino_1/gripper_command", 10
            ),
            "fr2": self.create_publisher(
                Bool, "/fairino_2/gripper_command", 10
            )
        }

        # Latest gripper reports. The initial value None means no report received.
        # In this setup, these generally reflect *commanded* state, not verified
        # physical jaw contact/position.
        self.gripper_states = {
            "fr1": None,
            "fr2": None,
        }
        # Arrival time of last gripper-state report (not physical position verification).
        self.gripper_state_at = {"fr1": 0.0, "fr2": 0.0}

        self.gripper_state_subs = []

        for arm, topic in (
            ("fr1", "/fairino_1/gripper_state"),
            ("fr2", "/fairino_2/gripper_state"),
        ):
            self.gripper_state_subs.append(
                self.create_subscription(
                    Bool,
                    topic,
                    lambda msg, a=arm: self._gripper_state(a, msg),
                    10,
                )
            )

    # Feedback handler: cache OPEN=1.0 or CLOSE=0.0 for 14D ACT observations.
    def _gripper_state(self, arm, msg):
        self.gripper_states[arm] = float(msg.data)
        self.gripper_state_at[arm] = time.monotonic()

    # Publish a command only; ROS publish() does not prove hardware actuation.
    # The receiving fairino_ctrl node makes the actual SDK SetDO call.
    def send_gripper_command(self, arm, open_flag):

        msg = Bool()
        msg.data = bool(open_flag)

        self.gripper_cmd_pubs[arm].publish(msg)

    # On every joint-state message, validate and cache the six angles plus a
    # monotonic arrival timestamp. Invalid feedback is logged and not accepted.
    def _joint(self, key, msg):
        try:
            self.data[key] = joint_rad(msg)
            self.arrivals[key] = time.monotonic()
            self.seq[key] += 1
        except Exception as exc:
            self.get_logger().error(f'{key} invalid joint feedback: {exc}')

    # On every RGB image message, decode and cache the array and arrival time.
    def _image(self, key, msg):
        try:
            self.data[key] = image_rgb(msg)
            self.arrivals[key] = time.monotonic()
            self.seq[key] += 1
        except Exception as exc:
            self.get_logger().error(f'{key} camera frame invalid: {exc}')

    # Decode the bridge's JSON status string. This status is informational and
    # may be retained/stale; it is NOT a safety-rated stop confirmation.
    def _status(self, arm, msg):
        try:
            self.statuses[arm] = json.loads(msg.data)
            self.status_at[arm] = time.monotonic()
            print(f'[SERVO STATUS {arm}]', msg.data, flush=True)
        except (ValueError, TypeError):
            self.get_logger().error(f'Malformed {arm} bridge status: {msg.data}')

    # ============================================================================
    # 5) BUILD ONE SYNCHRONIZED OBSERVATION
    # Reject snapshots unless all four live sources exist, have fresh messages,
    # arrived within max_age=0.30 s, and have arrival spread <=0.12 s.
    # Then combine 12 measured joint angles and 2 gripper states into 14D.
    # The separate image dict contains both RGB frames for the visual encoder.
    # Note: this uses ROS receipt times, not hardware timestamp synchronization.
    # ============================================================================
    def snapshot(self, since=None, max_age=0.30, max_spread=0.12):
        if len(self.data) != len(SOURCES):
            return None
        # Require each live stream to advance beyond the previous snapshot.
        if since is not None and any(self.seq[k] <= since[k] for k in SOURCES):
            return None
        # For diagnostic and deadline logic, use monotonic time so system clock
        # changes do not move deadlines backward. (Other uses follow below.)
        now = time.monotonic()
        times = list(self.arrivals.values())
        if len(times) != len(SOURCES):
            return None
        if max(now - t for t in times) > max_age:
            return None
        # Reject if the four latest arrivals are too far apart in receive time.
        if max(times) - min(times) > max_spread:
            return None
        # Initial gripper reports are required even for PREVIEW: the model expects
        # 14 inputs. Periodic state publication avoids startup None values.
        if any(v is None for v in self.gripper_states.values()):
            return None
        # q order is [FR1 J1..J6, FR2 J1..J6], all radians.
        q = np.concatenate((self.data['fr1'], self.data['fr2']))
        obs = {
            'joint_pos': q.copy(),
            'gripper_state': np.array([
                self.gripper_states['fr1'],
                self.gripper_states['fr2'],
            ], dtype=np.float64),
        }
        info = {'rgb_images': {
            'front': self.data['front'].copy(),
            'hand': self.data['hand'].copy(),
        }}
        return obs, info, dict(self.seq)

    # Spin ROS callbacks until a valid snapshot exists, or report which of the
    # joint/camera/gripper streams prevented a new synchronized observation.
    def wait_observation(self, since=None, timeout=6.0, hold_target=None, publish_hz=15.0):
        deadline = time.monotonic() + timeout
        next_hold = time.monotonic()
        while rclpy.ok() and time.monotonic() < deadline:
            # When a servo session is active, hold the last safe target while
            # collecting fresh camera/joint data. The bridge watchdog is 0.35 s.
            if hold_target is not None and time.monotonic() >= next_hold:
                self.send_targets(hold_target)
                next_hold = time.monotonic() + 1.0 / publish_hz
            rclpy.spin_once(self, timeout_sec=0.015)
            snap = self.snapshot(since=since)
            if snap is not None:
                return snap
        now = time.monotonic()

        print("\n========== OBSERVATION DIAGNOSTICS ==========")

        print("Received sources:", list(self.data.keys()))

        print("Missing sources:", [
            k for k in SOURCES if k not in self.data
        ])

        print("Source ages:", {
            k: round(now - self.arrivals[k], 3)
            if k in self.arrivals else None
            for k in SOURCES
        })

        if len(self.arrivals) == len(SOURCES):
            times = [self.arrivals[k] for k in SOURCES]
            print("Timestamp spread:", round(max(times) - min(times), 3))

        print("Gripper states:", self.gripper_states)
        print("============================================")

        raise RuntimeError(
            "No new, suitably synchronized camera + joint observations"
        )

    # Send the 12 *joint* components to the two ServoJ bridges. Each receives
    # six radians via JointState; the bridge converts them to degrees and drives
    # ServoJ through its existing safety checks and interpolation.
    def send_targets(self, q12_rad):
        q = np.asarray(q12_rad, dtype=np.float64)
        if q.shape != (12,) or not np.isfinite(q).all():
            raise ValueError('Expected 12 finite joint positions in radians')
        for arm, q6 in (('fr1', q[:6]), ('fr2', q[6:12])):
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = [f'joint_{i}' for i in range(1, 7)]
            msg.position = q6.tolist()  # ROS values are RADIANS
            self.target_pubs[arm].publish(msg)

    # Request both ServoJ bridge sessions to start. A Bool publish is a request;
    # the main loop separately waits for each bridge's acknowledgement.
    def arm_both(self):
        msg = Bool(); msg.data = True
        for pub in self.arm_pubs.values():
            pub.publish(msg)

    # Publish STOP several times to both bridges. IMPORTANT LIMITATION:
    # repeated STOP publications do not themselves certify ServoMoveEnd success.
    # This limitation matters because FR1 previously remained 'running'.
    def stop_both(self):
        msg = Bool(); msg.data = True
        for _ in range(5):
            for pub in self.stop_pubs.values():
                pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.04)


# ============================================================================
# SESSION-AWARE STOP -> GRIPPER -> VERIFIED MANUAL/AUTOMATIC RESUME
# Requires the updated bridge's session_id and servo_end_ok status fields.
# ============================================================================
def request_verified_stop(node, session_ids, timeout_s=5.0):
    sent_at = time.monotonic()
    msg = Bool()
    msg.data = True
    for pub in node.stop_pubs.values():
        pub.publish(msg)
    confirmed = set()
    deadline = sent_at + timeout_s
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.02)
        for arm in ('fr1', 'fr2'):
            if arm in confirmed or node.status_at[arm] <= sent_at:
                continue
            status = node.statuses[arm] or {}
            if status.get('session_id') != session_ids.get(arm):
                continue
            if status.get('state') in ('fault', 'disabled', 'busy'):
                raise RuntimeError(f'{arm} bridge fault during STOP: {status}')
            if status.get('state') == 'stopped':
                if status.get('servo_end_ok') is not True:
                    raise RuntimeError(f'{arm} ServoMoveEnd not confirmed: {status}')
                confirmed.add(arm)
        if len(confirmed) == 2:
            print('Both ServoMoveEnd calls acknowledged successfully.')
            return
    raise RuntimeError(f'STOP NOT CONFIRMED on both robots: {node.statuses}')


def command_grippers_when_stopped(node, requested, changed, timeout_s=3.0):
    sent_at = time.monotonic()
    for arm in changed:
        print(f'[GRIPPER] {arm}: {"OPEN" if requested[arm] else "CLOSE"}')
        node.send_gripper_command(arm, requested[arm])
    deadline = time.monotonic() + timeout_s
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.02)
        if all(node.gripper_state_at[a] > sent_at and
               node.gripper_states[a] is not None and
               bool(node.gripper_states[a]) == requested[a] for a in changed):
            print('New ROS commanded-state reports received; verify physical grippers.')
            return
    raise RuntimeError(f'Gripper ROS commanded-state timeout: {node.gripper_states}')


def verify_automatic_gripper_settle(node, requested, changed, session_ids, settle_s):
    """Supervised automatic mode: wait while both servo sessions remain stopped.

    This is a timed settling and ROS health check, NOT proof of physical
    gripper completion. The existing /gripper_state topics are commanded-state
    reports, not independently measured gripper positions.
    """
    arms = ('fr1', 'fr2')
    initial_q = {a: node.data[a].copy() for a in arms}
    deadline = time.monotonic() + settle_s

    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.02)
        now = time.monotonic()
        for arm in arms:
            status = node.statuses[arm] or {}
            if (status.get('state') != 'stopped'
                    or status.get('session_id') != session_ids[arm]
                    or status.get('servo_end_ok') is not True):
                raise RuntimeError(f'{arm}: lost confirmed STOP during gripper settle: {status}')
            if now - node.arrivals.get(arm, 0.0) > 0.30:
                raise RuntimeError(f'{arm}: joint feedback stale during gripper settle')
            moved_deg = np.max(np.abs(np.rad2deg(node.data[arm] - initial_q[arm])))
            if moved_deg > 1.0:
                raise RuntimeError(f'{arm}: moved {moved_deg:.2f} deg during servo pause')
        for arm in changed:
            if (node.gripper_states[arm] is None
                    or bool(node.gripper_states[arm]) != requested[arm]):
                raise RuntimeError(f'{arm}: commanded gripper state no longer matches')

    if not rclpy.ok():
        raise RuntimeError('ROS shutdown while waiting for gripper settle')
    print(f'[GRIPPER] {settle_s:.2f}s settling interval elapsed; physical jaw position unverified')


def resume_servoj_from_measured_pose(node, session_ids, period, timeout_s=3.0):
    """Re-arm both bridges, streaming a stationary measured-pose target.

    session_ids is mutated BEFORE ARM so cleanup has intended new IDs even
    if only one bridge accepts the ARM command.
    """
    for arm in ('fr1', 'fr2'):
        st = node.statuses[arm] or {}
        if (st.get('state') != 'stopped'
                or st.get('servo_end_ok') is not True
                or st.get('session_id') != session_ids[arm]):
            raise RuntimeError(f'{arm} is not verified stopped: {st}')

    # The bridge publishes 'stopped' just before its worker thread returns.
    # Allow time for the old worker to exit; bridge will reject overlapping ARM.
    until = time.monotonic() + 0.30
    while rclpy.ok() and time.monotonic() < until:
        rclpy.spin_once(node, timeout_sec=0.02)

    # Require *new* ROS joint feedback from both arms after the gripper operation.
    before = {a: node.seq[a] for a in ('fr1', 'fr2')}
    feedback_deadline = time.monotonic() + 2.0
    while rclpy.ok() and time.monotonic() < feedback_deadline:
        rclpy.spin_once(node, timeout_sec=0.015)
        if all(node.seq[a] > before[a] for a in ('fr1', 'fr2')):
            break
    else:
        raise RuntimeError('No fresh FR1+FR2 joint feedback after gripper action')

    now = time.monotonic()
    if any(now - node.arrivals.get(a, 0.0) > 0.30 for a in ('fr1', 'fr2')):
        raise RuntimeError('Joint feedback stale; refusing ServoJ restart')
    hold = np.concatenate((node.data['fr1'], node.data['fr2'])).copy()
    if hold.shape != (12,) or not np.all(np.isfinite(hold)):
        raise RuntimeError('Invalid measured joint positions for ServoJ restart')

    new_ids = {a: session_ids[a] + 1 for a in ('fr1', 'fr2')}
    session_ids.update(new_ids)  # cleanup tracks any partial ARM
    requested_at = time.monotonic()
    node.arm_both()

    # Republish fresh measured pose until BOTH workers report ServoMoveStart.
    # This avoids expiring the bridge's initial-target and target watchdogs.
    deadline = requested_at + timeout_s
    next_hold = 0.0
    while rclpy.ok() and time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_hold:
            node.send_targets(hold)
            next_hold = now + period
        rclpy.spin_once(node, timeout_sec=0.01)
        for arm in ('fr1', 'fr2'):
            if node.status_at[arm] <= requested_at:
                continue
            st = node.statuses[arm] or {}
            # A busy/fault can be issued under the previous ID if ARM is refused.
            if st.get('state') in ('fault', 'busy', 'disabled'):
                raise RuntimeError(f'{arm} rejected ServoJ restart: {st}')
            if (st.get('session_id') == new_ids[arm]
                    and st.get('state') in ('stopping', 'stopped')):
                raise RuntimeError(f'{arm} stopped during restart: {st}')
        if all(node.status_at[a] > requested_at
               and (node.statuses[a] or {}).get('session_id') == new_ids[a]
               and (node.statuses[a] or {}).get('state') == 'running'
               for a in ('fr1', 'fr2')):
            print(f'RESUMED: both ServoMoveStart acknowledged; sessions={new_ids}')
            return hold
    raise RuntimeError(f'RESUME NOT CONFIRMED: {node.statuses}')


# ============================================================================
# 6) ACT INFERENCE: 14D STATE + TWO RGB CAMERAS -> 25 x 14 ACTION CHUNK
# prepare_state(): [12 measured joint radians, FR1/FR2 gripper OPEN(1)/CLOSE(0)]
# plus normalization using checkpoint metadata.
# prepare_images(): stack front/hand images as model-ready tensors.
# policy(): predict a whole future action chunk, not only one action.
# denormalize_data(): convert neural-network outputs to meaningful units.
# ============================================================================
def inference(policy, meta, device, obs, info):
    q_t, measured = prepare_state(obs, meta, device)
    rgb_t = prepare_images(info, meta, device)
    # Disable gradient recording: inference only, reducing GPU memory/overhead.
    with torch.inference_mode():
        chunk_t = policy(q_t, rgb_t)
    chunk = chunk_t[0].detach().cpu().numpy()
    if chunk.ndim != 2 or chunk.shape[1] != 14:
        raise RuntimeError(f'Unexpected ACT output shape {chunk.shape}')
    # Each action row has [FR1 joints(6), FR2 joints(6), FR1 gripper, FR2 gripper].
    # The gripper outputs are continuous regression values, not probabilities.
    result = np.asarray([denormalize_data(a, meta['action']) for a in chunk], dtype=np.float64)
    if result.shape != chunk.shape or not np.isfinite(result).all():
        raise RuntimeError('ACT prediction contained NaN/Inf')
    return result, measured


# ============================================================================
# 7) PROGRAM ENTRY POINT AND USER SAFETY GATES
# Default: PREVIEW ONLY; absolutely no targets are sent unless --execute is
# supplied AND the operator types the confirmation phrase.
# --steps is the maximum number of waypoints; --hz controls repeat-publish
# cadence (NOT waypoint progression speed); --max-duration caps session time.
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--execute', action='store_true', help='Enable one timed real-hardware session')
    ap.add_argument('--steps', type=int, default=90)
    ap.add_argument('--hz', type=float, default=15.0)
    ap.add_argument('--max-duration', type=float, default=9.0)
    ap.add_argument('--allow-resume', action='store_true',
                    help='Permit resume after each gripper transition (manual confirmation by default)')
    ap.add_argument('--auto-resume', action='store_true',
                    help='Supervised automatic resume; requires --allow-resume')
    ap.add_argument('--gripper-settle-s', type=float, default=2.0,
                    help='Wait after reported gripper command (not physical completion verification)')
    ap.add_argument('--max-gripper-transitions', type=int, default=2,
                    help='Maximum gripper transitions permitted per automatic run')
    ap.add_argument('--max-wall-duration', type=float, default=40.0,
                    help='Whole-run wall time cap including all gripper pauses')
    args = ap.parse_args()
    if args.auto_resume and not args.allow_resume:
        ap.error('--auto-resume requires --allow-resume')
    if not (0.5 <= args.gripper_settle_s <= 5.0):
        ap.error('Require 0.5 <= --gripper-settle-s <= 5.0; validate against hardware')
    if not (1 <= args.max_gripper_transitions <= 5):
        ap.error('Require 1 <= --max-gripper-transitions <= 5')
    if not (15.0 <= args.max_wall_duration <= 120.0):
        ap.error('Require 15 <= --max-wall-duration <= 120 seconds')
    if args.steps < 1 or not (1.0 <= args.hz <= 15.0) or not (0 < args.max_duration <= 35.0):
        ap.error('Require steps>=1, 1<=hz<=15, 0<max-duration<=35s')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable (use --device cpu)')
    device = torch.device(args.device)
    # Load checkpoint + its adjacent model_meta_info.pkl; helper validates
    # 14D state/action keys and loads weights onto CPU/GPU.
    policy, meta = load_policy(args.checkpoint, device)
    skip = int(meta['data']['skip'])
    chunk_len = int(meta['data']['chunk_size'])
    if skip < 1 or skip > chunk_len or chunk_len < 1:
        raise RuntimeError(f'Unexpected ACT skip/chunk size: {skip}/{chunk_len}')
    # metadata skip is validated above but NOT used to skip actions here.
    # The current rollout consumes all 25 predicted waypoints sequentially.
    print(f'ACT predicts {chunk_len} actions; this test uses all actions in each chunk')
    rclpy.init()
    node = Observations()
    armed = False
    servo_ended = False
    session_ids = {}
    try:
        print('\nWaiting for real cameras and BOTH FR5 joint-state topics...', flush=True)
        # Wait for a valid live snapshot BEFORE asking ACT to predict motion.
        obs, info, seq = node.wait_observation(timeout=12.0)
        # Compute model output once with real cameras/joints/gripper values, and
        # show FR1/FR2 differences for operator inspection.
        preview_chunk, measured = inference(policy, meta, device, obs, info)
        print('\n=== LIVE ACT PREVIEW: BOTH FR5 ARMS ===')
        for arm, seg in (('FR5 #1', slice(0, 6)), ('FR5 #2', slice(6, 12))):
            q_now = np.rad2deg(measured[seg])
            q_proposed = np.rad2deg(preview_chunk[0, seg])
            print(f'{arm} measured:', np.round(q_now, 3))
            print(f'{arm} ACT:     ', np.round(q_proposed, 3))
            print(f'{arm} delta:   ', np.round(q_proposed-q_now, 3))
        print('ACT chunk:', preview_chunk.shape)
        print('Bridge statuses:', node.statuses)
        print("\n=== ACT GRIPPER PREDICTIONS ===")

        # Print the two gripper outputs for all 25 actions. Typical OPEN output is
        # around 1.0 (can be >1), CLOSE around 0.0 (can be negative).
        for i, action in enumerate(preview_chunk):
            g1, g2 = action[12:14]

            print(
                f"Action {i + 1:02d}: "
                f"FR1={g1:.3f}, FR2={g2:.3f}"
            )
        # CRITICAL GATE: return without arming or publishing targets in PREVIEW.
        if not args.execute:
            print('\nPREVIEW ONLY. No servo targets or arm commands sent.')
            return
        # Refuse to arm unless both bridges currently report IDLE. Because status
        # uses TRANSIENT_LOCAL, verify actual stop/session state independently.
        if not all(isinstance(node.statuses[a], dict)
                   and node.statuses[a].get('state') == 'idle'
                   and isinstance(node.statuses[a].get('session_id'), int)
                   for a in ('fr1', 'fr2')):
            raise RuntimeError(f'Both bridges must be IDLE and session-aware: {node.statuses}')
        session_ids = {a: node.statuses[a]['session_id'] + 1 for a in ('fr1', 'fr2')}
        print('\nWARNING: This is experimental physical ACT motion. No collision checking,')
        print('no gripper command, and the SDK may block on socket retries.')
        print('Verify swept-path clearance, active independent physical E-stop,')
        print('no existing controller fault, and no competing robot motion source.')
        # Second human-in-the-loop gate: physically inspect clearance, E-stop,
        # configuration and first targets before allowing movement.
        confirm = input('Type START to arm this bounded session: ').strip()
        if confirm != 'START':
            print('Cancelled without motion.')
            return
        obs, info, seq = node.wait_observation(since=seq, timeout=4.0)
        # Re-sample observations immediately before arming: reject if the first
        # joint target changed by more than 1 degree since preview.
        fresh_chunk, fresh_q = inference(policy, meta, device, obs, info)
        delta_preview = np.rad2deg(
            fresh_chunk[0, :12] - preview_chunk[0, :12]
        )
        if np.max(np.abs(delta_preview)) > 1.0:
            raise RuntimeError('ACT first target changed >1 deg since preview; rerun preview')
        print(
            'Fresh ACT joint targets (deg):',
            np.round(np.rad2deg(fresh_chunk[0, :12]), 3)
        )

        print(
            'Fresh ACT gripper predictions:',
            np.round(fresh_chunk[0, 12:14], 3)
        )
        # Ensure that if ARM publication partially succeeds we request STOP for both.
        # Set BEFORE publishing ARM, so finally tries to send STOP even if only one
        # arm accepts the request or the acknowledgement phase fails.
        armed = True
        arm_requested_at = time.monotonic()
        node.arm_both()
        # Wait (up to 1.5 s) for both bridge sessions to report armed/running.
        # Refuse to proceed on a fault, disabled bridge, or missing acknowledgement.
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
            states = {a: (node.statuses[a] or {}).get('state') for a in ('fr1', 'fr2')}
            if all(node.status_at[a] > arm_requested_at
                   and (node.statuses[a] or {}).get('session_id') == session_ids[a]
                   and states[a] in ('armed', 'running') for a in ('fr1', 'fr2')):
                break
            if any(node.status_at[a] > arm_requested_at
                   and states[a] in ('fault', 'disabled', 'busy', 'stopping', 'stopped')
                   for a in ('fr1', 'fr2')):
                raise RuntimeError(f'A bridge refused to arm: {node.statuses}')
        else:
            raise RuntimeError(f'No ARM acknowledgment from both bridges: {node.statuses}')
        # ============================================================================
        # 8) WAYPOINT PACING (NOT FIXED-RATE TRAJECTORY PLAYBACK)
        # Each predicted joint target is held until BOTH arms are within 0.75 deg.
        # The same target is republished at --hz to keep bridge watchdogs satisfied.
        # This changes the ACT chunk's temporal spacing relative to demonstrations;
        # --hz is the command republish frequency, NOT the ACT step rate.
        # ============================================================================
        period = 1.0 / args.hz
        # Both arms must reach each ACT waypoint before advancing.
        # Maximum joint error allowed for waypoint completion on either arm.
        REACH_TOL_DEG = 0.75
        # Abort a waypoint instead of silently skipping it if the arms cannot reach.
        WAYPOINT_TIMEOUT_S = 3.0
        started = time.monotonic()
        paused_duration = 0.0  # Manual/gripper pauses do not consume active-motion budget.
        hold_target = fresh_chunk[0, :12].copy()
        # Begin with the freshly preview-checked predicted chunk.
        action_chunk = fresh_chunk

        if any(node.gripper_states[a] is None for a in ("fr1", "fr2")):
            raise RuntimeError("Initial gripper states unavailable")

        # Track previous gripper choice across waypoints: do not repeatedly send a
        # CLOSE/OPEN command when predictions continue to request the same state.
        gripper_requested = {
            arm: bool(node.gripper_states[arm])
            for arm in ("fr1", "fr2")
        }

        gripper_sent = gripper_requested.copy()
        gripper_transition_count = 0

        # ============================================================================
        # 9) MAIN CLOSED-LOOP ACT EXECUTION
        # Consume each model action in order. After all 25 have been used, obtain
        # fresh joint+camera+gripper inputs and infer the next 25-action chunk.
        # ============================================================================
        for step in range(args.steps):
            if args.auto_resume and time.monotonic() - started >= args.max_wall_duration:
                raise RuntimeError('Automatic rollout wall-clock budget exceeded')
            if time.monotonic() - started - paused_duration >= args.max_duration:
                print('ACT session duration limit reached')
                break
            # The first action was explicitly previewed; after this, reobserve.
            # First target comes from the freshly checked observation; subsequent
            # targets use a new validated ROS observation after the previous waypoint.
            if step == 0:
                current_obs, current_info, current_seq = obs, info, seq
            else:
                current_obs, current_info, current_seq = node.wait_observation(
                    since=seq, timeout=0.30, hold_target=hold_target,
                    publish_hz=args.hz)
                seq = current_seq
            # At 25/50/75... steps, re-infer from the newest observation.
            # This is chunked receding-horizon prediction, not temporal ensembling.
            if step > 0 and step % chunk_len == 0:
                t0 = time.perf_counter()
                action_chunk, _ = inference(policy, meta, device, current_obs, current_info)
                latency = (time.perf_counter() - t0) * 1000
                print(f'ACT inference step={step}: {latency:.1f} ms')
            index = step % chunk_len
            action = action_chunk[index]

            # Split the 14D action: six FR1 radians, six FR2 radians, and two
            # continuous gripper predictions. Never pass all 14 to send_targets().
            arm_action = action[:12]
            gripper_action = action[12:14]

            # Update requested gripper states from ACT predictions.

            # Gripper hysteresis: >=0.75 selects OPEN; <=0.25 selects CLOSE;
            # between thresholds, keep the previous choice to limit oscillation.
            for arm, prediction in zip(
                ("fr1", "fr2"), gripper_action
            ):

                if prediction >= 0.75:
                    gripper_requested[arm] = True  # OPEN

                elif prediction <= 0.25:
                    gripper_requested[arm] = False  # CLOSE

            # Convert *only* the 12 joint positions to degrees for diagnostics and
            # displacement safety validation. Gripper values are not angles.
            q_current = np.rad2deg(np.asarray(current_obs['joint_pos']))
            q_target = np.rad2deg(arm_action)
            delta = q_target - q_current
            print(f'step={step+1:03d} FR1={np.round(delta[:6], 2)} FR2={np.round(delta[6:], 2)}', flush=True)
            max_error = float(np.max(np.abs(delta)))
            worst_joint = int(np.argmax(np.abs(delta)))
    
            robot = 1 if worst_joint < 6 else 2
            joint = (worst_joint % 6) + 1
            # Independent software guard: a model target >12 degrees from the last
            # received joint snapshot is rejected BEFORE publishing that target.
            # This is not inter-arm/self-collision checking.
            if max_error > 12.0:
                print("\n==========ACT diagnostic===========")
                print(f"ACT step: {step + 1}")
                print(f"Chunk action: {index + 1}/{chunk_len}")
    
                print(f"Robot: FR5#{robot}")
                print(f"Joint: J{joint}")
    
                print(f"Measured joint: {q_current[worst_joint]:.3f} deg")
                print(f"ACT target: : {q_target[worst_joint]:.3f} deg")
                print(f"Difference: {delta[worst_joint]:+.3f} deg")
    
                print(f"fr1 diff", np.round(delta[:6], 2))
                print(f"fr2 diff", np.round(delta[6:], 2))
                print("\n=========================")
    
                raise RuntimeError(
                    "ACT target exceeds validated displacement limit"
                )

            # =====================================================
            # SYNCHRONIZED DUAL-ARM ACT WAYPOINT EXECUTION
            # =====================================================

            # ============================================================================
            # 10) EXECUTE ONE JOINT WAYPOINT
            # Send the current target to BOTH bridges. Bridges handle the 125 Hz
            # underlying SDK ServoJ commands; this Python script is a ROS client.
            # ============================================================================
            node.send_targets(arm_action)
            hold_target = arm_action.copy()

            # Start a per-waypoint 3 s deadline; keep separate overall session deadline.
            waypoint_start = time.monotonic()
            last_publish = waypoint_start
            session_timed_out = False
            pause_test_completed = False

            while rclpy.ok():

                # Process feedback from BOTH robots.
                # Pump ROS callbacks: refresh joint feedback, camera frames, gripper state,
                # and status while the script is waiting for waypoint completion.
                rclpy.spin_once(node, timeout_sec=0.01)

                now = time.monotonic()
                if args.auto_resume and now - started >= args.max_wall_duration:
                    raise RuntimeError('Automatic rollout wall-clock budget exceeded')

                # Preserve the session duration limit.
                # Stop the session if total allowed wall time has elapsed.
                if now - started - paused_duration >= args.max_duration:
                    print("ACT session duration reached")
                    session_timed_out = True
                    break

                # Stop if either ServoJ controller reports a problem.
                # Abort on a bridge fault/stop. The status is a message from the bridge;
                # check actual hardware if status was retained or is no longer updating.
                for arm in ("fr1", "fr2"):
                    status = node.statuses[arm] or {}

                    if status.get("state") in (
                        "fault", "disabled", "stopping", "stopped"
                    ):
                        raise RuntimeError(
                            f"{arm} ServoJ stopped: {status}"
                        )

                # Require recent feedback from BOTH arms.
                # The second per-arm loop checks last actual JOINT MESSAGE receipt time.
                # The 0.30 s watchdog catches stalled ROS feedback. Keep it enabled.
                for arm in ("fr1", "fr2"):
                    age = now - node.arrivals.get(arm, 0.0)

                    # If feedback is late, log both arms' last received statuses before aborting;
                    # do NOT increase this threshold merely to mask controller/SDK faults.
                    if age > 0.30:

                        now_debug = time.monotonic()

                        print("\n========== FEEDBACK FAILURE ==========")

                        for a in ("fr1", "fr2"):

                            feedback_age = (
                                now_debug - node.arrivals.get(a, 0.0)
                            )

                            print(
                                f"{a}: "
                                f"age={feedback_age:.3f}s, "
                                f"seq={node.seq[a]}, "
                                f"servo_status={node.statuses[a]}, "
                                f"gripper={node.gripper_states[a]}"
                            )

                        print("======================================\n")

                        raise RuntimeError(
                            f"{arm} joint feedback stale: {age:.3f}s"
                        )

                # Current measured joint positions, in degrees.
                measured = np.rad2deg(
                    np.concatenate((
                        node.data["fr1"],
                        node.data["fr2"],
                    ))
                )

                # Compute signed joint errors in degrees for both robots.
                error = q_target - measured

                fr1_error = np.max(np.abs(error[:6]))
                fr2_error = np.max(np.abs(error[6:]))

                # Only advance when BOTH robots reached the waypoint.
                # The largest joint error on FR1 AND on FR2 must each be <=0.75 degrees.
                arms_reached = (
                    fr1_error <= REACH_TOL_DEG
                    and fr2_error <= REACH_TOL_DEG
                )

                # IMPORTANT: both arms at the current waypoint; gripper commands (if enabled)
                # are evaluated only now, not concurrently with moving toward this target.
                if arms_reached:
                    changed = [a for a in ('fr1', 'fr2')
                               if gripper_sent[a] != gripper_requested[a]]
                    if changed:
                        gripper_transition_count += 1
                        if args.auto_resume and gripper_transition_count > args.max_gripper_transitions:
                            raise RuntimeError('Automatic gripper-transition limit exceeded')
                        print(f'ACT waypoint {step + 1}: gripper transition {changed}')
                        pause_started = time.monotonic()
                        request_verified_stop(node, session_ids, timeout_s=5.0)
                        servo_ended = True
                        command_grippers_when_stopped(
                            node, gripper_requested, changed, timeout_s=3.0)
                        for arm in changed:
                            gripper_sent[arm] = gripper_requested[arm]

                        if not args.allow_resume:
                            print('PAUSE-ONLY COMPLETE: both ServoJ sessions ended.')
                            print('Inspect grippers physically. No automatic re-ARM.')
                            pause_test_completed = True
                            break

                        if args.auto_resume:
                            print('[AUTO] ServoJ ended on both; waiting for gripper to settle')
                            verify_automatic_gripper_settle(
                                node, gripper_requested, changed, session_ids,
                                settle_s=args.gripper_settle_s)
                            if time.monotonic() - started >= args.max_wall_duration:
                                raise RuntimeError('Wall-clock budget exceeded before automatic re-ARM')
                            print('[AUTO] Starting verified re-ARM; operator E-stop must remain available')
                        else:
                            print('Verify grippers physically, workspace clearance,')
                            print('robot status, and independent E-stop.')
                            if input('Type RESUME to proceed: ').strip() != 'RESUME':
                                raise RuntimeError('Operator declined resume; both bridges stay stopped')

                        # An excessively long pause invalidates this rollout.
                        if time.monotonic() - pause_started > 60.0:
                            raise RuntimeError('Pause exceeded 60 s; start a new rollout instead')

                        # Each rearmed worker begins from its current SDK joints,
                        # and the client streams a fresh measured-pose HOLD target.
                        # Mark session as potentially live before sending ARM.
                        servo_ended = False
                        hold_target = resume_servoj_from_measured_pose(
                            node, session_ids, period=period, timeout_s=3.0)
                        paused_duration += time.monotonic() - pause_started
                        print(f'Resumed after waypoint {step + 1}; next action is {step + 2}')
                        break

                    print(f'Waypoint {step + 1} reached: '
                          f'FR1={fr1_error:.3f}°, FR2={fr2_error:.3f}°, '
                          f'grippers={node.gripper_states}')
                    break

                # Keep sending the SAME target so the bridge's
                # existing command watchdog does not expire.
                # Keep sending the SAME joint target while waiting; do NOT advance the
                # policy index on every 15 Hz tick, because arms might be lagging.
                if now - last_publish >= period:
                    node.send_targets(arm_action)
                    last_publish = now

                # Never skip a waypoint because of timeout.
                # Abort on late waypoint completion instead of silently skipping commands.
                if now - waypoint_start >= WAYPOINT_TIMEOUT_S:
                    raise RuntimeError(
                        f"Waypoint {step + 1} timeout: "
                        f"FR1 error={fr1_error:.2f}°, "
                        f"FR2 error={fr2_error:.2f}°"
                    )

            if session_timed_out or pause_test_completed:
                break
        # This print means the loop returned, not that both bridges confirmed STOP.
        print('ACT bounded session finished; requesting servo end.')
    # ============================================================================
    # 11) CLEANUP ON SUCCESS, ERROR, OR INTERRUPT
    # Even exceptions should trigger the dual STOP request if ARM was attempted.
    # STOP publication is not a physical E-stop and a reported 'fault' is NOT
    # equivalent to verified ServoMoveEnd success. Previous FR1 runs stayed
    # 'running' here: confirm both sessions ended before any new test.
    # ============================================================================
    finally:
        try:
            if armed and not servo_ended:
                print('Requesting STOP and verifying both ServoJ sessions...')
                request_verified_stop(node, session_ids, timeout_s=5.0)
                servo_ended = True
        except Exception as exc:
            print(f'CRITICAL: ServoJ shutdown not confirmed: {exc}')
            print('Do NOT re-arm: verify robot stop using established hardware procedures.')
        finally:
            print('Final bridge statuses:', node.statuses)
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted by operator. Confirm robot is stopped.')
        sys.exit(130)
