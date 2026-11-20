#!/usr/bin/env python3
"""Preview-only ACT vs dual-FR3 ServoJ speed/acceleration feasibility check.

Requires the existing test_dual_fr5_act.py and
 test_dual_fr3_act_closed_loop_timed_skipaware.py in the same scripts folder.

Does NOT ARM, publish joint targets, or command either gripper. The simulated
trajectory is only the bridge's commanded setpoint follower, NOT physical robot
tracking; it ignores hardware latency, friction, and communication delays.
"""

import argparse
import csv
import math
import os
import time

import numpy as np
import rclpy
import torch

from test_dual_fr5_act import load_policy
from test_dual_fr3_act_closed_loop_timed_skipaware import Observations, inference


def simulate_chunk(actions_deg, initial_deg, action_period, dt, speed_limit, accel_limit,
                   target_gap_limit, lag_limit):
    """Model the rate-limited bridge commanded positions (optimistic baseline)."""
    q = np.array(initial_deg, dtype=np.float64).copy()
    qdot = np.zeros(12, dtype=np.float64)
    previous_target = q.copy()
    rows = []
    for i, target in enumerate(actions_deg):
        delta_deg = np.abs(target - previous_target)
        joint = int(np.argmax(delta_deg))
        required_speed = float(delta_deg[joint] / action_period)
        gap = np.abs(target - q)
        lag_index = int(np.argmax(gap))
        lag = float(gap[lag_index])
        # The real bridge independently updates each arm at 125 Hz. This is an
        # idealized synchronized 12-joint simulation, not a robot dynamics model.
        ticks = max(1, round(action_period / dt))
        for _ in range(ticks):
            requested = np.clip((target - q) / dt, -speed_limit, speed_limit)
            dv = np.clip(requested - qdot, -accel_limit * dt, accel_limit * dt)
            v_next = np.clip(qdot + dv, -speed_limit, speed_limit)
            move = np.sign(target - q) * np.minimum(
                np.abs(target - q), np.abs(v_next * dt))
            move = np.where(np.sign(v_next) == np.sign(target - q), move, 0.0)
            q = q + move
            qdot = move / dt
        end_err = float(np.max(np.abs(target - q)))
        rows.append({
            'action': i + 1,
            'nominal_time_s': round(i * action_period, 4),
            'largest_step_deg': round(float(delta_deg[joint]), 4),
            'step_joint': f'FR3_{joint//6+1}_J{joint%6+1}',
            'required_step_speed_deg_s': round(required_speed, 3),
            'speed_infeasible': int(required_speed > speed_limit + 1e-6),
            'ideal_lag_before_action_deg': round(lag, 4),
            'lag_joint': f'FR3_{lag_index//6+1}_J{lag_index%6+1}',
            'ideal_lag_exceeds_rollout_guard': int(lag > lag_limit),
            'bridge_target_gap_exceeded': int(lag > target_gap_limit),
            'ideal_lag_after_interval_deg': round(end_err, 4),
        })
        previous_target = target.copy()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--recorded-hz', type=float, default=15.0)
    ap.add_argument('--max-speed', type=float, default=5.0,
                    help='Bridge MAX_VELOCITY_DEG_S, verify installed controller')
    ap.add_argument('--max-accel', type=float, default=25.0,
                    help='Bridge MAX_ACCELERATION_DEG_S2, verify installed controller')
    ap.add_argument('--lag-limit', type=float, default=3.0)
    ap.add_argument('--output', default='act_timing_preflight.csv')
    args = ap.parse_args()
    if min(args.recorded_hz, args.max_speed, args.max_accel, args.lag_limit) <= 0:
        ap.error('Rates, limits, and acceleration must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        ap.error('CUDA unavailable; use --device cpu')
    device = torch.device(args.device)
    policy, meta = load_policy(args.checkpoint, device)
    skip = int(meta['data']['skip'])
    action_period = skip / args.recorded_hz
    if not 0.01 < action_period <= 5:
        raise ValueError(f'Unexpected action spacing {action_period}')
    print(f'\n[PREVIEW ONLY] skip={skip}, spacing={action_period:.3f}s, '
          f'ACT action rate={1/action_period:.2f}Hz')
    print(f'[BRIDGE LIMITS] speed={args.max_speed:.2f} deg/s, '
          f'accel={args.max_accel:.2f} deg/s^2, target lag={args.lag_limit:.2f} deg')
    print('No robot ARM/STOP/target/gripper commands are generated.\n')
    rclpy.init()
    node = Observations()
    try:
        obs, info, _ = node.wait_observation(timeout=12.0)
        actions, state = inference(policy, meta, device, obs, info)
        if actions.ndim != 2 or actions.shape[1] != 14 or state.shape[0] < 12:
            raise ValueError(f'Unexpected state/action shapes: {state.shape}, {actions.shape}')
        actions_deg = np.rad2deg(actions[:, :12])
        current_deg = np.rad2deg(state[:12])
        if not (np.isfinite(actions_deg).all() and np.isfinite(current_deg).all()):
            raise ValueError('NaN/Inf in ACT output or observed joint positions')
        rows = simulate_chunk(actions_deg, current_deg, action_period,
                              dt=0.008, speed_limit=args.max_speed,
                              accel_limit=args.max_accel,
                              target_gap_limit=12.0, lag_limit=args.lag_limit)
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print('ACT action | required step speed | ideal pre-action lag | limiting joint | finding')
        for r in rows:
            status = 'RATE HIGH' if r['speed_infeasible'] else 'rate OK'
            if r['ideal_lag_exceeds_rollout_guard']:
                status += ', LAG GUARD'
            if r['bridge_target_gap_exceeded']:
                status += ', BRIDGE GAP'
            print(f"{r['action']:>4d}       | {r['required_step_speed_deg_s']:>7.2f} deg/s      | "
                  f"{r['ideal_lag_before_action_deg']:>5.2f} deg            | "
                  f"{r['lag_joint']:<9s} | {status}")
        speed_bad = sum(r['speed_infeasible'] for r in rows)
        lag_bad = sum(r['ideal_lag_exceeds_rollout_guard'] for r in rows)
        print(f'\n[RESULT] {speed_bad}/{len(rows)} action intervals require faster '
              f'target changes than {args.max_speed:.1f} deg/s; '
              f'{lag_bad} exceed the {args.lag_limit:.1f} deg guard in the '
              'idealized bridge-follower model.')
        print('The model is an optimistic command-path estimate; actual robot tracking can be worse.')
        print('If the result says OK, an SDK/feedback/robot-side cause remains possible.')
        print('CSV saved:', os.path.abspath(args.output))
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
