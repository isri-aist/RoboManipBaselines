| dt [s] | 許容角度差 [rad] | 許容角度差 [deg] | 備考 |
| --- | --- | --- | --- |
| 0.02 | 0.125663706 | 7.20 | `scaled_joint_vel_limit = 2\pi` rad/s |
| 0.05 | 0.314159265 | 18.00 | 同上 |

**速度制限**

- `joint_vel_limit = π` rad/s (`robo_manip_baselines/envs/real/RealEnvBase.py:68`)
- `joint_vel_limit_scale = 2.0` （「overwrite_command_for_safety」で使用）
- `scaled_joint_vel_limit = joint_vel_limit_scale * joint_vel_limit = 2π ≈ 6.28319` rad/s

> 各値は `scaled_joint_vel_limit = joint_vel_limit_scale * joint_vel_limit = 2.0 * \pi` rad/s を想定し、`許容角度差 = scaled_joint_vel_limit * dt` で計算。
