# Experiments with real UR5e

https://github.com/user-attachments/assets/1b894ea0-6b5a-4bf8-abaa-8cd3ec2c4f85

## Install
See [here](./install.md#Real-UR5e-environments) for installation.

## Preparation
Hardware-specific information for robots and sensors is described in [RealUR5eDemoEnv.yaml](../robo_manip_baselines/envs/configs/RealUR5eDemoEnv.yaml).

Edit this file as follows:
- Specify the IP address of UR5e in `robot_ip`.
- Specify the RealSense serial number in `camera_ids` and the GelSight serial number in `gelsight_ids`, if any.

## Teleoperation
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ uv run python ./bin/Teleop.py RealUR5eDemo --config ./envs/configs/RealUR5eDemoEnv.yaml
```

## Policy rollout
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ uv run python ./bin/Rollout.py Act RealUR5eDemo --config ./envs/configs/RealUR5eDemoEnv.yaml --checkpoint <checkpoint_path> --wait_before_start
```
