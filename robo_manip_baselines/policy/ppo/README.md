# Multi-layer perceptron (MLP): Simplest policy

## Install
See [here](../../../doc/install.md#MLP) for installation.

## Dataset preparation
Collect demonstration data by [teleoperation](../../teleop).

## Model training
Train a model:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Train.py Mlp --dataset_dir ./dataset/<dataset_name> --checkpoint_dir ./checkpoint/Mlp/<checkpoint_name>
```

## Policy rollout
Run a trained policy:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Rollout.py Mlp MujocoUR5eCable --checkpoint ./checkpoint/Mlp/<checkpoint_name>/policy_last.ckpt
```

## Technical Details
This policy consists of a simple model structure with only linear layers. ResNet is used for image feature extraction.
