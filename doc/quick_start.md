# Quick start
This quick start allows you to collect data in the MuJoCo simulation and train and rollout the ACT policy.

## Install
Install RoboManipBaselines:
```console
$ git clone git@github.com:isri-aist/RoboManipBaselines.git --recursive
$ cd RoboManipBaselines
$ pip install -e .[act]
```

Install ACT from a third party:
```console
$ cd third_party/act/detr
$ pip install -e .
```

## Data collection by teleoperation
> [!TIP]
> Instead of collecting data by teleoperation, you can download the public dataset `TeleopMujocoUR5eCable_Dataset30` from [here](./dataset_list.md#Demonstrations-in-MuJoCo-environments).

Operate the robot in the simulation and save the data:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ # Connect a SpaceMouse to your PC
$ python ./bin/Teleop.py MujocoUR5eCable --world_idx_list 0 5 --input_device keyboard
```

> [!TIP]
> A teleoperation input device such as a 3D mouse can be used instead of a keyboard. See [here](../robo_manip_baselines/teleop/README.md).

In our experience, models can be trained stably with roughly 30 data sets.
The teleoperation data is saved in the `robo_manip_baselines/dataset/MujocoUR5eCable_<date_suffix>` directory (e.g., `MujocoUR5eCable_20240101_120000`) in HDF5 format.

## Model training
Train the ACT:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Train.py Act --dataset_dir ./dataset/MujocoUR5eCable_20240101_120000
```
The learned parameters are saved in the `robo_manip_baselines/checkpoint/Act/<dataset_name>_Act_<date_suffix>` directory (e.g., `MujocoUR5eCable_20240101_120000_Act_20240101_130000`) in HDF5 format.

> [!NOTE]
> The following error will occur if the chunk_size is larger than the time series length of the training data.
> In such a case, either set the `--skip` option to a small value, or set the `--chunk_size` option to a small value.
> ```console
> RuntimeError: The size of tensor a (70) must match the size of tensor b (102) at non-singleton dimension 0
> ```

## Policy rollout
Rollout the ACT in the simulation:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Rollout.py Act MujocoUR5eCable \
--checkpoint ./checkpoint/Act/MujocoUR5eCable_20240101_120000_Act_20240101_130000/policy_last.ckpt \
--world_idx 0
```
