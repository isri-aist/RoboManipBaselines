# Multi-Task Action Chunking Transformer (MT-ACT)

## Install
See [here](../../../doc/install.md#MT-ACT) for installation.

## Dataset preparation
Collect demonstration data by [teleoperation](../../teleop).

## Data preprocessing
Specify the task description (language instruction) and store it in the dataset:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ uv run python ./misc/RefineRmbData.py ./dataset/<dataset_name> --task_desc <task_desc>
```
An example of `<task_desc>` is "Pass the cable between the two pins."

## Model training
Train a model:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ uv run python ./bin/Train.py MtAct --dataset_dir ./dataset/<dataset_name> --checkpoint_dir ./checkpoint/MtAct/<checkpoint_name>
```

## Policy rollout
Run a trained policy:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ uv run python ./bin/Rollout.py MtAct MujocoUR5eCable --checkpoint ./checkpoint/MtAct/<checkpoint_name>/policy_last.ckpt --task_desc "Pick up an object"
```

## Technical Details
For more information on the technical details, please see the following paper:
```bib
@INPROCEEDINGS{MT_ACT_ICRA2024,
  author = {Homanga Bharadhwaj and Jay Vakil and Mohit Sharma and Abhinav Gupta and Shubham Tulsiani and Vikash Kumar},
  title = {Roboagent: Generalization and efficiency in robot manipulation via semantic augmentations and action chunking},
  booktitle = {IEEE International Conference on Robotics and Automation},
  year = {2024},
  doi = {10.1109/ICRA57147.2024.10611293}
}
```
