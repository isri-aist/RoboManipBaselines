# Spatial attention recurrent neural network (SARNN)

## Install
See [here](../../../doc/install.md#SARNN) for installation.

## Dataset preparation
Collect demonstration data by [teleoperation](../../teleop).

## Model training
Train a model:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ uv run python ./bin/Train.py Sarnn --dataset_dir ./dataset/<dataset_name> --checkpoint_dir ./checkpoint/Sarnn/<checkpoint_name>
```
The `--image_crop_size_list` option should be specified appropriately for each task.

## Policy rollout
Run a trained policy:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ uv run python ./bin/Rollout.py Sarnn MujocoUR5eCable --checkpoint ./checkpoint/Sarnn/<checkpoint_name>/policy_last.ckpt
```

## Technical Details
For more information on the technical details, please see the following paper:
```bib
@INPROCEEDINGS{SARNN_ICRA2022,
  author = {Ichiwara, Hideyuki and Ito, Hiroshi and Yamamoto, Kenjiro and Mori, Hiroki and Ogata, Tetsuya},
  title = {Contact-Rich Manipulation of a Flexible Object based on Deep Predictive Learning using Vision and Tactility},
  booktitle = {International Conference on Robotics and Automation},
  year = {2022},
  pages = {5375-5381},
  doi = {10.1109/ICRA46639.2022.9811940}
}
```
