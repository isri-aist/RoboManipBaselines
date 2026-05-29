# Action Chunking with Transformers (ACT)

## Install
See [here](../../../doc/install.md#ACT) for installation.

## Dataset preparation
Collect demonstration data by [teleoperation](../../teleop).

## Model training
Train a model:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ uv run python ./bin/Train.py Act --dataset_dir ./dataset/<dataset_name> --checkpoint_dir ./checkpoint/Act/<checkpoint_name>
```

> [!NOTE]
> The following error will occur if the chunk_size is larger than the time series length of the training data.
> In such a case, either set the `--skip` option to a small value, or set the `--chunk_size` option to a small value.
> ```console
> RuntimeError: The size of tensor a (70) must match the size of tensor b (102) at non-singleton dimension 0
> ```

## Policy rollout
Run a trained policy:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ uv run python ./bin/Rollout.py Act MujocoUR5eCable --checkpoint ./checkpoint/Act/<checkpoint_name>/policy_last.ckpt
```

## Technical Details
For more information on the technical details, please see the following paper:
```bib
@INPROCEEDINGS{ACT_RSS23,
  author = {Tony Z. Zhao and Vikash Kumar and Sergey Levine and Chelsea Finn},
  title = {Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  booktitle = {Proceedings of Robotics: Science and Systems},
  year = {2023},
  month = {July},
  doi = {10.15607/RSS.2023.XIX.016}
}
```
