# 3D Diffusion Policy

## Install
See [here](../../../doc/install.md#3D-Diffusion-policy) for installation.

## Dataset preparation
Collect demonstration data by [teleoperation](../../teleop).

## Data preprocessing
See [here](../../../doc/preprocessing_pointcloud.md) to perform data preprocessing for 3D point clouds.

## Model training
Train a model:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ uv run python ./bin/Train.py DiffusionPolicy3d --dataset_dir ./dataset/<dataset_name> --checkpoint_dir ./checkpoint/DiffusionPolicy3d/<checkpoint_name>
```

> [!NOTE]
> If you encounter the following error,
> ```console
> ImportError: cannot import name 'cached_download' from 'huggingface_hub'
> ```
> downgrade `huggingface_hub` by the following command.
> ```console
> $ pip install huggingface_hub==0.21.4
> ```

## Policy rollout
Run a trained policy:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ uv run python ./bin/Rollout.py DiffusionPolicy3d MujocoUR5eCable --checkpoint ./checkpoint/DiffusionPolicy3d/<checkpoint_name>/policy_last.ckpt
```

## Technical Details
For more information on the technical details, please see the following paper:
```bib
@inproceedings{3DDiffusionPolicy_RSS2024,
  author = {Yanjie Ze and Gu Zhang and Kangning Zhang and Chenyuan Hu and Muhan Wang and Huazhe Xu},
  title = {3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations},
  booktitle = {Proceedings of Robotics: Science and Systems},
  year = {2024}
}
```
