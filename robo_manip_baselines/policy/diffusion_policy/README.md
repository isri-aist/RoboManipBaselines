# Diffusion Policy

## Install
See [here](../../../doc/install.md#Diffusion-policy) for installation.

## Dataset preparation
Collect demonstration data by [teleoperation](../../teleop).

> [!NOTE]
> If you are using `pyenv` and encounter the error `No module named '_bz2'`, apply the following solution.  
> https://stackoverflow.com/a/71457141

## Model training
Train a model:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ uv run python ./bin/Train.py DiffusionPolicy --dataset_dir ./dataset/<dataset_name> --checkpoint_dir ./checkpoint/DiffusionPolicy/<checkpoint_name>
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
$ uv run python ./bin/Rollout.py DiffusionPolicy MujocoUR5eCable --checkpoint ./checkpoint/DiffusionPolicy/<checkpoint_name>/policy_last.ckpt
```

## Technical Details
For more information on the technical details, please see the following paper:
```bib
@INPROCEEDINGS{DiffusionPolicy_RSS23,,
  author = {Chi, Cheng and Feng, Siyuan and Du, Yilun and Xu, Zhenjia and Cousineau, Eric and Burchfiel, Benjamin and Song, Shuran},
  title = {Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  booktitle = {Proceedings of Robotics: Science and Systems},
  year = {2023},
  month = {July},
  doi = {10.15607/RSS.2023.XIX.026}
}
```
