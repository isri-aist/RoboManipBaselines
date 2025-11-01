# ManiFlow Policy

## Install
See [here](../../../doc/install.md#maniflow-policy) for installation.

## Dataset preparation
Collect demonstration data by [teleoperation](../../teleop).

## Data preprocessing (for pointcloud policy)
See [here](../../../doc/preprocessing_pointcloud.md) to perform data preprocessing for 3D point clouds.

## Model training
Train a model with camera images:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Train.py ManiFlowPolicy image --dataset_dir ./dataset/<dataset_name> --checkpoint_dir ./checkpoint/ManiFlowPolicy/<checkpoint_name>
```

Train a model with pointcloud:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Train.py ManiFlowPolicy pointcloud --dataset_dir ./dataset/<dataset_name> --checkpoint_dir ./checkpoint/ManiFlowPolicy/<checkpoint_name>
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
$ python ./bin/Rollout.py ManiFlowPolicy MujocoUR5eCable --checkpoint ./checkpoint/ManiFlowPolicy/<checkpoint_name>/policy_last.ckpt
```

## Technical Details
For more information on the technical details, please see the following paper:
```bib
@inproceedings{ManiFlowPolicy_CoRL2025,
  author={Yan, Ge and Zhu, Jiyue and Deng, Yuquan and Yang, Shiqi and Qiu, Ri-Zhao and Cheng, Xuxin and Memmel, Marius and Krishna, Ranjay and Goyal, Ankit and Wang, Xiaolong and Fox, Dieter},
  title={{ManiFlow}: A General Robot Manipulation Policy via Consistency Flow Training},
  booktitle={Conference on Robot Learning},
  year={2025}
}
```
