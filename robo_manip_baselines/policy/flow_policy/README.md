# Flow Policy

## Install
See [here](../../../doc/install.md#flow-policy) for installation.

## Dataset preparation
Collect demonstration data by [teleoperation](../../teleop).

## Data preprocessing
See [here](../../../doc/preprocessing_pointcloud.md) to perform data preprocessing for 3D point clouds.

## Model training
Train a model:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Train.py FlowPolicy --dataset_dir ./dataset/<dataset_name> --checkpoint_dir ./checkpoint/FlowPolicy/<checkpoint_name>
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
$ python ./bin/Rollout.py FlowPolicy MujocoUR5eCable --checkpoint ./checkpoint/FlowPolicy/<checkpoint_name>/policy_last.ckpt
```

## Technical Details
For more information on the technical details, please see the following paper:
```bib
@inproceedings{FlowPolicy_AAAI2025,
  author = {Qinglun Zhang and Zhen Liu and Haoqiang Fan and Guanghui Liu and Bing Zeng and Shuaicheng Liu},
  title = {FlowPolicy: Enabling Fast and Robust 3D Flow-based Policy via Consistency Flow Matching for Robot Manipulation},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {39},
  number = {14},
  pages = {14754--14762},
  year = {2025}
}
```
