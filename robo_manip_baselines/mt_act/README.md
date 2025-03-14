# Multi-Task Action Chunking Transformer (MT-ACT)

**Caveats**: In this repository, this policy is less maintained than other policies.

## Install

Install [SARNN](../sarnn) according to [here](../sarnn/README.md#Install).

Install [roboagent](https://github.com/robopen/roboagent.git) by the following commands.
``` console
$ # Go to the top directory of this repository
$ git submodule update --init --recursive
$ pip install \
torchvision \
torch \
pyquaternion \
pyyaml \
rospkg \
pexpect \
mujoco \
dm_control \
opencv-python \
matplotlib \
einops \
packaging \
h5py \
h5py_cache
$ pip install ipython lru_cache click
$ cd third_party/roboagent/detr
$ pip install -e .
```

Check the version of `mujoco` with `pip show mujoco` and if it is not `3.1.6`, do the following commands.
```console
$ pip install mujoco==3.1.6
```

## Dataset preparation

Put your data collected under `data` directory. Here, we assume the name of your dataset directory as `teleop_data_sample`.

Sample data can be downloaded from the following links.
**This data is in an old format and will be replaced with a new format soon.**
- [Full data (270 trials)](https://aist.box.com/s/9qtkspyyzcxqvrssvumahfgvi31h5cet)
- [Partial data (24 trials)](https://aist.box.com/s/ks8l2ajmxhj48abxdvg4lowtp9134by5)

```console
$ tree data/teleop_data_sample/
data/teleop_data_sample/
├── task0_between-two
│   ├── env1
│   │   ├── UR5eCableEnv_env1_000.npz
│   │   └── UR5eCableEnv_env1_001.npz
│   ├── env3
│   │   ├── UR5eCableEnv_env3_000.npz
│   │   └── UR5eCableEnv_env3_001.npz
│   └── env5
│       ├── UR5eCableEnv_env5_000.npz
│       └── UR5eCableEnv_env5_001.npz
├── task1_around-red
│   ├── env1
│   │   ├── UR5eCableEnv_env1_000.npz
│   │   └── UR5eCableEnv_env1_001.npz
│   ├── env3
│   │   ├── UR5eCableEnv_env3_000.npz
│   │   └── UR5eCableEnv_env3_001.npz
│   └── env5
│       ├── UR5eCableEnv_env5_000.npz
│       └── UR5eCableEnv_env5_001.npz
├── task2_turn-blue
│   ├── env1
│   │   ├── UR5eCableEnv_env1_000.npz
│   │   └── UR5eCableEnv_env1_001.npz
│   ├── env3
│   │   ├── UR5eCableEnv_env3_000.npz
│   │   └── UR5eCableEnv_env3_001.npz
│   └── env5
│       ├── UR5eCableEnv_env5_000.npz
│       └── UR5eCableEnv_env5_001.npz
└── task3_around-two
    ├── env1
    │   ├── UR5eCableEnv_env1_000.npz
    │   └── UR5eCableEnv_env1_001.npz
    ├── env3
    │   ├── UR5eCableEnv_env3_000.npz
    │   └── UR5eCableEnv_env3_001.npz
    └── env5
        ├── UR5eCableEnv_env5_000.npz
        └── UR5eCableEnv_env5_001.npz
```

Make numpy files in each of `train` (for training) and `test` directories (for validation).

```console
$ python ../utils/make_multi_dataset.py \
--in_dir ./data/teleop_data_sample \
--out_dir ./data/learning_data_sample \
--skip 3 \
--train_keywords env1 env5 \
--test_keywords env3 \
--nproc `nproc`
```

## Model training

Train the model. The trained weights are saved in the `log` folder.
The training hyperparameters here (such as chunk_size) are the same as those in [act#training-models](https://github.com/isri-aist/RoboManipBaselines/tree/master/robo_manip_baselines/act#model-training).

```console
$ python ./bin/train.py \
--dataset_dir ./data/learning_data_sample --ckpt_dir ./log/YEAR_DAY_TIME \
--policy_class ACT \
--kl_weight 10 \
--chunk_size 100 \
--hidden_dim 512 \
--batch_size 8 \
--dim_feedforward 3200 \
--seed 0 \
--temporal_agg \
--num_epochs 20000 \
--lr 1e-5 \
--multi_task \
--run_name multi_task_run
```

## Policy rollout
Run a trained policy in the simulator.

```console
$ python ./bin/rollout/RolloutMtActMujocoUR5eCable.py \
--ckpt_dir ./log/YEAR_DAY_TIME --ckpt_name policy_best.ckpt --task_name task0_between-two \
--chunk_size 100 --seed 42 --skip 3 --world_idx 0
```
The Python script is named `RolloutMtAct<task_name>.py`. The followings are supported as task_name: `MujocoUR5eCable`, `MujocoUR5eRing`, `MujocoUR5eParticle`, `MujocoUR5eCloth`.

Repeatedly run a trained policy in different environments in the simulator.

```console
$ ./scripts/iterate_rollout.sh ./log/YEAR_DAY_TIME/ policy_last.ckpt MujocoUR5eCable 3
```

## Technical Details
For more information on the technical details, please see the following paper:
```bib
@INPROCEEDINGS{MTACT_ICRA2024,
  author = {Bharadhwaj, Homanga and Vakil, Jay and Sharma, Mohit and Gupta, Abhinav and Tulsiani, Shubham and Kumar, Vikash},
  title = {RoboAgent: Generalization and Efficiency in Robot Manipulation via Semantic Augmentations and Action Chunking},
  booktitle = {International Conference on Robotics and Automation},
  year = {2024},
  pages = {4788-4795},
  doi = {10.1109/ICRA57147.2024.10611293}
}
```
