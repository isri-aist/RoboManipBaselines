# Spatial attention recurrent neural network (SARNN)

## Install
See [here](../../doc/install.md#SARNN) for installation.

## Dataset preparation

Put your data collected under `data` directory. Here, we assume the name of your dataset directory as `teleop_data_sample`.

```console
$ tree data/teleop_data_sample/
data/teleop_data_sample/
├── env0
│   ├── UR5eCableEnv_env0_000.npz
│   └── UR5eCableEnv_env0_006.npz
├── env1
│   ├── UR5eCableEnv_env1_001.npz
│   └── UR5eCableEnv_env1_007.npz
├── env2
│   ├── UR5eCableEnv_env2_002.npz
│   └── UR5eCableEnv_env2_008.npz
├── env3
│   ├── UR5eCableEnv_env3_003.npz
│   └── UR5eCableEnv_env3_009.npz
├── env4
│   ├── UR5eCableEnv_env4_004.npz
│   └── UR5eCableEnv_env4_010.npz
└── env5
    ├── UR5eCableEnv_env5_005.npz
    └── UR5eCableEnv_env5_011.npz
```

Make numpy files in each of `train` (for training) and `test` directories (for validation).

```console
$ python ../utils/make_dataset.py --in_dir ./data/teleop_data_sample --out_dir ./data/learning_data_sample --train_keywords env0 env1 env4 env5 --test_keywords env2 env3 --nproc `nproc` --skip 6 --cropped_img_size 280 --resized_img_size 64
```

Visualize the generated data (optional).

```console
$ python ../utils/check_data.py --in_dir ./data/learning_data_sample --idx 0
```

## Model Training

Train the model. The trained weights are saved in the `log` folder.

```console
$ python ./bin/TrainSarnn.py --data_dir ./data/learning_data_sample --no_side_image --no_wrench --with_mask
```

## Test

Save a gif animation of the predicted image, attention points, and predicted joint angles in the `output` folder.

```console
$ python ./bin/test.py --data_dir ./data/learning_data_sample --filename ./log/YEAR_DAY_TIME/SARNN.pth --no_side_image --no_wrench
```

## Visualization of internal representation using PCA

Save the internal representation of the RNN as a gif animation in the `output` folder.

```console
$ python ./bin/test_pca.py --data_dir ./data/learning_data_sample --filename ./log/YEAR_DAY_TIME/SARNN.pth --no_side_image --no_wrench
```

## Policy rollout

Run a trained policy in the simulator.

```console
$ python ./bin/rollout/RolloutSarnnMujocoUR5eCable.py \
--checkpoint ./log/YEAR_DAY_TIME/SARNN.pth \
--cropped_img_size 280 --skip 6 --world_idx 1
```
The Python script is named `RolloutSarnn<task_name>.py`. The followings are supported as task_name: `MujocoUR5eCable`, `MujocoUR5eRing`, `MujocoUR5eParticle`, `MujocoUR5eCloth`.

Repeatedly run a trained policy in different environments in the simulator.

```console
$ ./scripts/iterate_rollout.sh ./log/YEAR_DAY_TIME SARNN.pth MujocoUR5eCable 280 6
```