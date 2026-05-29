# Isaac GR00T

## Install

You need four virtual environments for data preparation, LeRobot data conversion, training, and rollout.

### Env 1: Environment for data preparation

Install [RoboManipBaselines](https://github.com/isri-aist/RoboManipBaselines) according to [here](../../../doc/install.md#common-installation).

Install [LeRobot](https://github.com/huggingface/lerobot) by the following commands.
```console
# Go to the top directory of this repository
$ cd third_party/lerobot
$ pip install -e .
```

### Env 2: Environment for LeRobot data conversion

Install [lerobot_conversion](https://github.com/yu-kitagawa/Isaac-GR00T/tree/main/scripts/lerobot_conversion) by the following commands.
```console
# Go to any directory
$ git clone https://github.com/yu-kitagawa/Isaac-GR00T
$ cd Isaac-GR00T/scripts/lerobot_conversion
$ pip install -e .
```

### Env 3: Environment for training

Install [GR00T](https://github.com/yu-kitagawa/Isaac-GR00T) by the following commands.
```console
# Go to any directory
$ git clone https://github.com/yu-kitagawa/Isaac-GR00T
$ cd Isaac-GR00T
$ pip install -e .
$ pip install wheel packaging ninja
$ pip install --no-build-isolation flash-attn==2.7.4.post1
```

> [!NOTE]
> When running on ABCI, the following command must be executed on a GPU node:
> ```console
> $ pip install --no-build-isolation flash-attn==2.7.4.post1
> ```
>
> Before running this command on a GPU node, you need to load the CUDA module:
> ```console
> $ source /etc/profile.d/modules.sh
> $ module load cuda/12.6/12.6.1
> ```
>
> and set a temporary directory:
> ```console
> $ export TMPDIR=$HOME/tmp
> $ mkdir -p $TMPDIR
> ```

### Env 4: Environment for rollout

Install [GR00T](https://github.com/yu-kitagawa/Isaac-GR00T) as in Env 3.

Install [RoboManipBaselines](https://github.com/isri-aist/RoboManipBaselines) according to [here](../../../doc/install.md#common-installation).

If `numpy>=2.0.0`, install `numpy<2.0.0`.
```console
$ pip install "numpy<2.0.0"
```

## Data Preparation

Convert RMB dataset into the LeRobot dataset format.

```console
# Use Env 1
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ uv run python misc/ConvertRmbDataToLerobot.py <rmb_dataset_dir> --output_dir <lerobot_dataset_dir>
```

## LeRobot Data Conversion

Convert from LeRobot v3 to v2.

```console
# Use Env 2
# Go to Isaac-GR00T directory
$ cd scripts/lerobot_conversion
$ uv run python convert_v3_to_v2.py --root <dataset_root> --repo-id <dataset_repo_id>
$ cp <lerobot_dataset_dir>_v3.0/meta/modality.json <lerobot_dataset_dir>/meta
```

> [!NOTE]
> Let `<dataset_repo_id>` be the name of the final directory in `<lerobot_dataset_dir>`, and `<dataset_root>` be the path to its parent directory.
>
> For example, if `<lerobot_dataset_dir>` is `./dir_aaa/dir_bbb/dir_ccc`, then `<dataset_repo_id>` is `dir_ccc`, and `<dataset_root>` is `./dir_aaa/dir_bbb`.

## Model Training

Train a model.
Here is an example command for UR5e.

```console
# Use Env 3
# Go to Isaac-GR00T directory
$ export NUM_GPUS=1
$ CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
  --dataset-path <lerobot_dataset_dir> \
  --output-dir <checkpoint_dir> \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/UR5e/ur5e_config.py \
  --num-gpus $NUM_GPUS \
  --save-total-limit 5 \
  --save-steps 60000 \
  --max-steps 60000 \
  --global-batch-size 64 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader-num-workers 4
```

> [!NOTE]
> When running on ABCI, you need to load the CUDA module:
> ```console
> $ source /etc/profile.d/modules.sh
> $ module load cuda/12.6/12.6.1
> ```

## Policy Rollout

Run a trained policy in the simulator.

```console
# Use Env 4
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ uv run python ./bin/Rollout.py Gr00t <task_name> --checkpoint <checkpoint_dir> --world_idx 0 --no_plot --task_desc <task_description_text>
```

> [!NOTE]
> Specify the path to a `checkpoint-**` directory (e.g., `checkpoint-60000`) for `<checkpoint_dir>`.

## Technical Details
For more information on the technical details, please see the following paper:
```bib
@article{GR00T_arXiv2025,
  title={{GR00T}: An open foundation model for generalist humanoid robots},
  author={Bjorck, Johan and Casta{\~n}eda, Fernando and Cherniadev, Nikita and Da, Xingye and Ding, Runyu and Fan, Linxi and Fang, Yu and Fox, Dieter and Hu, Fengyuan and Huang, Spencer and others},
  journal={arXiv preprint arXiv:2503.14734},
  year={2025}
}
```
