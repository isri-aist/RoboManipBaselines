# LeRobot pi0

## Install

You need three virtual environments for data preparation, training, and rollout.

### Env 1: Environment for data preparation

Install [RoboManipBaselines](https://github.com/isri-aist/RoboManipBaselines) according to [here](../../../doc/install.md#common-installation).

Install [LeRobot](https://github.com/huggingface/lerobot) by the following commands.
```console
# Go to the top directory of this repository
$ cd third_party/lerobot
$ pip install -e .
```

### Env 2: Environment for training

Install [LeRobot](https://github.com/huggingface/lerobot) by the following commands.
```console
# Go to any directory
$ git clone https://github.com/huggingface/lerobot -b v0.4.4
$ cd lerobot
$ pip install -e .[pi]
```

### Env 3: Environment for rollout

Install [RoboManipBaselines](https://github.com/isri-aist/RoboManipBaselines) according to [here](../../../doc/install.md#common-installation).

Install [LeRobot](https://github.com/huggingface/lerobot) by the following commands.
```console
# Go to the top directory of this repository
$ cd third_party/lerobot
$ pip install -e .[pi]
```

## Account Registration for PaLI-Gemma

Log in to your Hugging Face account, open the model card page for [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224), click the "Acknowledge license" button, and complete the required steps.

After that, register your Hugging Face token in the environment used for training and rollout with the following command:
```console
$ hf auth login
```

## Data Preparation

Convert RMB dataset into the LeRobot dataset format.

```console
# Use Env 1
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ uv run python misc/ConvertRmbDataToLerobot.py <rmb_dataset_dir> --output_dir <lerobot_dataset_dir>
```

## Model Training

Train a model.
Here is an example command for UR5e.

```console
# Use Env 2
$ lerobot-train \
  --dataset.root=<lerobot_dataset_dir> \
  --output_dir=<checkpoint_dir> \
  --dataset.repo_id=null \
  --policy.type=pi0 \
  --job_name=pi0_training \
  --policy.pretrained_path=lerobot/pi0_base \
  --policy.repo_id=local_repo \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=false \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --policy.input_features='{"observation.images.front_rgb": {"shape":[3,224,224], "type":"VISUAL"}, "observation.images.hand_rgb": {"shape":[3,224,224], "type":"VISUAL"}, "observation.state": {"shape":[7], "type":"STATE"}}' \
  --policy.output_features='{"action": {"shape":[7], "type":"ACTION"}}' \
  --policy.n_action_steps=8 \
  --policy.chunk_size=16 \
  --batch_size=32
```

> [!NOTE]
> When running on ABCI, the following commands are required in advance:
> ```console
> $ conda install -c conda-forge ffmpeg
> $ conda install -c conda-forge libstdcxx-ng
> $ export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
> ```

> [!NOTE]
> If `~/.cache/huggingface/hub/models--lerobot--pi0_base/snapshots/**/config.json` exists and contains unintended image observation keys in the `input_features` entry, remove those keys.

## Policy Rollout

Run a trained policy in the simulator.

```console
# Use Env 3
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ uv run python ./bin/Rollout.py Pi0 <task_name> --checkpoint <checkpoint_dir> --world_idx 0 --task_desc <task_description_text>
```

> [!NOTE]
> Specify the path to a `checkpoints/**/pretrained_model` directory (e.g., `checkpoints/100000/pretrained_model`) for `<checkpoint_dir>`.

## Technical Details
For more information on the technical details, please see the following paper:
```bib
@article{Pi0_arXiv2024,
  title={$$\backslash$pi\_0 $: A Vision-Language-Action Flow Model for General Robot Control},
  author={Black, Kevin and Brown, Noah and Driess, Danny and Esmail, Adnan and Equi, Michael and Finn, Chelsea and Fusai, Niccolo and Groom, Lachy and Hausman, Karol and Ichter, Brian and others},
  journal={arXiv preprint arXiv:2410.24164},
  year={2024}
}
```
