# Action Chunking with Transformers (ACT) from LeRobot

## Install
See [here](../../../doc/install.md#LeRobot ACT) for installation.

## Dataset preparation
Collect demonstration data by [teleoperation](../../teleop).

convert to lerobot:

python3 ./misc/convert_rmb_data_to_lerobotv30.py         --raw-dir ../../dataset_baseline/MujocoUR5eCable_20250609/         --repo-id MujocoUR5eCable_20250609_v30_act         --config_path ./envs/configs/lerobot_ur5_dataset.yaml

## Model training
source /opt/venvs/lerobot_env/bin/activate

python /home/dev/RoboManipBaselines/third_party/lerobot/src/lerobot/scripts/lerobot_train.py   --dataset.repo_id=/home/dev/RoboManipBaselines/robo_manip_baselines/dataset/lerobot/MujocoUR5eCable_20250609_v30_act/   --policy.type=act   --output_dir=/home/dev/checkpoint_baseline/lerobot_act/act_MujocoUR5eCable_20250609/   --job_name=act_MujocoUR5eCable_20250609   --policy.device=cuda   --wandb.enable=false   --policy.push_to_hub=false --batch_size 24

## Policy rollout

python ./bin/Rollout.py LerobotAct MujocoUR5eCable --checkpoint /home/dev/checkpoint_baseline/lerobot_act/act_MujocoUR5eCable_20250609/checkpoints/last/pretrained_model/ --world_idx 0 --config-path ./envs/configs/lerobot_ur5_dataset.yaml

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
python3 ./misc/convert_rmb_data_to_lerobotv30.py \
        --raw-dir ./dataset/MujocoUR5eSimplePick/ \
        --repo-id MujocoUR5eSimplePick \
        --config_path ./envs/configs/lerobot_ur5e_dataset.yaml
        --camera_crops front:120,2,279,373 side:203,1,232,357
        --image mod

python /home/dev/RoboManipBaselines/third_party/lerobot/src/lerobot/scripts/lerobot_train.py   --dataset.repo_id=/home/dev/dataset_baseline/lerobot/MujocoUR5eSimplePick/   --policy.type=act   --output_dir=/home/dev/checkpoint_baseline/lerobot_act/MujocoUR5eSimplePick/   --job_name=act_MujocoUR5eSimplePick   --policy.device=cuda   --wandb.enable=false   --policy.push_to_hub=false --batch_size 24 --save_freq 1000

python ./bin/Rollout.py LerobotAct MujocoUR5eSimplePick --checkpoint /home/dev/checkpoint_baseline/lerobot_act/MujocoUR5eSimplePick/checkpoints/last/pretrained_model/ --world_idx 0 --data_config_path ./envs/configs/lerobot_ur5e_dataset.yaml --camera_crops front:135,92,256,256 side:192,112,256,256

