# Action Chunking with Transformers (ACT) from LeRobot

## Install
See [here](../../../doc/install.md#LeRobot ACT) for installation.

## Dataset preparation
Collect demonstration data by [teleoperation](../../teleop).

convert to lerobot:
python3 ./misc/RefineRmbData.py --task_desc "Put the yellow cube on the black cube." ../../dataset_baseline/RealUR10eDemo_yellow_black_cubes/

python3 ./misc/convert_rmb_data_to_lerobotv30.py         --raw-dir ../../dataset_baseline/RealUR10eDemo_yellow_black_cubes/         --repo-id RealUR10eDemo_yellow_black_cubes_v30_pi0         --config_path ./envs/configs/lerobot_ur10e_dataset.yaml

## Model training


python3 ./misc/convert_rmb_data_to_lerobotv30.py \
        --raw-dir ../../dataset_baseline/RealUR10eDemo_yellow_black_cubes/ \
        --repo-id RealUR10eDemo_yellow_black_cubes_v30_pi0 \
        --config_path ./envs/configs/lerobot_ur10e_dataset.yaml

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /home/dev/RoboManipBaselines/third_party/lerobot/src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=/home/dev/RoboManipBaselines/robo_manip_baselines/dataset/lerobot/RealUR10eDemo_yellow_black_cubes_v30_pi0/ \
    --output_dir=/home/dev/checkpoint_baseline/lerobot_pi0/pi0_RealUR10eDemo_yellow_black_cubes/ \
    --job_name=pi0_RealUR10eDemo_yellow_black_cubes \
    --policy.device=cuda \
    --wandb.enable=false \
    --policy.push_to_hub=false \
    --batch_size 2 \
    --save_freq=5000 \
    --policy.path=lerobot/pi0

python /home/dev/RoboManipBaselines/third_party/lerobot/src/lerobot/scripts/lerobot_train.py   --dataset.repo_id=/home/dev/dataset_baseline/lerobot/MujocoUR5eSimplePick/   --policy.type=act   --output_dir=/home/dev/checkpoint_baseline/lerobot_act/MujocoUR5eSimplePick/   --job_name=act_MujocoUR5eSimplePick   --policy.device=cuda   --wandb.enable=false   --policy.push_to_hub=false --batch_size 24 --save_freq 1000

for i in {1..20}; do echo "--- Starting run $i of 20 ---"; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /home/dev/RoboManipBaselines/third_party/lerobot/src/lerobot/scripts/lerobot_train.py --config_path=/home/dev/checkpoint_baseline/lerobot_pi0/pi0_RealUR10eDemo_yellow_black_cubes/checkpoints/last/pretrained_model/train_config.json --dataset.repo_id=/home/dev/RoboManipBaselines/robo_manip_baselines/dataset/lerobot/RealUR10eDemo_yellow_black_cubes_v30_pi0/ --output_dir=/home/dev/checkpoint_baseline/lerobot_pi0/pi0_RealUR10eDemo_yellow_black_cubes/ --job_name=pi0_RealUR10eDemo_yellow_black_cubes --policy.device=cuda --wandb.enable=false --policy.push_to_hub=false --batch_size 2 --save_freq=5000 --resume=true; done

python ./bin/Rollout.py LerobotPi0 RealUR10eDemo --checkpoint /home/dev/checkpoint_baseline/lerobot_pi0/pi0_RealUR10eDemo_yellow_black_cubes/checkpoints/last/pretrained_model/ --config-path ./envs/configs/lerobot_ur10e_dataset.yaml --target_task "Put the yellow cube on the black cube." --config ./envs/configs/RealUR10eDemoEnv.yaml

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /home/dev/RoboManipBaselines/third_party/lerobot/src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=/home/dev/RoboManipBaselines/robo_manip_baselines/dataset/lerobot/MujocoUR5eSimplePick/ \
    --output_dir=/home/dev/checkpoint_baseline/lerobot_pi0/MujocoUR5eSimplePick/ \
    --job_name=MujocoUR5eSimplePick \
    --policy.device=cuda \
    --wandb.enable=false \
    --policy.push_to_hub=false \
    --batch_size 2 \
    --save_freq=5000 \
    --policy.path=lerobot/pi05