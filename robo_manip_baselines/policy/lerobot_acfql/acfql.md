python3 bin/Train.py LerobotAcfql --data_config_path ./envs/configs/lerobot_ur5e_dataset.yaml --policy_config_path ./envs/configs/lerobot_ur5e_acfql.yaml --dataset_dir /home/dev/RoboManipBaselines/robo_manip_baselines/dataset/MujocoUR5eSimplePick/ --checkpoint_dir /home/dev/checkpoint_baseline/lerobot_acfql/acfql_MujocoUR5eSimplePick/ --camera_crops front:135,92,256,256 side:192,112,256,256 

python ./bin/Rollout.py LerobotAcfql MujocoUR5eSimplePick     --data_config_path ./envs/configs/lerobot_ur5e_dataset.yaml     --policy_config_path ./envs/configs/lerobot_ur5e_acfql.yaml     --checkpoint /home/dev/checkpoint_baseline/lerobot_acfql/acfql_MujocoUR5eSimplePick/checkpoints/last/pretrained_model/     --online     --teleop_device keyboard_azerty     --auto_reset     --resume 

########
input_device spacemouse --skip_draw 8
python ./misc/find_camera_crops.py /home/dev/dataset_baseline/MujocoXarm7Pusht_Dataset100_20250922/ --target_size 128,128

--camera_crops front:209,189,256,256 side:268,197,256,256


python3 bin/Train.py LerobotAcfql --data_config_path ./envs/configs/lerobot_ur5e_dataset.yaml --policy_config_path ./envs/configs/lerobot_ur5e_acfql.yaml --dataset_dir /home/dev/RoboManipBaselines/robo_manip_baselines/dataset/MujocoUR5eSimplePick/ --checkpoint_dir /home/dev/checkpoint_baseline/lerobot_acfql/acfql_MujocoUR5eSimplePick/ --camera_crops front:135,92,256,256 side:192,112,256,256 

python3 ./misc/convert_rmb_data_to_lerobotv30.py \
        --raw-dir /home/dev/dataset_baseline/MujocoXarm7Pusht_Dataset100_20250922/ \
        --repo-id MujocoXarm7Pusht_Dataset100_20250922 \
        --config_path ./envs/configs/lerobot_xarm7_dataset.yaml
        --camera_crops front:209,189,256,256 side:268,197,256,256\
        --mode image


python3 bin/Train.py LerobotAcfql --data_config_path ./envs/configs/lerobot_xarm7_dataset.yaml --policy_config_path ./envs/configs/lerobot_xarm7_acfql.yaml --dataset_dir /home/dev/dataset_baseline/MujocoXarm7Pusht_Dataset100_20250922/ --checkpoint_dir /home/dev/checkpoint_baseline/lerobot_acfql/acfql_MujocoXarm7Pusht/ --camera_crops front:209,189,256,256 side:268,197,256,256 

python ./bin/Rollout.py LerobotAcfql MujocoXarm7Pusht     --data_config_path ./envs/configs/lerobot_xarm7_dataset.yaml     --policy_config_path ./envs/configs/lerobot_xarm7_acfql.yaml     --checkpoint /home/dev/checkpoint_baseline/lerobot_acfql/acfql_MujocoXarm7Pusht/checkpoints/last/pretrained_model/     --online     --teleop_device keyboard_azerty     --auto_reset     --resume 

python /home/dev/RoboManipBaselines/third_party/lerobot/src/lerobot/scripts/lerobot_train.py   --dataset.repo_id=/home/dev/dataset_baseline/lerobot/MujocoXarm7Pusht_Dataset100_20250922/   --policy.type=act   --output_dir=/home/dev/checkpoint_baseline/lerobot_act/act_MujocoXarm7Pusht_Dataset100_20250922/   --job_name=act_MujocoXarm7Pusht_Dataset100_20250922   --policy.device=cuda   --wandb.enable=false   --policy.push_to_hub=false --batch_size 24 --save_freq 3000

python ./bin/Rollout.py LerobotAct MujocoXarm7Pusht  --config ./envs/configs/RealUR10eDemoEnv.yaml --checkpoint /home/dev/checkpoint_baseline/lerobot_act/act_MujocoXarm7Pusht_Dataset100_20250922/checkpoints/last/pretrained_model/ --data_config_path ./envs/configs/lerobot_xarm7_dataset.yaml --target_camera_resolution 64,64 --camera_crops front:209,189,256,256 side:268,197,256,256