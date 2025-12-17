Training with Human-in-the-Loop Sample-Efficient RL (HIL-SERL)

This guide provides a complete workflow for training a real-world robot policy within the robomanipbaseline framework using the Human-in-the-Loop Sample-Efficient Reinforcement Learning (HIL-SERL) methodology from lerobot.
The HIL-SERL Method

HIL-SERL is a sample-efficient reinforcement learning algorithm that combines offline human demonstrations, online learning, and live human interventions to rapidly teach robots complex tasks. The approach integrates three key components:

    Offline Demonstrations & Reward Classifier: Training begins with a small set of human-teleoperated episodes. This data is used to train a vision-based reward classifier that provides a dense reward signal to the policy.

    Distributed Actor-Learner Loop: The system uses a distributed setup where a central learner (typically on a powerful machine with a GPU) continuously updates a Soft Actor-Critic (SAC) policy. One or more actors (the physical robots) execute this policy in the real world, gathering new experience.

    Human-in-the-Loop Interventions: During online training, a human operator supervises the robot. Using a teleoperation device (like a spacemouse), the human can intervene at any time to correct unproductive behaviors, providing crucial feedback that guides the policy's exploration.

This combination allows the policy to learn from diverse data sources—offline demos, online exploration, and human corrections—making it highly efficient for real-world robotics.
HIL-SERL Workflow Overview

    Collect Demonstration Data: Record successful and unsuccessful episodes for your task using robomanipbaseline's teleoperation tools.

    Train the Reward Classifier: Use the demonstration data to train a classifier that can predict task success from images.

    Configure the Training Run: Set up a single YAML configuration file with all parameters for the environment, policy, and actor-learner communication.

    Start the Learner: Launch the central learner process, which will wait for actors to connect.

    Start the Actor: Launch the actor process on the robot. The robot will begin executing the policy, and you can intervene as needed.

python ./bin/Teleop.py RealUR10eDemo --config ./envs/configs/RealUR10eDemoEnv.yaml

python3 ./misc/InferSafetyLimits.py --position_leeway 0.25 --velocity_leeway 0.25 --output_path /home/dev/RoboManipBaselines/robo_manip_baselines/dataset/RealUR10eDemo_20250922_122447/safety_limits.json /home/dev/dataset_baseline/RealUR10eDemo_20250922_122447/ 

add /home/dev/RoboManipBaselines/robo_manip_baselines/dataset/RealUR10eDemo_20250922_122447/safety_limits.json to RealUR10eDemoEnv.yaml
python3 bin/Train.py LerobotHilSerlClassifier --dataset_dir /home/dev/RoboManipBaselines/robo_manip_baselines/dataset/RealUR10eDemo_20250922_122447/ --camera_names zed zed2i mini oak --checkpoint_dir ./checkpoint/lerobot_hilserl/RewardClassifierRealUR10eDemo_PickUpYellow/

python3 ./misc/convert_rmb_data_to_lerobotv30.py \
        --raw-dir /home/dev/dataset_baseline/RealUR10eDemo_20250922_122447/ \
        --repo-id RealUR10eDemo_PickUpYellow \
        --config_path ./envs/configs/lerobot_ur10e.yaml

/home/dev/dataset_baseline/lerobot/RealUR10eDemo_PickUpYellow

python3 bin/Train.py LerobotHilSerl --config_path ./envs/configs/lerobot_ur10e_hilserl.yaml --dataset_dir /home/dev/dataset_baseline/RealUR10eDemo_20250922_122447/ --state_keys measured_eef_pose measured_gripper_joint_pos --action_keys command_eef_pose command_gripper_joint_pos  --camera_names zed zed2i mini oak

python3 bin/Rollout.py LerobotHilSerl RealUR10eDemo --config_path ./envs/configs/lerobot_ur10e_hilserl.yaml --spacemouse --config ./envs/configs/RealUR10eDemoEnv.yaml --image-size 128 128 --checkpoint /home/dev/RoboManipBaselines/robo_manip_baselines/checkpoint/LerobotHilSerl/RealUR10eDemo_20250922_122447_LerobotHilSerl_20250923_103747/


###########
 bin/Train.py LerobotHilSerlClassifier --dataset_dir /home/dev/dataset_baseline/RealUR10eDemo_20250922_122447/ --camera_names mini --checkpoint_dir ./checkpoint/lerobot_hilserl/RewardClassifierRealUR10eDemo_PickUpYellow_mini/
Trajectory mode: Loading trajectories from '/home/dev/dataset_baseline/RealUR10eDemo_20250922_122447/' and labeling last 5 frames as success.

python3 ./misc/convert_rmb_data_to_lerobotv30.py \
        --raw-dir /home/dev/dataset_baseline/RealUR10eDemo_20250922_122447/ \
        --repo-id RealUR10eDemo_PickUpYellow_mini \
        --config_path ./envs/configs/lerobot_ur10e.yaml

python3 bin/Train.py LerobotHilSerl --config_path ./envs/configs/lerobot_ur10e_hilserl.yaml --dataset_dir /home/dev/dataset_baseline/RealUR10eDemo_20250922_122447/ --state_keys measured_eef_pose measured_gripper_joint_pos --action_keys command_eef_pose command_gripper_joint_pos  --camera_names mini

python3 bin/Rollout.py LerobotHilSerl RealUR10eDemo --config_path ./envs/configs/lerobot_ur10e_hilserl.yaml --spacemouse --config ./envs/configs/RealUR10eDemoEnv.yaml --image-size 128 128 --checkpoint /home/dev/RoboManipBaselines/robo_manip_baselines/checkpoint/LerobotHilSerl/RealUR10eDemo_20250922_122447_LerobotHilSerl_20250923_173321/
########

###########

 bin/Train.py LerobotHilSerlClassifier --dataset_dir /home/dev/dataset_baseline/RealUR10eDemo_20250922_122447/ --camera_names mini --checkpoint_dir ./checkpoint/lerobot_hilserl/RewardClassifierRealUR10eDemo_PickUpYellow_mini/

Trajectory mode: Loading trajectories from '/home/dev/dataset_baseline/RealUR10eDemo_20250922_122447/' and labeling last 5 frames as success.

python3 ./misc/convert_rmb_data_to_lerobotv30.py \
        --raw-dir /home/dev/dataset_baseline/RealUR10eDemo_20250922_122447/ \
        --repo-id RealUR10eDemo_PickUpYellow \
        --config_path ./envs/configs/lerobot_ur10e_dataset.yaml

python /home/dev/RoboManipBaselines/third_party/lerobot/src/lerobot/scripts/lerobot_train.py   --dataset.repo_id=/home/dev/RoboManipBaselines/robo_manip_baselines/dataset/lerobot/RealUR10eDemo_PickUpYellow/ --output_dir=/home/dev/checkpoint_baseline/lerobot_hilserl/reward_classifer_RealUR10eDemo_PickUpYellow/   --job_name=reward_classifer_RealUR10eDemo_PickUpYellow   --policy.device=cuda   --wandb.enable=false   --policy.push_to_hub=false --config_path /home/dev/RoboManipBaselines/robo_manip_baselines/envs/configs/lerobot_hislerl_reward_classifier.json


python /home/dev/RoboManipBaselines/third_party/lerobot/src/lerobot/scripts/lerobot_train.py   --dataset.repo_id=/home/dev/RoboManipBaselines/robo_manip_baselines/dataset/lerobot/RealUR10eDemo_PickUpYellow/ --output_dir=/home/dev/checkpoint_baseline/lerobot_hilserl/hilserl_RealUR10eDemo_PickUpYellow/   --job_name=hilserl_RealUR10eDemo_PickUpYellow   --policy.device=cuda   --wandb.enable=false   --policy.push_to_hub=false --config_path /home/dev/RoboManipBaselines/robo_manip_baselines/envs/configs/lerobot_hislerl_ur10e_learner.json --env.discover_packages_path=robo_manip_baselines.envs.real.ur10e



python /home/dev/RoboManipBaselines/third_party/lerobot/src/lerobot/scripts/lerobot_train.py \
    --config_path /home/localadm/Workdir/ur10_ros2_robomanipbaselines/scripts/RoboManipBaselines/robo_manip_baselines/envs/configs/lerobot_hislerl_ur10e_learner.json \
    

python3 bin/Train.py LerobotHilSerl --data_config_path ./envs/configs/lerobot_ur10e_dataset.yaml --policy_config_path ./envs/configs/lerobot_ur10e_hilserl.yaml --dataset_dir /home/dev/dataset_baseline/RealUR10eDemo_20250922_122447/ --state_keys measured_eef_pose measured_gripper_joint_pos --action_keys command_eef_pose_rel command_gripper_joint_pos  --camera_names mini oak zed zed2i --checkpoint_dir /home/dev/checkpoint_baseline/lerobot_hilserl/hilserl_RealUR10eDemo_PickUpYellow/

python3 bin/Rollout.py LerobotHilSerl RealUR10eDemo --config_path ./envs/configs/lerobot_ur10e_hilserl.yaml --spacemouse --config ./envs/configs/RealUR10eDemoEnv.yaml --image-size 128 128 --checkpoint /home/dev/RoboManipBaselines/robo_manip_baselines/checkpoint/LerobotHilSerl/RealUR10eDemo_20250922_122447_LerobotHilSerl_20250923_190259/ 
########
python3 ./misc/convert_rmb_data_to_lerobotv30.py         --raw-dir /home/dev/dataset_baseline/RealUR10eDemo_20250922_122447/         --repo-id RealUR10eDemo_PickUpYellow         --config_path ./envs/configs/lerobot_ur10e_dataset.yaml


python /home/dev/RoboManipBaselines/third_party/lerobot/src/lerobot/scripts/lerobot_train.py   --dataset.repo_id=/home/dev/dataset_baseline/lerobot/RealUR10eDemo_PickUpYellow/ --output_dir=/home/dev/checkpoint_baseline/lerobot_hilserl/RealUR10eDemo_20250922_122447_hilserl_reward_classifier/   --job_name=reward_classifier_RealUR10eDemo_PickUpYellow   --policy.device=cuda   --wandb.enable=false   --policy.push_to_hub=false --config_path /home/dev/RoboManipBaselines/robo_manip_baselines/policy/lerobot_hil_serl/ur10e_lerobot_hislerl_reward_classifier.json

python3 bin/Train.py LerobotHilSerl --data_config_path ./envs/configs/lerobot_ur10e_dataset.yaml --policy_config_path ./envs/configs/lerobot_ur10e_hilserl.yaml --dataset_dir /home/dev/dataset_baseline/RealUR10eDemo_20250922_122447/ --checkpoint_dir /home/dev/checkpoint_baseline/lerobot_hilserl/hilserl_RealUR10eDemo_PickUpYellow/ --offline_steps 100

python3 bin/Train.py LerobotHilSerl --data_config_path ./envs/configs/lerobot_ur10e_dataset.yaml --policy_config_path ./envs/configs/lerobot_ur10e_hilserl.yaml --dataset_dir /home/dev/dataset_baseline/RealUR10eDemo_20250922_122447/ --checkpoint_dir /home/dev/checkpoint_baseline/lerobot_hilserl/hilserl_RealUR10eDemo_PickUpYellow/ --resume

python ./bin/Rollout.py LerobotHilSerl RealUR10eDemo --data_config_path ./envs/configs/lerobot_ur10e_dataset.yaml --policy_config_path ./envs/configs/lerobot_ur10e_hilserl.yaml --checkpoint /home/dev/checkpoint_baseline/lerobot_hilserl/hilserl_RealUR10eDemo_PickUpYellow/ --config ./envs/configs/RealUR10eDemoEnv.yaml --online --teleop_device spacemouse

###
python3 ./misc/convert_rmb_data_to_lerobotv30.py         --raw-dir ../../dataset_baseline/MujocoUR5eCable_20250609/         --repo-id MujocoUR5eCable_20250609_v30_hilserl         --config_path ./envs/configs/lerobot_ur5e_dataset.yaml

python /home/dev/RoboManipBaselines/third_party/lerobot/src/lerobot/scripts/lerobot_train.py   --dataset.repo_id=/home/dev/dataset_baseline/lerobot/MujocoUR5eCable_20250609_v30_hilserl/ --output_dir=/home/dev/checkpoint_baseline/lerobot_hilserl/MujocoUR5eCable_20250609_v30_hilserl_reward_classifier/   --job_name=reward_classifer_MujocoUR5eCable_20250609_v30_hilserl   --policy.device=cuda   --wandb.enable=false   --policy.push_to_hub=false --config_path /home/dev/RoboManipBaselines/robo_manip_baselines/policy/lerobot_hil_serl/mujoco_lerobot_hislerl_reward_classifier.json

python3 bin/Train.py LerobotHilSerl --data_config_path ./envs/configs/lerobot_ur5e_dataset.yaml --policy_config_path ./envs/configs/lerobot_ur5e_hilserl.yaml --dataset_dir /home/dev/dataset_baseline/MujocoUR5eCable_20250609/ --checkpoint_dir /home/dev/checkpoint_baseline/lerobot_hilserl/hilserl_MujocoUR5eCable_20250609/ --offline_steps 10000

python ./bin/Rollout.py LerobotHilSerl MujocoUR5eCable --data_config_path ./envs/configs/lerobot_ur5e_dataset.yaml --policy_config_path ./envs/configs/lerobot_ur5e_hilserl.yaml --checkpoint /home/dev/checkpoint_baseline/lerobot_hilserl/hilserl_MujocoUR5eCable_20250609/ --online --teleop_device keyboard --enable_manual_reward


###########
python3 ./misc/convert_rmb_data_to_lerobotv30.py         --raw-dir /home/dev/dataset_baseline/RealUR10eDemo_yellow_black_cubes_50/         --repo-id RealUR10eDemo_yellow_black_cubes_50         --config_path ./envs/configs/lerobot_ur10e_dataset.yaml

python /home/dev/RoboManipBaselines/third_party/lerobot/src/lerobot/scripts/lerobot_train.py   --dataset.repo_id=/home/dev/dataset_baseline/lerobot/RealUR10eDemo_yellow_black_cubes_50/ --output_dir=/home/dev/checkpoint_baseline/lerobot_hilserl/RealUR10eDemo_20250922_122447_hilserl_reward_classifier/   --job_name=reward_classifier_RealUR10eDemo_yellow_black_cubes_50   --policy.device=cuda   --wandb.enable=false   --policy.push_to_hub=false --config_path /home/dev/RoboManipBaselines/robo_manip_baselines/policy/lerobot_hil_serl/ur10e_lerobot_hislerl_reward_classifier.json

python3 bin/Train.py LerobotHilSerl --data_config_path ./envs/configs/lerobot_ur10e_dataset.yaml --policy_config_path ./envs/configs/lerobot_ur10e_hilserl.yaml --dataset_dir /home/dev/dataset_baseline/RealUR10eDemo_yellow_black_cubes_50/ --checkpoint_dir /home/dev/checkpoint_baseline/lerobot_hilserl/RealUR10eDemo_yellow_black_cubes_50/ --offline_steps 2000

python3 bin/Train.py LerobotHilSerl --data_config_path ./envs/configs/lerobot_ur10e_dataset.yaml --policy_config_path ./envs/configs/lerobot_ur10e_hilserl.yaml --dataset_dir /home/dev/dataset_baseline/RealUR10eDemo_20250922_122447/ --checkpoint_dir /home/dev/checkpoint_baseline/lerobot_hilserl/RealUR10eDemo_yellow_black_cubes_50/ --resume

python ./bin/Rollout.py LerobotHilSerl RealUR10eDemo --data_config_path ./envs/configs/lerobot_ur10e_dataset.yaml --policy_config_path ./envs/configs/lerobot_ur10e_hilserl.yaml --checkpoint /home/dev/checkpoint_baseline/lerobot_hilserl/hilserl_RealUR10eDemo_PickUpYellow/ --config ./envs/configs/RealUR10eDemoEnv.yaml --online --teleop_device spacemouse
