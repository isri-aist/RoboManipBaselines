import argparse
import csv
import json
import os

import cv2
import matplotlib.pylab as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from robo_manip_baselines.common import (
    DataKey,
    RolloutBase,
    denormalize_data,
    normalize_data,
)

def gripper_q_robomanip_to_maniskill(q_robomanip):
    """Convert RoboManip gripper scalar to ManiSkill scale."""

    return (q_robomanip - 840.0) / (-1000.0)


def gripper_q_maniskill_to_robomanip(q_maniskill):
    """Convert ManiSkill gripper scalar to RoboManip scale."""

    return q_maniskill * (-1000.0) + 840.0


def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ManiSkillPpoAgent(nn.Module):
    """Reproduction of ManiSkill PPO agent architecture for rollout."""

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        self.critic = nn.Sequential(
            _layer_init(nn.Linear(self.obs_dim, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            _layer_init(nn.Linear(self.obs_dim, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, self.action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, self.action_dim) * -0.5)

    def get_value(self, obs):
        return self.critic(obs)

    def get_action(self, obs, deterministic=False):
        action_mean = self.actor_mean(obs)
        if deterministic:
            return action_mean

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()


_NORMALIZED_ACTION_LOW = torch.tensor(-1.0, dtype=torch.float32)
_NORMALIZED_ACTION_HIGH = torch.tensor(1.0, dtype=torch.float32)

_DELTA_PHYSICAL_LOW = torch.tensor(
    [
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
    ],
    dtype=torch.float32,
)

_DELTA_PHYSICAL_HIGH = torch.tensor(
    [
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
    ],
    dtype=torch.float32,
)

_JOINT_POSITION_LOW = torch.tensor(
    [
        -6.2831853,
        -2.059,
        -6.2831853,
        -0.19198,
        -6.2831853,
        -1.69297,
        -6.2831853,
        0.05,
    ],
    dtype=torch.float32,
)

_JOINT_POSITION_HIGH = torch.tensor(
    [
        6.2831853,
        2.0944,
        6.2831853,
        3.927,
        6.2831853,
        3.1415927,
        6.2831853,
        0.84,
    ],
    dtype=torch.float32,
)


class RolloutPpo(RolloutBase):
    def set_additional_args(self, parser):
        super().set_additional_args(parser)

        parser.add_argument(
            "--ppo-deterministic",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use deterministic ManiSkill PPO actions (default: True).",
        )
        parser.add_argument(
            "--ppo-use-cuda",
            action=argparse.BooleanOptionalAction,
            default=torch.cuda.is_available(),
            help="Enable CUDA for ManiSkill PPO if available (default: enabled when CUDA exists).",
        )
    def setup_model_meta_info(self):
        checkpoint_dir = os.path.split(self.args.checkpoint)[0]
        model_meta_info_path = os.path.join(checkpoint_dir, "model_meta_info.pkl")

        if os.path.isfile(model_meta_info_path):
            super().setup_model_meta_info()
            return

        self.model_meta_info = self._build_default_model_meta_info()
        print(
            f"[{self.__class__.__name__}] model_meta_info.pkl not found. Using default ManiSkill-compatible meta info."
        )

        self.state_keys = self.model_meta_info["state"]["keys"]
        self.action_keys = self.model_meta_info["action"]["keys"]
        self.camera_names = self.model_meta_info["image"]["camera_names"]
        self.state_dim = len(self.model_meta_info["state"]["example"])
        self.action_dim = len(self.model_meta_info["action"]["example"])

        if self.args.skip is None:
            self.args.skip = self.model_meta_info["data"]["skip"]
        if self.args.skip_draw is None:
            self.args.skip_draw = self.args.skip

    def _build_default_model_meta_info(self):
        default_state_keys = [
            DataKey.MEASURED_JOINT_POS,
            DataKey.MEASURED_JOINT_VEL,
            DataKey.MEASURED_EEF_WRENCH,
        ]
        default_action_keys = [DataKey.COMMAND_JOINT_POS]

        state_dim = sum(DataKey.get_dim(key, self.env) for key in default_state_keys)
        action_dim = sum(DataKey.get_dim(key, self.env) for key in default_action_keys)

        state_template = np.zeros(state_dim, dtype=np.float32)
        action_template = np.zeros(action_dim, dtype=np.float32)

        if hasattr(self.env, "camera_names"):
            camera_names = list(self.env.camera_names)
        else:
            camera_names = []

        return {
            "state": {
                "keys": default_state_keys,
                "example": state_template.copy(),
                "mean": state_template.copy(),
                "std": np.ones_like(state_template),
            },
            "action": {
                "keys": default_action_keys,
                "example": action_template.copy(),
                "mean": action_template.copy(),
                "std": np.ones_like(action_template),
            },
            "image": {"camera_names": camera_names},
            "data": {"skip": 1, "n_obs_steps": 1, "n_action_steps": 1},
            "policy": {},
        }

    def setup_policy(self):
        # Print policy information
        self.print_policy_info()
        print(
            f"  - obs steps: {self.model_meta_info['data']['n_obs_steps']}, action steps: {self.model_meta_info['data']['n_action_steps']}"
        )

        state_dict = torch.load(self.args.checkpoint, map_location="cpu")

        if "actor_mean.0.weight" not in state_dict or "actor_logstd" not in state_dict:
            raise KeyError(
                f"[{self.__class__.__name__}] ManiSkill PPO checkpoint does not contain expected keys."
            )

        obs_dim = state_dict["actor_mean.0.weight"].shape[1]
        action_dim_from_ckpt = int(state_dict["actor_logstd"].shape[-1])
        if action_dim_from_ckpt != self.action_dim:
            raise ValueError(
                f"[{self.__class__.__name__}] action dim mismatch: meta={self.action_dim}, checkpoint={action_dim_from_ckpt}"
            )

        self.policy = ManiSkillPpoAgent(obs_dim, self.action_dim)
        self.policy.load_state_dict(state_dict)

        use_cuda = self.args.ppo_use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.policy.to(self.device)
        self.policy.eval()

        self._normalized_action_low = torch.full(
            (self.action_dim,), float(_NORMALIZED_ACTION_LOW.item()), device=self.device
        )
        self._normalized_action_high = torch.full(
            (self.action_dim,), float(_NORMALIZED_ACTION_HIGH.item()), device=self.device
        )

        if self.action_dim != len(_DELTA_PHYSICAL_LOW):
            raise ValueError(
                f"[{self.__class__.__name__}] action dim mismatch for delta bounds: "
                f"meta={self.action_dim}, expected={len(_DELTA_PHYSICAL_LOW)}"
            )

        self._delta_physical_low = _DELTA_PHYSICAL_LOW.to(self.device)
        self._delta_physical_high = _DELTA_PHYSICAL_HIGH.to(self.device)
        self._joint_position_low = _JOINT_POSITION_LOW.to(self.device)
        self._joint_position_high = _JOINT_POSITION_HIGH.to(self.device)

        print(
            f"[{self.__class__.__name__}] Load ManiSkill PPO checkpoint on {self.device}"
        )

        checkpoint_dir = os.path.dirname(os.path.abspath(self.args.checkpoint))
        default_name = f"{self.__class__.__name__.lower()}_debug_log.tsv"
        self._log_path = os.path.join(checkpoint_dir, default_name)
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        with open(self._log_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["step_idx", "obs", "direct_joint_command"])
        print(
            f"[{self.__class__.__name__}] Logging observations and actions to {self._log_path}"
        )

    def setup_plot(self):
        num_cols = max(len(self.camera_names), 1)
        fig_ax = plt.subplots(
            2,
            num_cols,
            figsize=(13.5, 6.0),
            dpi=60,
            squeeze=False,
            constrained_layout=True,
        )
        super().setup_plot(fig_ax)

    def reset_variables(self):
        super().reset_variables()

        self.state_buf = None
        self.images_buf = None
        self.policy_action_buf = None

    def get_state(self):
        # Get latest value
        if len(self.state_keys) == 0:
            state = np.zeros(0, dtype=np.float32)
        else:
            state = np.concatenate(
                [
                    self.motion_manager.get_data(state_key, self.obs)
                    for state_key in self.state_keys
                ]
            )

        qpos = self.motion_manager.get_data(DataKey.MEASURED_JOINT_POS, self.obs)
        qvel = self.motion_manager.get_data(DataKey.MEASURED_JOINT_VEL, self.obs)
        target_qpos = np.array(
            [
                -0.00451699561347621,
                -0.4779577590016519,
                -0.0059982858387227518,
                0.8576097778375098,
                -0.032158957391327556,
                1.27989111575085,
                0.05005294852914137,
                830,
            ],
            dtype=np.float32,
        )
        qpos_ms = qpos.astype(np.float32).copy()
        qpos_ms[-1] = gripper_q_robomanip_to_maniskill(qpos_ms[-1])
        qvel_ms = qvel.astype(np.float32).copy()
        if qvel_ms.size > 0:
            qvel_ms[-1] = gripper_q_robomanip_to_maniskill(qvel_ms[-1])
        target_qpos_ms = target_qpos.copy()
        target_qpos_ms[-1] = gripper_q_robomanip_to_maniskill(target_qpos_ms[-1])

        self.state_for_ppo = np.concatenate([qpos_ms, qvel_ms, target_qpos_ms]).astype(
            np.float32
        )

        norm_state = normalize_data(state, self.model_meta_info["state"])

        state = torch.tensor(norm_state, dtype=torch.float32)

        # Store and return
        if self.state_buf is None:
            self.state_buf = [
                state for _ in range(self.model_meta_info["data"]["n_obs_steps"])
            ]
        else:
            self.state_buf.pop(0)
            self.state_buf.append(state)

        state = torch.stack(self.state_buf, dim=0)[torch.newaxis].to(self.device)

        return state

    def get_images(self):
        # Get latest value
        if len(self.camera_names) == 0:
            return None

        images = []
        for camera_name in self.camera_names:
            image = self.info["rgb_images"][camera_name]

            image = np.moveaxis(image, -1, -3)
            image = torch.tensor(image.copy(), dtype=torch.uint8)
            image = self.image_transforms(image)

            images.append(image)

        # Store and return
        if self.images_buf is None:
            self.images_buf = [
                [image for _ in range(self.model_meta_info["data"]["n_obs_steps"])]
                for image in images
            ]
        else:
            for single_images_buf, image in zip(self.images_buf, images):
                single_images_buf.pop(0)
                single_images_buf.append(image)

        images = torch.stack(
            [
                torch.stack(single_images_buf, dim=0)[torch.newaxis].to(self.device)
                for single_images_buf in self.images_buf
            ]
        )

        return images

    def infer_policy(self):
        # Infer
        if self.policy_action_buf is None or len(self.policy_action_buf) == 0:
            self.get_state()  # update buffers and logs

            obs_tensor = torch.tensor(
                self.state_for_ppo, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            with torch.no_grad():
                raw_action = self.policy.get_action(
                    obs_tensor, deterministic=self.args.ppo_deterministic
                )

            raw_action = raw_action.squeeze(0)
            clipped_action = torch.clamp(
                raw_action, self._normalized_action_low, self._normalized_action_high
            )

            normalized_span = self._normalized_action_high - self._normalized_action_low
            delta_scale = (clipped_action - self._normalized_action_low) / normalized_span
            denormalized_delta = self._delta_physical_low + delta_scale * (
                self._delta_physical_high - self._delta_physical_low
            )

            current_joint_pos = obs_tensor[..., : self.action_dim].squeeze(0)
            direct_joint_command = current_joint_pos + denormalized_delta
            direct_joint_command = torch.max(
                torch.min(direct_joint_command, self._joint_position_high),
                self._joint_position_low,
            )

            if direct_joint_command.numel() > 0:
                direct_joint_command = direct_joint_command.clone()
                direct_joint_command[-1] = gripper_q_maniskill_to_robomanip(
                    direct_joint_command[-1]
                )

            physical_np = direct_joint_command.detach().cpu().numpy().astype(np.float64)

            if hasattr(self, "_log_path") and self._log_path:
                obs_list = (
                    obs_tensor.squeeze(0).detach().cpu().numpy().astype(np.float64).tolist()
                )
                direct_list = physical_np.tolist()
                with open(self._log_path, "a", newline="") as f:
                    writer = csv.writer(f, delimiter="\t")
                    writer.writerow(
                        [
                            int(getattr(self, "rollout_time_idx", 0)),
                            json.dumps(obs_list),
                            json.dumps(direct_list),
                        ]
                    )

            self.policy_action_buf = [physical_np]

        # Store action
        self.policy_action = denormalize_data(
            self.policy_action_buf.pop(0), self.model_meta_info["action"]
        )
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )

    def draw_plot(self):
        # Clear plot
        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        # Plot images
        self.plot_images(self.ax[0, 0 : len(self.camera_names)])

        # Plot action
        self.plot_action(self.ax[1, 0])

        # Finalize plot
        self.canvas.draw()
        cv2.imshow(
            self.policy_name,
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )
