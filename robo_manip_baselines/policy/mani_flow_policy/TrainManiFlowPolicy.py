import argparse
import copy
import os
import sys

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../../../third_party/ManiFlow_Policy/ManiFlow",
    )
)

import torchvision.transforms as v2
from maniflow.common.pytorch_util import dict_apply, optimizer_to
from maniflow.model.common.lr_scheduler import get_scheduler
from maniflow.model.diffusion.ema_model import EMAModel
from maniflow.model.vision_2d.timm_obs_encoder import TimmObsEncoder

from robo_manip_baselines.common import (
    DataKey,
    TrainBase,
    TrainPointCloudMixin,
)

from .ManiFlowImageDataset import ManiFlowImageDataset
from .ManiFlowPointcloudDataset import ManiFlowPointcloudDataset


class TrainManiFlowPolicy(TrainBase, TrainPointCloudMixin):
    DatasetClass = None

    def setup_args(self):
        super().setup_args()

        # Set default value depending on policy_type
        if self.args.batch_size is None:
            if self.args.policy_type == "image":
                self.args.batch_size = 128
            else:  # if self.args.policy_type == "pointcloud":
                self.args.batch_size = 256

        if self.args.num_epochs is None:
            if self.args.policy_type == "image":
                self.args.num_epochs = 500
            else:  # if self.args.policy_type == "pointcloud":
                self.args.num_epochs = 1000

    def set_additional_args(self, parser):
        parser.set_defaults(enable_rmb_cache=True)

        parser.set_defaults(norm_type="limits")

        parser.set_defaults(batch_size=None)  # depends on policy_type
        parser.set_defaults(num_epochs=None)  # depends on policy_type
        parser.set_defaults(lr=1e-4)

        parser.add_argument(
            "policy_type",
            type=str,
            choices=["image", "pointcloud"],
            help="Input format for vision data (2D/3D)",
        )

        parser.add_argument(
            "--weight_decay", type=float, default=1e-3, help="weight decay"
        )

        parser.add_argument(
            "--use_ema",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable or disable exponential moving average (EMA)",
        )

        parser.add_argument(
            "--horizon", type=int, default=16, help="prediction horizon"
        )
        parser.add_argument(
            "--n_obs_steps",
            type=int,
            default=2,
            help="number of steps in the observation sequence to input in the policy",
        )
        parser.add_argument(
            "--n_action_steps",
            type=int,
            default=16,
            help="number of steps in the action sequence to output from the policy",
        )

        parser.add_argument(
            "--flow_batch_ratio",
            type=float,
            default=0.75,
            help="ratio of data for flow in batch",
        )

        parser.add_argument(
            "--use_pc_color",
            action="store_true",
            help="Whether to use color information (for pointcloud policy)",
        )

        parser.add_argument(
            "--image_size",
            type=int,
            nargs=2,
            default=[224, 224],
            help="image size (for image policy)",
        )

    def setup_model_meta_info(self):
        super().setup_model_meta_info()

        self.model_meta_info["data"]["horizon"] = self.args.horizon
        self.model_meta_info["data"]["n_obs_steps"] = self.args.n_obs_steps
        self.model_meta_info["data"]["n_action_steps"] = self.args.n_action_steps

        if self.args.policy_type == "image":
            self.model_meta_info["data"]["image_size"] = self.args.image_size
        else:  # if self.args.policy_type == "pointcloud":
            self.model_meta_info["data"]["use_pc_color"] = self.args.use_pc_color
            num_points, image_size, min_bound, max_bound, rpy_angle = (
                self.setup_pointcloud_info()
            )
            self.model_meta_info["data"]["num_points"] = num_points
            self.model_meta_info["data"]["image_size"] = image_size
            self.model_meta_info["data"]["min_bound"] = min_bound
            self.model_meta_info["data"]["max_bound"] = max_bound
            self.model_meta_info["data"]["rpy_angle"] = rpy_angle

        self.model_meta_info["policy"]["use_ema"] = self.args.use_ema
        self.model_meta_info["policy"]["policy_type"] = self.args.policy_type

    def setup_dataset(self):
        if self.args.policy_type == "image":
            self.DatasetClass = ManiFlowImageDataset
        else:  # if self.args.policy_type == "pointcloud":
            self.DatasetClass = ManiFlowPointcloudDataset
        return super().setup_dataset()

    def set_data_stats(self):
        super().set_data_stats()
        if self.args.policy_type == "pointcloud":
            self.set_pointcloud_stats()

    def get_extra_norm_config(self):
        if self.args.norm_type == "limits":
            return {
                "out_min": -1.0,
                "out_max": 1.0,
            }
        else:
            return super().get_extra_norm_config()

    def setup_policy(self):
        # Set meta shape
        shape_meta = OmegaConf.create(
            {
                "obs": {},
                "action": {
                    "shape": [len(self.model_meta_info["action"]["example"])],
                    "horizon": self.args.horizon,
                },
            }
        )
        if len(self.args.state_keys) > 0:
            shape_meta["obs"]["state"] = {
                "shape": [len(self.model_meta_info["state"]["example"])],
                "type": "low_dim",
                "horizon": self.args.n_obs_steps,
            }
        if self.args.policy_type == "image":
            for camera_name in self.model_meta_info["image"]["camera_names"]:
                shape_meta["obs"][DataKey.get_rgb_image_key(camera_name)] = {
                    "shape": [3, self.args.image_size[1], self.args.image_size[0]],
                    "type": "rgb",
                    "horizon": self.args.n_obs_steps,
                }
        else:  # if self.args.policy_type == "pointcloud":
            point_dim = 6 if self.args.use_pc_color else 3
            pointcloud_shape = (
                self.model_meta_info["data"]["num_points"],
                point_dim,
            )
            shape_meta["obs"]["point_cloud"] = {
                "shape": pointcloud_shape,
                "type": "point_cloud",
            }

        # Set policy args
        policy_args = {
            "shape_meta": shape_meta,
            "horizon": self.args.horizon,
            "n_action_steps": self.args.n_action_steps,
            "n_obs_steps": self.args.n_obs_steps,
            "num_inference_steps": 10,
            "obs_as_global_cond": True,
            "block_type": "DiTX",
            "n_layer": 12,
            "n_head": 8,
            "n_emb": 768,
            "max_lang_cond_len": 1024,
            "qkv_bias": True,
            "qk_norm": True,
            "language_conditioned": False,
            "pre_norm_modality": False,
            "flow_batch_ratio": self.args.flow_batch_ratio,
            "consistency_batch_ratio": 1.0 - self.args.flow_batch_ratio,
            "sample_t_mode_flow": "beta",
            "sample_t_mode_consistency": "discrete",
            "sample_dt_mode_consistency": "uniform",
            "sample_target_t_mode": "relative",  # "absolute", "relative"
            "denoise_timesteps": 10,
            "diffusion_timestep_embed_dim": 128,
            "diffusion_target_t_embed_dim": 128,
        }
        if self.args.policy_type == "image":
            from maniflow.policy.maniflow_image_policy import (
                ManiFlowTransformerImagePolicy,
            )

            obs_encoder_transforms = [
                v2.RandomCrop(size=int(0.95 * self.args.image_size[0])),
                v2.Resize(size=self.args.image_size[0], antialias=True),
                v2.RandomRotation(degrees=[-5.0, 5.0]),
                v2.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
            ]
            obs_encoder_conf = {
                "shape_meta": shape_meta,
                "model_name": "r3m",
                "pretrained": False,
                "frozen": False,
                "global_pool": "",
                "feature_aggregation": None,
                "position_encording": "sinusoidal",
                "downsample_ratio": 32,
                "use_group_norm": True,
                "share_rgb_model": False,
                "imagenet_norm": True,
                "transforms": obs_encoder_transforms,
            }
            obs_encoder = TimmObsEncoder(**obs_encoder_conf)
            policy_args.update(
                {
                    "visual_cond_len": 1024,
                    "obs_encoder": obs_encoder,
                }
            )
            PolicyClass = ManiFlowTransformerImagePolicy
        else:  # if self.args.policy_type == "pointcloud":
            from maniflow.policy.maniflow_pointcloud_policy import (
                ManiFlowTransformerPointcloudPolicy,
            )

            encoder_output_dim = 128
            visual_cond_len = 128
            pointcloud_encoder_conf = OmegaConf.create(
                {
                    "in_channels": point_dim,
                    "out_channels": encoder_output_dim,
                    "use_layernorm": True,
                    "final_norm": "layernorm",
                    "normal_channel": False,
                    "num_points": visual_cond_len,
                    "pointwise": True,
                }
            )
            policy_args.update(
                {
                    "visual_cond_len": visual_cond_len,
                    "use_point_crop": True,
                    "crop_shape": [80, 80],
                    "encoder_type": "DP3Encoder",
                    "encoder_output_dim": encoder_output_dim,
                    "use_pc_color": self.args.use_pc_color,
                    "pointnet_type": "pointnet",
                    "downsample_points": True,
                    "pointcloud_encoder_cfg": pointcloud_encoder_conf,
                }
            )
            PolicyClass = ManiFlowTransformerPointcloudPolicy

        self.model_meta_info["policy"]["args"] = policy_args

        # Construct policy
        self.policy = PolicyClass(
            **self.model_meta_info["policy"]["args"],
        )

        # Construct exponential moving average (EMA)
        if self.args.use_ema:
            self.ema_policy = copy.deepcopy(self.policy)
            self.ema = EMAModel(
                model=self.ema_policy,
                update_after_step=0,
                inv_gamma=1.0,
                power=0.75,
                min_value=0.0,
                max_value=0.9999,
            )

        # Construct optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=(len(self.train_dataloader) * self.args.num_epochs),
        )

        # Transfer to device
        self.policy.cuda()
        if self.args.use_ema:
            self.ema_policy.cuda()
        optimizer_to(self.optimizer, "cuda")

        # Print policy information
        self.print_policy_info()
        print(f"  - policy type: {self.args.policy_type}")
        print(f"  - use ema: {self.args.use_ema}")
        print(
            f"  - horizon: {self.args.horizon}, obs steps: {self.args.n_obs_steps}, action steps: {self.args.n_action_steps}"
        )
        data_info = self.model_meta_info["data"]
        if self.args.policy_type == "image":
            print(f"  - image size: {data_info['image_size']}")
        else:  # if self.args.policy_type == "pointcloud":
            print(
                f"  - with color: {self.args.use_pc_color}, num points: {data_info['num_points']}, image size: {data_info['image_size']}, min bound: {data_info['min_bound']}, max bound: {data_info['max_bound']}, rpy_angle: {data_info['rpy_angle']}"
            )

    def train_loop(self):
        ema_model = self.ema_policy if self.args.use_ema else None
        for epoch in tqdm(range(self.args.num_epochs)):
            # Run train step
            batch_result_list = []
            for data in self.train_dataloader:
                loss, _ = self.policy.compute_loss(
                    dict_apply(data, lambda x: x.cuda()), ema_model=ema_model
                )
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                if self.args.use_ema:
                    self.ema.step(self.policy)
                batch_result_list.append(
                    self.detach_batch_result(
                        {"loss": loss, "lr": self.lr_scheduler.get_last_lr()[0]}
                    )
                )
            self.log_epoch_summary(batch_result_list, "train", epoch)

            # Run validation step
            if self.args.use_ema:
                policy = self.ema_policy
            else:
                policy = self.policy
            policy.eval()
            with torch.inference_mode():
                batch_result_list = []
                for data in self.val_dataloader:
                    loss, _ = policy.compute_loss(
                        dict_apply(data, lambda x: x.cuda()), ema_model=ema_model
                    )
                    batch_result_list.append(self.detach_batch_result({"loss": loss}))
                epoch_summary = self.log_epoch_summary(batch_result_list, "val", epoch)

                # Update best checkpoint
                self.update_best_ckpt(epoch_summary, policy=policy)
            policy.train()

            # Save current checkpoint
            if epoch % max(self.args.num_epochs // 10, 1) == 0:
                self.save_current_ckpt(f"epoch{epoch:0>4}", policy=policy)

        # Save last checkpoint
        self.save_current_ckpt("last", policy=policy)

        # Save best checkpoint
        self.save_best_ckpt()
