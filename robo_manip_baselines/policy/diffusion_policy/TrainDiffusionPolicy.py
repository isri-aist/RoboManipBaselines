import argparse
import copy
import os
import sys

import torch
from tqdm import tqdm

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../third_party/diffusion_policy")
)
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from robo_manip_baselines.common import DataKey, TrainBase

from .DiffusionPolicyDataset import DiffusionPolicyDataset


class TrainDiffusionPolicy(TrainBase):
    DatasetClass = DiffusionPolicyDataset

    def setup_args(self):
        super().setup_args()

        if self.args.backbone not in ("cnn", "transformer"):
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid backbone: {self.args.backbone}"
            )

        if self.args.scheduler not in ("ddpm", "ddim"):
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid scheduler: {self.args.scheduler}"
            )

        if self.args.backbone == "transformer" and self.args.scheduler == "ddim":
            raise ValueError(
                f"[{self.__class__.__name__}] The transformer backbone and ddim scheduler cannot be used simultaneously."
            )

        if self.args.horizon is None:
            if self.args.backbone == "cnn":
                self.args.horizon = 16
            else:  # if self.args.backbone == "transformer"
                self.args.horizon = 10

    def set_additional_args(self, parser):
        parser.set_defaults(enable_rmb_cache=True)

        parser.set_defaults(norm_type="limits")

        parser.set_defaults(batch_size=64)
        parser.set_defaults(num_epochs=2000)
        parser.set_defaults(lr=1e-4)

        parser.add_argument(
            "--weight_decay", type=float, default=1e-6, help="weight decay"
        )

        parser.add_argument(
            "--use_ema",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="enable or disable exponential moving average (EMA)",
        )
        parser.add_argument(
            "--backbone",
            type=str,
            default="cnn",
            choices=["cnn", "transformer"],
            help="type of model backbone ('cnn' or 'transformer')",
        )
        parser.add_argument(
            "--scheduler",
            type=str,
            default="ddpm",
            choices=["ddpm", "ddim"],
            help="type of noise scheduler ('ddpm' or 'ddim')",
        )

        parser.add_argument(
            "--horizon", type=int, default=None, help="prediction horizon"
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
            default=8,
            help="number of steps in the action sequence to output from the policy",
        )

        parser.add_argument(
            "--image_size",
            type=int,
            nargs=2,
            default=[320, 240],
            help="Image size (width, height) to be resized before crop. In the case of multiple image inputs, it is assumed that all images share the same size.",
        )
        parser.add_argument(
            "--image_crop_size",
            type=int,
            nargs=2,
            default=[288, 216],
            help="Image size (width, height) to be cropped after resize. In the case of multiple image inputs, it is assumed that all images share the same size.",
        )

    def setup_model_meta_info(self):
        super().setup_model_meta_info()

        self.model_meta_info["data"]["image_size"] = self.args.image_size
        self.model_meta_info["data"]["image_crop_size"] = self.args.image_crop_size

        self.model_meta_info["data"]["horizon"] = self.args.horizon
        self.model_meta_info["data"]["n_obs_steps"] = self.args.n_obs_steps
        self.model_meta_info["data"]["n_action_steps"] = self.args.n_action_steps

        self.model_meta_info["policy"]["use_ema"] = self.args.use_ema
        self.model_meta_info["policy"]["backbone"] = self.args.backbone
        self.model_meta_info["policy"]["scheduler"] = self.args.scheduler

    def get_extra_norm_config(self):
        if self.args.norm_type == "limits":
            return {
                "out_min": -1.0,
                "out_max": 1.0,
            }
        else:
            return super().get_extra_norm_config()

    def setup_policy(self):
        # Set policy args
        shape_meta = {
            "obs": {},
            "action": {"shape": [len(self.model_meta_info["action"]["example"])]},
        }
        if len(self.args.state_keys) > 0:
            shape_meta["obs"]["state"] = {
                "shape": [len(self.model_meta_info["state"]["example"])],
                "type": "low_dim",
            }
        for camera_name in self.args.camera_names:
            shape_meta["obs"][DataKey.get_rgb_image_key(camera_name)] = {
                "shape": [3, self.args.image_size[1], self.args.image_size[0]],
                "type": "rgb",
            }
        self.model_meta_info["policy"]["args"] = {
            "shape_meta": shape_meta,
            "horizon": self.args.horizon,
            "n_action_steps": self.args.n_action_steps,
            "n_obs_steps": self.args.n_obs_steps,
            "crop_shape": self.args.image_crop_size[::-1],  # (height, width)
            "obs_encoder_group_norm": True,
            "eval_fixed_crop": True,
        }
        if self.args.backbone == "cnn":
            self.model_meta_info["policy"]["args"].update(
                {
                    "num_inference_steps": 100,
                    "down_dims": [512, 1024, 2048],
                    "obs_as_global_cond": True,
                    "diffusion_step_embed_dim": 128,
                    "kernel_size": 5,
                    "n_groups": 8,
                    "cond_predict_scale": True,
                }
            )
        else:  # if self.args.backbone == "transformer"
            self.model_meta_info["policy"]["args"].update(
                {
                    "n_layer": 8,
                    "n_cond_layers": 0,
                    "n_head": 4,
                    "n_emb": 256,
                    "p_drop_emb": 0.0,
                    "p_drop_attn": 0.3,
                    "causal_attn": True,
                    "time_as_cond": True,
                    "obs_as_cond": True,
                }
            )
        self.model_meta_info["policy"]["noise_scheduler_args"] = {
            "beta_end": 0.02,
            "beta_schedule": "squaredcos_cap_v2",
            "beta_start": 0.0001,
            "clip_sample": True,
            "num_train_timesteps": 100,
            "prediction_type": "epsilon",
        }

        # Construct scheduler
        if self.model_meta_info["policy"]["scheduler"] == "ddpm":
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

            self.model_meta_info["policy"]["noise_scheduler_args"].update(
                {
                    "variance_type": "fixed_small",
                }
            )
            noise_scheduler = DDPMScheduler(
                **self.model_meta_info["policy"]["noise_scheduler_args"]
            )
        else:  # if self.model_meta_info["policy"]["scheduler"] == "ddim"
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler

            self.model_meta_info["policy"]["args"].update(
                {
                    "num_inference_steps": 8,
                    "down_dims": [256, 512, 1024],
                }
            )
            self.model_meta_info["policy"]["noise_scheduler_args"].update(
                {
                    "set_alpha_to_one": True,
                    "steps_offset": 0,
                }
            )
            noise_scheduler = DDIMScheduler(
                **self.model_meta_info["policy"]["noise_scheduler_args"]
            )

        # Construct policy
        if self.args.backbone == "cnn":
            from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
                DiffusionUnetHybridImagePolicy,
            )

            PolicyClass = DiffusionUnetHybridImagePolicy
        else:  # if self.args.backbone == "transformer"
            from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import (
                DiffusionTransformerHybridImagePolicy,
            )

            PolicyClass = DiffusionTransformerHybridImagePolicy
        self.policy = PolicyClass(
            noise_scheduler=noise_scheduler,
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
        if self.args.backbone == "cnn":
            self.optimizer = torch.optim.AdamW(
                self.policy.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
                betas=(0.95, 0.999),
                eps=1e-8,
            )
        else:  # if self.args.backbone == "transformer"
            self.optimizer = self.policy.get_optimizer(
                transformer_weight_decay=1e-3,
                obs_encoder_weight_decay=1e-6,
                learning_rate=self.args.lr,
                betas=(0.9, 0.95),
            )

        if self.args.backbone == "cnn":
            num_warmup_steps = 500
        else:  # if self.args.backbone == "transformer"
            num_warmup_steps = 1000
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=(len(self.train_dataloader) * self.args.num_epochs),
        )

        # Transfer to device
        self.policy.cuda()
        if self.args.use_ema:
            self.ema_policy.cuda()
        optimizer_to(self.optimizer, "cuda")

        # Print policy information
        self.print_policy_info()
        print(
            f"  - use ema: {self.args.use_ema}, backbone: {self.args.backbone}, scheduler: {self.args.scheduler}"
        )
        print(
            f"  - horizon: {self.args.horizon}, obs steps: {self.args.n_obs_steps}, action steps: {self.args.n_action_steps}"
        )
        print(
            f"  - image size: {self.args.image_size}, image crop size: {self.args.image_crop_size}"
        )

    def train_loop(self):
        for epoch in tqdm(range(self.args.num_epochs)):
            # Run train step
            batch_result_list = []
            for data in self.train_dataloader:
                loss = self.policy.compute_loss(dict_apply(data, lambda x: x.cuda()))
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
                    loss = policy.compute_loss(dict_apply(data, lambda x: x.cuda()))
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
