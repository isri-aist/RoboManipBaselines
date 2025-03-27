import argparse
import copy

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from tqdm import tqdm

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
    DiffusionUnetHybridImagePolicy,
)
from robo_manip_baselines.common import DataKey, TrainBase

from .DiffusionPolicyDataset import DiffusionPolicyDataset


class TrainDiffusionPolicy(TrainBase):
    DatasetClass = DiffusionPolicyDataset

    def setup_args(self):
        super().setup_args()

        # Set image size list
        def refine_size_list(size_list):
            if len(size_list) == 2:
                return [tuple(size_list)] * len(self.args.camera_names)
            else:
                assert len(size_list) == len(self.args.camera_names) * 2
                return [
                    (size_list[i], size_list[i + 1])
                    for i in range(0, len(size_list), 2)
                ]

        self.args.image_size_list = refine_size_list(self.args.image_size_list)

    def set_additional_args(self, parser):
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
            help="Enable or disable exponential moving average (EMA)",
        )

        parser.add_argument(
            "--horizon", type=int, default=16, help="prediction horizon"
        )
        parser.add_argument(
            "--n_obs_steps",
            type=int,
            default=2,
            help="number of steps in observation to input in the policy",
        )
        parser.add_argument(
            "--n_action_steps",
            type=int,
            default=8,
            help="number of steps in the action to output from the policy",
        )

        parser.add_argument(
            "--image_size_list",
            type=int,
            nargs="+",
            default=[320, 240],
            help="List of image size (width, height). Specify a 2-dimensional array if all images have the same size, or an array of <number-of-images> * 2 dimensions if the size differs for each individual image.",
        )

    def setup_model_meta_info(self):
        super().setup_model_meta_info()

        self.model_meta_info["data"]["image_size_list"] = self.args.image_size_list
        self.model_meta_info["data"]["horizon"] = self.args.horizon
        self.model_meta_info["data"]["n_obs_steps"] = self.args.n_obs_steps
        self.model_meta_info["data"]["n_action_steps"] = self.args.n_action_steps

        self.model_meta_info["policy"]["use_ema"] = self.args.use_ema

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
        for camera_name, image_size in zip(
            self.args.camera_names, self.args.image_size_list
        ):
            shape_meta["obs"][DataKey.get_rgb_image_key(camera_name)] = {
                "shape": [3, image_size[1], image_size[0]],
                "type": "rgb",
            }
        self.model_meta_info["policy"]["args"] = {
            "shape_meta": shape_meta,
            "horizon": self.args.horizon,
            "n_action_steps": self.args.n_action_steps,
            "n_obs_steps": self.args.n_obs_steps,
            "num_inference_steps": 8,
            "obs_as_global_cond": True,
            "crop_shape": None,
            "diffusion_step_embed_dim": 128,
            "down_dims": [256, 512, 1024],
            "kernel_size": 5,
            "n_groups": 8,
            "cond_predict_scale": True,
            "obs_encoder_group_norm": True,
            "eval_fixed_crop": True,
        }
        self.model_meta_info["policy"]["noise_scheduler_args"] = {
            "num_train_timesteps": 100,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "squaredcos_cap_v2",
            "clip_sample": True,
            "set_alpha_to_one": True,
            "steps_offset": 0,
            "prediction_type": "epsilon",
        }

        # Construct policy
        noise_scheduler = DDIMScheduler(
            **self.model_meta_info["policy"]["noise_scheduler_args"]
        )
        self.policy = DiffusionUnetHybridImagePolicy(
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
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.95, 0.999),
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
        print("!!!! eval-dp-ddim branch !!!")
        self.print_policy_info()
        print(f"  - use ema: {self.args.use_ema}")
        print(
            f"  - horizon: {self.args.horizon}, obs steps: {self.args.n_obs_steps}, action steps: {self.args.n_action_steps}"
        )
        print(f"  - image size list: {self.args.image_size_list}")

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
