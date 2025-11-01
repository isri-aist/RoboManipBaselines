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
        "../../../third_party/FlowPolicy/FlowPolicy",
    )
)
from flow_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from flow_policy_3d.model.common.lr_scheduler import get_scheduler
from flow_policy_3d.model.flow.ema_model import EMAModel
from flow_policy_3d.policy.flowpolicy import FlowPolicy

from robo_manip_baselines.common import (
    TrainBase,
    TrainPointCloudMixin,
)

from .FlowPolicyDataset import FlowPolicyDataset


class TrainFlowPolicy(TrainBase, TrainPointCloudMixin):
    DatasetClass = FlowPolicyDataset

    def set_additional_args(self, parser):
        for action in parser._actions:
            if action.dest == "camera_names":
                action.nargs = 1

        parser.set_defaults(enable_rmb_cache=True)

        parser.set_defaults(norm_type="limits")

        parser.set_defaults(batch_size=128)
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

        parser.add_argument("--horizon", type=int, default=8, help="prediction horizon")
        parser.add_argument(
            "--n_obs_steps",
            type=int,
            default=2,
            help="number of steps in the observation sequence to input in the policy",
        )
        parser.add_argument(
            "--n_action_steps",
            type=int,
            default=4,
            help="number of steps in the action sequence to output from the policy",
        )

        parser.add_argument(
            "--use_pc_color",
            action="store_true",
            help="Whether to use color information of point cloud",
        )

        parser.add_argument(
            "--encoder_output_dim",
            type=int,
            nargs=1,
            default=64,
            help="output dimensions of encoder in policy",
        )

    def setup_model_meta_info(self):
        super().setup_model_meta_info()

        self.model_meta_info["data"]["horizon"] = self.args.horizon
        self.model_meta_info["data"]["n_obs_steps"] = self.args.n_obs_steps
        self.model_meta_info["data"]["n_action_steps"] = self.args.n_action_steps
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

    def set_data_stats(self):
        super().set_data_stats()

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
        # Set policy args
        shape_meta = OmegaConf.create(
            {
                "obs": {},
                "action": {"shape": [len(self.model_meta_info["action"]["example"])]},
            }
        )
        if len(self.args.state_keys) > 0:
            shape_meta["obs"]["state"] = {
                "shape": [len(self.model_meta_info["state"]["example"])],
                "type": "low_dim",
            }
        point_dim = 6 if self.args.use_pc_color else 3
        pointcloud_shape = (
            self.model_meta_info["data"]["num_points"],
            point_dim,
        )
        shape_meta["obs"]["point_cloud"] = {
            "shape": pointcloud_shape,
            "type": "point_cloud",
        }
        pointcloud_encoder_conf = OmegaConf.create(
            {
                "in_channels": point_dim,
                "out_channels": self.args.encoder_output_dim,
                "use_layernorm": True,
                "final_norm": "layernorm",
                "normal_channel": False,
            }
        )
        flow_match_conf = OmegaConf.create(
            {
                "eps": 1e-2,
                "num_segments": 2,
                "boundary": 1,
                "delta": 1e-2,
                "alpha": 1e-5,
                "num_inference_step": 1,
            }
        )
        self.model_meta_info["policy"]["args"] = {
            "shape_meta": shape_meta,
            "horizon": self.args.horizon,
            "n_action_steps": self.args.n_action_steps,
            "n_obs_steps": self.args.n_obs_steps,
            "num_inference_steps": 10,
            "obs_as_global_cond": True,
            "diffusion_step_embed_dim": 128,
            "down_dims": [512, 1024, 2048],
            "kernel_size": 5,
            "n_groups": 8,
            "condition_type": "film",
            "use_down_condition": True,
            "use_mid_condition": True,
            "use_up_condition": True,
            "encoder_output_dim": self.args.encoder_output_dim,
            "use_pc_color": self.args.use_pc_color,
            "pointnet_type": "mlp",
            "pointcloud_encoder_cfg": pointcloud_encoder_conf,
            "Conditional_ConsistencyFM": flow_match_conf,
        }

        # Construct policy
        self.policy = FlowPolicy(
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
        self.print_policy_info()
        print(f"  - use ema: {self.args.use_ema}")
        print(
            f"  - horizon: {self.args.horizon}, obs steps: {self.args.n_obs_steps}, action steps: {self.args.n_action_steps}"
        )
        data_info = self.model_meta_info["data"]
        print(
            f"  - with color: {self.args.use_pc_color}, num points: {data_info['num_points']}, image size: {data_info['image_size']}, min bound: {data_info['min_bound']}, max bound: {data_info['max_bound']}, rpy_angle: {data_info['rpy_angle']}"
        )

    def train_loop(self):
        for epoch in tqdm(range(self.args.num_epochs)):
            # Run train step
            batch_result_list = []
            for data in self.train_dataloader:
                loss, _ = self.policy.compute_loss(dict_apply(data, lambda x: x.cuda()))
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
                    loss, _ = policy.compute_loss(dict_apply(data, lambda x: x.cuda()))
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
