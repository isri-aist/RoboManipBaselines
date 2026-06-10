import torch
from tqdm import tqdm

from robo_manip_baselines.common import RmbData, TrainBase, find_rmb_files

from .AsyncVlaDataset import AsyncVlaDataset
from .AsyncVlaPolicy import AsyncVlaPolicy


class TrainAsyncVla(TrainBase):
    """Train the AsyncVLA Edge Adapter (paper Stage-1, base VLA frozen).

    The frozen base VLA (pi0) is not loaded here: its guidance is precomputed
    offline by ``misc/AddPi0GuidanceToRmbData.py`` and cached in the dataset.
    """

    DatasetClass = AsyncVlaDataset

    def set_additional_args(self, parser):
        parser.set_defaults(enable_rmb_cache=True)
        parser.set_defaults(skip=1)
        parser.set_defaults(batch_size=64)
        parser.set_defaults(num_epochs=200)
        parser.set_defaults(lr=1e-4)

        parser.add_argument(
            "--weight_decay", type=float, default=1e-4, help="weight decay"
        )

        parser.add_argument(
            "--n_action_steps",
            type=int,
            default=8,
            help="number of steps in the action chunk (should match the base VLA)",
        )
        parser.add_argument(
            "--d_model",
            type=int,
            default=512,
            help="transformer token / model dimension (official obs_encoding_size)",
        )
        parser.add_argument(
            "--n_heads",
            type=int,
            default=8,
            help="number of attention heads in the Edge Adapter transformer",
        )
        parser.add_argument(
            "--n_layers",
            type=int,
            default=4,
            help="number of transformer encoder layers in the Edge Adapter",
        )
        parser.add_argument(
            "--ff_dim_factor",
            type=int,
            default=4,
            help="feed-forward expansion factor in the transformer",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            default=0.1,
            help="dropout in the Edge Adapter transformer encoder",
        )
        parser.add_argument(
            "--image_size",
            type=int,
            default=96,
            help="image size fed to the EfficientNet encoders (official 96)",
        )

        parser.add_argument(
            "--delay_min",
            type=int,
            default=1,
            help="minimum injected base-VLA delay [step]",
        )
        parser.add_argument(
            "--delay_max",
            type=int,
            default=6,
            help="maximum injected base-VLA delay [step] (match rollout staleness)",
        )
        parser.add_argument(
            "--dth",
            type=float,
            default=0.1,
            help="reweighting threshold in normalized action space",
        )
        parser.add_argument(
            "--reweight_gain",
            type=float,
            default=1.0,
            help="up-weighting gain for delay-sensitive samples",
        )
        parser.add_argument(
            "--smooth_weight",
            type=float,
            default=0.1,
            help="weight of the smoothness loss",
        )
        parser.add_argument(
            "--guidance_key",
            type=str,
            default="pi0_guidance",
            help="key of the cached base-VLA guidance in the RMB data",
        )

    def setup_model_meta_info(self):
        super().setup_model_meta_info()

        if self.args.use_cached_dataset:
            raise ValueError(
                f"[{self.__class__.__name__}] '--use_cached_dataset' must be disabled "
                "because the dataset uses per-sample random delay injection and reweighting."
            )

        # The relative-action and smoothness losses difference consecutive steps, so a
        # single-step chunk yields an empty time axis and NaN losses (corrupting the
        # checkpoint). Require at least two steps.
        if self.args.n_action_steps < 2:
            raise ValueError(
                f"[{self.__class__.__name__}] --n_action_steps must be >= 2 "
                f"(got {self.args.n_action_steps}); the relative/smoothness losses "
                "require at least two steps."
            )

        self.model_meta_info["data"]["n_obs_steps"] = 1
        self.model_meta_info["data"]["n_action_steps"] = self.args.n_action_steps
        self.model_meta_info["data"]["delay_min"] = self.args.delay_min
        self.model_meta_info["data"]["delay_max"] = self.args.delay_max
        self.model_meta_info["data"]["dth"] = self.args.dth
        self.model_meta_info["data"]["reweight_gain"] = self.args.reweight_gain
        self.model_meta_info["data"]["guidance_key"] = self.args.guidance_key
        self.model_meta_info["train"]["smooth_weight"] = self.args.smooth_weight

        # Read the pi0 guidance embedding dims from the cached data so the Edge
        # Adapter transformer can be sized before the policy is constructed.
        guidance_key = self.args.guidance_key
        rmb_files = find_rmb_files(self.args.dataset_dir)
        if len(rmb_files) == 0:
            raise ValueError(
                f"[{self.__class__.__name__}] No RMB files found in "
                f"{self.args.dataset_dir}."
            )
        with RmbData(rmb_files[0]) as rmb_data:
            if guidance_key not in rmb_data.keys():
                raise ValueError(
                    f"[{self.__class__.__name__}] Cached guidance '{guidance_key}' not "
                    f"found in {rmb_files[0]}. Run misc/AddPi0GuidanceToRmbData.py first."
                )
            # Reject stale caches from the pre-port action-chunk implementation. That
            # stored a (T, n_action_steps, action_dim) action chunk under the SAME
            # default key; it has the same ndim as the new (T, n_guidance_tokens,
            # embed_dim) hidden-state guidance, so without this check it would silently
            # size the Edge Adapter to e.g. (8, 7). Training would then finish but
            # rollout would reject the checkpoint against the live pi0 chunk_size /
            # expert_width. The faithful-port cache (AddPi0GuidanceToRmbData) writes
            # these metadata attrs alongside the data, so require them and fail fast.
            tokens_attr = guidance_key + "_n_guidance_tokens"
            embed_attr = guidance_key + "_embed_dim"
            if tokens_attr not in rmb_data.attrs or embed_attr not in rmb_data.attrs:
                raise ValueError(
                    f"[{self.__class__.__name__}] Cached guidance '{guidance_key}' in "
                    f"{rmb_files[0]} is missing the '{tokens_attr}' / '{embed_attr}' "
                    "attributes written by the hidden-state guidance cache. It was "
                    "likely produced by an older action-chunk implementation. Re-run "
                    "misc/AddPi0GuidanceToRmbData.py with --overwrite to regenerate it."
                )
            # (T, n_guidance_tokens, embed_dim)
            guidance_shape = rmb_data[guidance_key].shape
            attr_tokens = int(rmb_data.attrs[tokens_attr])
            attr_embed = int(rmb_data.attrs[embed_attr])
            if (
                len(guidance_shape) != 3
                or int(guidance_shape[1]) != attr_tokens
                or int(guidance_shape[2]) != attr_embed
            ):
                raise ValueError(
                    f"[{self.__class__.__name__}] Cached guidance '{guidance_key}' shape "
                    f"{tuple(guidance_shape)} is inconsistent with its metadata "
                    f"(n_guidance_tokens={attr_tokens}, embed_dim={attr_embed}) in "
                    f"{rmb_files[0]}. Re-run misc/AddPi0GuidanceToRmbData.py --overwrite."
                )
        self.model_meta_info["data"]["n_guidance_tokens"] = int(guidance_shape[1])
        self.model_meta_info["data"]["guidance_embed_dim"] = int(guidance_shape[2])

    def setup_policy(self):
        # Set policy args (faithful transformer Edge Adapter)
        self.model_meta_info["policy"]["args"] = {
            "n_action_steps": self.args.n_action_steps,
            "guidance_embed_dim": self.model_meta_info["data"]["guidance_embed_dim"],
            "n_guidance_tokens": self.model_meta_info["data"]["n_guidance_tokens"],
            "d_model": self.args.d_model,
            "n_heads": self.args.n_heads,
            "n_layers": self.args.n_layers,
            "ff_dim_factor": self.args.ff_dim_factor,
            "dropout": self.args.dropout,
            "image_size": self.args.image_size,
        }

        # Construct policy
        self.policy = AsyncVlaPolicy(
            len(self.model_meta_info["state"]["example"]),
            len(self.model_meta_info["action"]["example"]),
            len(self.args.camera_names),
            **self.model_meta_info["policy"]["args"],
        )
        self.policy.cuda()

        # Construct optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        # Print policy information
        self.print_policy_info()
        print(
            f"  - action steps: {self.args.n_action_steps}, "
            f"delay: [{self.args.delay_min}, {self.args.delay_max}], "
            f"smooth weight: {self.args.smooth_weight}"
        )

    def calc_loss(self, data):
        state, images, delta_image, guidance, gt_action, weight = [
            d.cuda() for d in data
        ]
        pred_action = self.policy(state, images, delta_image, guidance)

        # Imitation loss (absolute + relative), reweighted per sample
        abs_loss = ((pred_action - gt_action) ** 2).mean(dim=(1, 2))
        pred_rel = pred_action[:, 1:] - pred_action[:, :-1]
        gt_rel = gt_action[:, 1:] - gt_action[:, :-1]
        rel_loss = ((pred_rel - gt_rel) ** 2).mean(dim=(1, 2))
        imitation_loss = (weight * (abs_loss + rel_loss)).mean()

        # Smoothness loss on the predicted chunk
        smooth_loss = (pred_rel**2).mean()

        loss = (
            imitation_loss
            + self.model_meta_info["train"]["smooth_weight"] * smooth_loss
        )
        return {
            "loss": loss,
            "imitation": imitation_loss,
            "smooth": smooth_loss,
        }

    def train_loop(self):
        for epoch in tqdm(range(self.args.num_epochs)):
            # Run train step
            self.policy.train()
            batch_result_list = []
            for data in self.train_dataloader:
                self.optimizer.zero_grad()
                batch_result = self.calc_loss(data)
                batch_result["loss"].backward()
                self.optimizer.step()
                batch_result_list.append(self.detach_batch_result(batch_result))
            self.log_epoch_summary(batch_result_list, "train", epoch)

            # Run validation step
            with torch.inference_mode():
                self.policy.eval()
                batch_result_list = []
                for data in self.val_dataloader:
                    batch_result = self.calc_loss(data)
                    batch_result_list.append(self.detach_batch_result(batch_result))
                epoch_summary = self.log_epoch_summary(batch_result_list, "val", epoch)

                # Update best checkpoint
                self.update_best_ckpt(epoch_summary)

            # Save current checkpoint
            if epoch % max(self.args.num_epochs // 10, 1) == 0:
                self.save_current_ckpt(f"epoch{epoch:0>3}")

        # Save last checkpoint
        self.save_current_ckpt("last")

        # Save best checkpoint
        self.save_best_ckpt()
