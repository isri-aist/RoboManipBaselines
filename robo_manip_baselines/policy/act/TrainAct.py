import os
import sys

import torch
from tqdm import tqdm
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third_party/act"))
from detr.models.detr_vae import DETRVAE
from policy import ACTPolicy

from robo_manip_baselines.common import TrainBase

from .ActDataset import ActDataset


class TrainAct(TrainBase):
    DatasetClass = ActDataset

    def set_additional_args(self, parser):
        parser.set_defaults(enable_rmb_cache=True)

        parser.set_defaults(image_aug_std=0.1)

        parser.set_defaults(batch_size=8)
        parser.set_defaults(num_epochs=1000)
        parser.set_defaults(lr=1e-5)

        parser.add_argument("--kl_weight", type=int, default=10, help="KL weight")
        parser.add_argument(
            "--chunk_size", type=int, default=100, help="action chunking size"
        )
        parser.add_argument(
            "--hidden_dim", type=int, default=512, help="hidden dimension"
        )
        parser.add_argument(
            "--dim_feedforward", type=int, default=3200, help="feedforward dimension"
        )

    def setup_model_meta_info(self):
        super().setup_model_meta_info()

        self.model_meta_info["data"]["chunk_size"] = self.args.chunk_size

    def setup_policy(self):
        # Set policy args
        self.model_meta_info["policy"]["args"] = {
            "lr": self.args.lr,
            "num_queries": self.args.chunk_size,
            "kl_weight": self.args.kl_weight,
            "hidden_dim": self.args.hidden_dim,
            "dim_feedforward": self.args.dim_feedforward,
            "lr_backbone": 1e-5,
            "backbone": "resnet18",
            "enc_layers": 4,
            "dec_layers": 7,
            "nheads": 8,
            "camera_names": self.args.camera_names,
        }

        # Construct policy
        DETRVAE.set_state_dim(len(self.model_meta_info["state"]["example"]))
        DETRVAE.set_action_dim(len(self.model_meta_info["action"]["example"]))
        self.policy = ACTPolicy(self.model_meta_info["policy"]["args"])
        self.policy.cuda()

        # Construct optimizer
        self.optimizer = self.policy.configure_optimizers()

        # Print policy information
        self.print_policy_info()
        print(f"  - chunk size: {self.args.chunk_size}")

    def train_loop(self):
        # Update args nếu có wandb.config (sweep mode)
        if wandb.run is not None and hasattr(wandb, "config"):
            config = wandb.config
            self.args.lr = config.lr
            self.args.kl_weight = config.kl_weight
            self.args.chunk_size = config.chunk_size
            self.args.hidden_dim = config.hidden_dim

        # Experiment Tracking & Visualization
        wandb.init(
            project="robomanip-act",
            config={
                "learning_rate": self.args.lr,
                "epochs": self.args.num_epochs,
                "batch_size": self.args.batch_size,
                "kl_weight": self.args.kl_weight,
                "chunk_size": self.args.chunk_size,
                "hidden_dim": self.args.hidden_dim,
                "dim_feedforward": self.args.dim_feedforward,
                "model": "ACTPolicy",
                "dataset": getattr(self.args, "dataset_name", "Unknown"),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            },
            tags=["experiment-tracking", "sweep-ready"],
            notes="Training with wandb integration",
        )

        # Log gradients and parameters
        wandb.watch(self.policy, log="all")

        step = 0
        for epoch in tqdm(range(self.args.num_epochs)):
            # Run train step
            self.policy.train()
            batch_result_list = []
            for data in self.train_dataloader:
                self.optimizer.zero_grad()
                batch_result = self.policy(*[d.cuda() for d in data])
                loss = batch_result["loss"]
                loss.backward()
                self.optimizer.step()
                batch_result_detached = self.detach_batch_result(batch_result)
                batch_result_list.append(batch_result_detached)

                step += 1
                wandb.log({
                    "step": step,
                    "train_loss_step": loss.item(),
                    "train_l1_step": batch_result_detached.get("l1", 0),
                    "train_kl_step": batch_result_detached.get("kl", 0),
                })

            train_epoch_summary = self.log_epoch_summary(batch_result_list, "train", epoch)

            # Run validation step
            with torch.inference_mode():
                self.policy.eval()
                batch_result_list = []
                for data in self.val_dataloader:
                    batch_result = self.policy(*[d.cuda() for d in data])
                    batch_result_list.append(self.detach_batch_result(batch_result))
                epoch_summary = self.log_epoch_summary(batch_result_list, "val", epoch)

                # Update best checkpoint
                self.update_best_ckpt(epoch_summary)

            wandb.log({
                "epoch": epoch,
                "train_loss": train_epoch_summary.get("loss", 0),
                "train_l1": train_epoch_summary.get("l1", 0),
                "train_kl": train_epoch_summary.get("kl", 0),
                "val_loss": epoch_summary.get("loss", 0),
                "val_l1": epoch_summary.get("l1", 0),
                "val_kl": epoch_summary.get("kl", 0),
            })

            if epoch % max(self.args.num_epochs // 10, 1) == 0:
                model_path = self.save_current_ckpt(f"epoch{epoch:0>3}")
                if model_path:
                    wandb.save(model_path)

        # Save last model
        last_ckpt_path = self.save_current_ckpt("last")
        if last_ckpt_path:
            wandb.save(last_ckpt_path)

        # Save best model
        best_ckpt_path = self.save_best_ckpt()
        if best_ckpt_path:
            wandb.save(best_ckpt_path)

        wandb.finish()

    # Sweep entrypoint
    @classmethod
    def sweep_entrypoint(cls):
        def sweep_train():
            trainer = cls()
            trainer.run()
            trainer.close()
        return sweep_train

    # Sweep config
    @classmethod
    def get_sweep_config(cls):
        return {
            "method": "bayes",
            "metric": {"name": "val_loss", "goal": "minimize"},
            "parameters": {
                "lr": {"min": 1e-6, "max": 5e-4},
                "kl_weight": {"values": [1, 5, 10, 20]},
                "chunk_size": {"values": [50, 100, 200]},
                "hidden_dim": {"values": [256, 512, 1024]},
            },
        }
