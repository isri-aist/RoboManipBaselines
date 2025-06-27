import argparse
import copy
import datetime
import glob
import os
import pickle
import random
import sys
from abc import ABC, abstractmethod

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..data.CachedDataset import CachedDataset
from ..data.DataKey import DataKey
from ..data.RmbData import RmbData
from ..utils.DataUtils import get_skipped_data_seq
from ..utils.MathUtils import set_random_seed
from ..utils.MiscUtils import remove_prefix


class TrainBase(ABC):
    @property
    def policy_name(self):
        return remove_prefix(self.__class__.__name__, "Train")

    def __init__(self):
        self.setup_args()

        set_random_seed(self.args.seed)

        self.setup_model_meta_info()

        self.setup_dataset()

        self.setup_policy()

    def setup_args(self, parser=None, argv=None):
        if parser is None:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )

        parser.add_argument(
            "--dataset_dir",
            type=str,
            required=True,
            help="dataset directory",
        )
        parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default=None,
            help="checkpoint directory",
        )

        parser.add_argument(
            "--enable_rmb_cache",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Whether to enable data caching in RmbData. This uses RAM heavily, so it should be disabled on computers with small RAM.",
        )
        parser.add_argument(
            "--use_cached_dataset",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Whether to use CachedDataset. When enabling this, make sure that non-reproducible processes such as data augmentation are not performed on the original dataset.",
        )

        parser.add_argument(
            "--state_keys",
            type=str,
            nargs="*",
            default=[DataKey.MEASURED_JOINT_POS],
            choices=DataKey.MEASURED_DATA_KEYS,
            help="state data keys",
        )
        parser.add_argument(
            "--action_keys",
            type=str,
            nargs="+",
            default=[DataKey.COMMAND_JOINT_POS],
            choices=DataKey.COMMAND_DATA_KEYS,
            help="action data keys",
        )
        parser.add_argument(
            "--camera_names",
            type=str,
            nargs="+",
            default=["front"],
            help="camera names",
        )

        parser.add_argument(
            "--train_ratio", type=float, default=0.8, help="ratio of train data"
        )
        parser.add_argument(
            "--val_ratio", type=float, default=None, help="ratio of validation data"
        )

        parser.add_argument(
            "--norm_type",
            type=str,
            default="gaussian",
            choices=["gaussian", "limits"],
            help="normalization type",
        )
        parser.add_argument(
            "--state_aug_std",
            type=float,
            default=0.0,
            help="Standard deviation of random noise added to state",
        )
        parser.add_argument(
            "--action_aug_std",
            type=float,
            default=0.0,
            help="Standard deviation of random noise added to action",
        )
        parser.add_argument(
            "--image_aug_std",
            type=float,
            default=0.0,
            help="Standard deviation of random noise added to images",
        )
        parser.add_argument(
            "--image_aug_erasing_scale",
            type=float,
            default=0.0,
            help="Scale of random erasing applied to the images",
        )
        parser.add_argument(
            "--image_aug_color_scale",
            type=float,
            default=0.0,
            help="Scale of color random noise added to the images",
        )
        parser.add_argument(
            "--image_aug_affine_scale",
            type=float,
            default=0.0,
            help="Scale of affine random noise added to the images",
        )

        parser.add_argument(
            "--skip",
            type=int,
            default=3,
            help="skip interval of data sequence (set 1 for no skip)",
        )

        parser.add_argument("--batch_size", type=int, help="batch size")
        parser.add_argument("--num_epochs", type=int, help="number of epochs")
        parser.add_argument("--lr", type=float, help="learning rate")
        parser.add_argument(
            "--num_workers", type=int, default=4, help="number of workers in dataloader"
        )

        parser.add_argument("--seed", type=int, default=42, help="random seed")

        self.set_additional_args(parser)

        if argv is None:
            argv = sys.argv
        self.args = parser.parse_args(argv[1:])

        # Set checkpoint directory if it is not specified
        if self.args.checkpoint_dir is None:
            dataset_dirname = os.path.basename(os.path.normpath(self.args.dataset_dir))
            checkpoint_dirname = "{}_{}_{:%Y%m%d_%H%M%S}".format(
                dataset_dirname, self.policy_name, datetime.datetime.now()
            )
            self.args.checkpoint_dir = os.path.normpath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "checkpoint",
                    self.policy_name,
                    checkpoint_dirname,
                )
            )

    def set_additional_args(self, parser):
        pass

    def setup_model_meta_info(self):
        self.model_meta_info = {
            "data": {
                "name": os.path.basename(os.path.normpath(self.args.dataset_dir)),
                "skip": self.args.skip,
            },
            "policy": {"name": self.policy_name},
            "train": {
                "batch_size": self.args.batch_size,
                "num_epochs": self.args.num_epochs,
                "lr": self.args.lr,
            },
            "state": {
                "keys": self.args.state_keys,
                "aug_std": self.args.state_aug_std,
            },
            "action": {
                "keys": self.args.action_keys,
                "aug_std": self.args.action_aug_std,
            },
            "image": {
                "camera_names": self.args.camera_names,
                "aug_std": self.args.image_aug_std,
                "aug_erasing_scale": self.args.image_aug_erasing_scale,
                "aug_color_scale": self.args.image_aug_color_scale,
                "aug_affine_scale": self.args.image_aug_affine_scale,
            },
        }

    def setup_dataset(self):
        if self.args.enable_rmb_cache and self.args.use_cached_dataset:
            raise ValueError(
                f"[{self.__class__.__name__}] Both 'enable_rmb_cache' and 'use_cached_dataset' options cannot be True at the same time."
            )

        # Get file list
        all_filenames = [
            f
            for f in glob.glob(f"{self.args.dataset_dir}/**/*.*", recursive=True)
            if f.endswith(".rmb")
            or (f.endswith(".hdf5") and not f.endswith(".rmb.hdf5"))
        ]
        random.shuffle(all_filenames)
        train_num = max(
            int(np.clip(self.args.train_ratio, 0.0, 1.0) * len(all_filenames)), 1
        )
        if self.args.val_ratio is None:
            val_num = max(len(all_filenames) - train_num, 1)
        else:
            val_num = max(
                int(np.clip(self.args.val_ratio, 0.0, 1.0) * len(all_filenames)), 1
            )
        train_filenames = all_filenames[:train_num]
        val_filenames = all_filenames[-1 * val_num :]

        # Set data stats
        self.set_data_stats(all_filenames)

        # Make dataloader
        self.train_dataloader = self.make_dataloader(train_filenames, shuffle=True)
        self.val_dataloader = self.make_dataloader(val_filenames, shuffle=False)

        # Setup tensorboard
        self.writer = SummaryWriter(self.args.checkpoint_dir)

        # Print dataset information
        self.print_dataset_info()

    def set_data_stats(self, all_filenames):
        # Load dataset
        all_state = []
        all_action = []
        rgb_image_example = None
        depth_image_example = None
        episode_len_list = []
        for filename in all_filenames:
            with RmbData(filename) as rmb_data:
                episode_len = rmb_data[DataKey.TIME][:: self.args.skip].shape[0]
                episode_len_list.append(episode_len)

                # Load state
                if len(self.args.state_keys) == 0:
                    state = np.zeros((episode_len, 0), dtype=np.float64)
                else:
                    state = np.concatenate(
                        [
                            get_skipped_data_seq(rmb_data[key][:], key, self.args.skip)
                            for key in self.args.state_keys
                        ],
                        axis=1,
                    )
                all_state.append(state)

                # Load action
                if len(self.args.action_keys) == 0:
                    action = np.zeros((episode_len, 0), dtype=np.float64)
                else:
                    action = np.concatenate(
                        [
                            get_skipped_data_seq(rmb_data[key][:], key, self.args.skip)
                            for key in self.args.action_keys
                        ],
                        axis=1,
                    )
                all_action.append(action)

                # Load image
                if rgb_image_example is None:
                    rgb_image_example = {
                        camera_name: rmb_data[DataKey.get_rgb_image_key(camera_name)][0]
                        for camera_name in self.args.camera_names
                        if DataKey.get_rgb_image_key(camera_name) in rmb_data
                    }
                if depth_image_example is None:
                    depth_image_example = {
                        camera_name: rmb_data[DataKey.get_depth_image_key(camera_name)][
                            0
                        ]
                        for camera_name in self.args.camera_names
                        if DataKey.get_depth_image_key(camera_name) in rmb_data
                    }
        all_state = np.concatenate(all_state, dtype=np.float64)
        all_action = np.concatenate(all_action, dtype=np.float64)

        self.model_meta_info["state"].update(self.calc_stats_from_seq(all_state))
        self.model_meta_info["action"].update(self.calc_stats_from_seq(all_action))
        self.model_meta_info["image"].update(
            {
                "rgb_example": rgb_image_example,
                "depth_example": depth_image_example,
            }
        )
        self.model_meta_info["data"].update(
            {
                "mean_episode_len": np.mean(episode_len_list),
                "min_episode_len": np.min(episode_len_list),
                "max_episode_len": np.max(episode_len_list),
            }
        )

    def calc_stats_from_seq(self, data_seq):
        data_min = data_seq.min(axis=0)
        data_max = data_seq.max(axis=0)
        data_range = np.clip(data_max - data_min, 1e-3, 1e10)
        data_mean = data_seq.mean(axis=0)
        data_std = np.clip(data_seq.std(axis=0), 1e-3, 1e10)

        data_stats = {
            "norm_config": {
                "type": self.args.norm_type,
                **self.get_extra_norm_config(),
            },
            "min": data_min,
            "max": data_max,
            "range": data_range,
            "mean": data_mean,
            "std": data_std,
            "example": data_seq[0],
        }
        return data_stats

    def get_extra_norm_config(self):
        if self.args.norm_type == "gaussian":
            return {}
        elif self.args.norm_type == "limits":
            return {
                "out_min": 0.0,
                "out_max": 1.0,
            }
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid normalization type: {self.args.norm_type}"
            )

    def make_dataloader(self, filenames, shuffle=True):
        dataset = self.DatasetClass(
            filenames, self.model_meta_info, self.args.enable_rmb_cache
        )
        if self.args.use_cached_dataset:
            dataset = CachedDataset(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=self.args.num_workers,
            persistent_workers=True,
            prefetch_factor=4,
        )

        return dataloader

    def print_dataset_info(self):
        print(
            f"[{self.__class__.__name__}] Load dataset from {self.args.dataset_dir}\n"
            f"  - train size: {len(self.train_dataloader.dataset)}, files: {len(self.train_dataloader.dataset.filenames)}\n"
            f"  - val size: {len(self.val_dataloader.dataset)}, files: {len(self.val_dataloader.dataset.filenames)}\n"
            f"  - episode len mean: {int(self.model_meta_info['data']['mean_episode_len'])}, min: {self.model_meta_info['data']['min_episode_len']}, max: {self.model_meta_info['data']['max_episode_len']}"
        )
        print(
            f"  - aug std state: {self.model_meta_info['state']['aug_std']}, action: {self.model_meta_info['action']['aug_std']}, "
            f"image: {self.model_meta_info['image']['aug_std']}, erasing: {self.model_meta_info['image']['aug_erasing_scale']}, color: {self.model_meta_info['image']['aug_color_scale']}, affine: {self.model_meta_info['image']['aug_affine_scale']}"
        )
        image_transform_str_list = [
            f"{image_transform.__class__.__name__}"
            for image_transform in self.train_dataloader.dataset.image_transforms.transforms
        ]
        print(
            f"  - norm type: {self.args.norm_type}, image transforms: {image_transform_str_list}"
        )

    @abstractmethod
    def setup_policy(self):
        pass

    def print_policy_info(self):
        print(
            f"[{self.__class__.__name__}] Construct {self.policy_name} policy.\n"
            f"  - state dim: {len(self.model_meta_info['state']['example'])}, action dim: {len(self.model_meta_info['action']['example'])}, camera num: {len(self.args.camera_names)}\n"
            f"  - state keys: {self.args.state_keys}\n"
            f"  - action keys: {self.args.action_keys}\n"
            f"  - camera names: {self.args.camera_names}\n"
            f"  - skip: {self.args.skip}, batch size: {self.args.batch_size}, num epochs: {self.args.num_epochs}, num workers: {self.args.num_workers}"
        )

    def run(self):
        # Save model meta info
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        model_meta_info_path = os.path.join(
            self.args.checkpoint_dir, "model_meta_info.pkl"
        )
        with open(model_meta_info_path, "wb") as f:
            pickle.dump(self.model_meta_info, f)
        print(
            f"[{self.__class__.__name__}] Save model meta info: {model_meta_info_path}"
        )

        # Train loop
        print(
            f"[{self.__class__.__name__}] Train with saving checkpoints: {self.args.checkpoint_dir}"
        )
        self.best_ckpt_info = {"loss": np.inf, "epoch": -1}
        self.train_loop()

    @abstractmethod
    def train_loop(self):
        pass

    def detach_batch_result(self, batch_result):
        for k, v in batch_result.items():
            if isinstance(v, torch.Tensor):
                batch_result[k] = v.item()
        return batch_result

    def log_epoch_summary(self, batch_result_list, label, epoch):
        epoch_summary = {"epoch": epoch}

        if label == "train":
            epoch_summary["GPU_usage [GB]"] = (
                torch.cuda.max_memory_reserved() / 1024**3
            )  # [GB]
            epoch_summary["RAM_usage [GB]"] = (
                self.get_total_memory_usage() / 1024**3
            )  # [GB]
            epoch_summary["best_epoch"] = self.best_ckpt_info["epoch"]

        for k in batch_result_list[0]:
            epoch_summary[k] = np.mean(
                [batch_result[k] for batch_result in batch_result_list]
            )

        for k, v in epoch_summary.items():
            self.writer.add_scalar(f"{k}/{label}", v, epoch)

        return epoch_summary

    def update_best_ckpt(self, epoch_summary, policy=None):
        if policy is None:
            policy = self.policy

        if epoch_summary["loss"] < self.best_ckpt_info["loss"]:
            self.best_ckpt_info = {
                "epoch": epoch_summary["epoch"],
                "loss": epoch_summary["loss"],
                "state_dict": copy.deepcopy(policy.state_dict()),
            }

    def save_current_ckpt(self, ckpt_suffix, policy=None):
        if policy is None:
            policy = self.policy

        ckpt_path = os.path.join(self.args.checkpoint_dir, f"policy_{ckpt_suffix}.ckpt")
        torch.save(policy.state_dict(), ckpt_path)
        return ckpt_path

    def save_best_ckpt(self):
        ckpt_path = os.path.join(self.args.checkpoint_dir, "policy_best.ckpt")
        torch.save(self.best_ckpt_info["state_dict"], ckpt_path)
        print(
            f"[{self.__class__.__name__}] Best val loss is {self.best_ckpt_info['loss']:.3f} at epoch {self.best_ckpt_info['epoch']}"
        )
        return ckpt_path

    def get_total_memory_usage(self):
        process = psutil.Process()
        mem_usage = process.memory_info().rss

        for child_process in process.children(recursive=True):
            mem_usage += child_process.memory_info().rss

        return mem_usage

    def close(self):
        self.writer.close()
