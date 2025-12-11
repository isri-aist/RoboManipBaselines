import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.CachedDataset import CachedDataset
from ..data.DataKey import DataKey
from ..data.RmbData import RmbData
from ..utils.DataUtils import get_skipped_data_seq
from .TrainBase import TrainBase


class CroppedTrainBase(TrainBase):
    """
    An extension of TrainBase that adds support for fixed image cropping from command-line arguments.
    """

    def _parse_camera_crops(self, crop_args):
        """Parse camera crop arguments from command line."""
        camera_crops = {}
        if crop_args:
            for crop_arg in crop_args:
                parts = crop_arg.split(":")
                if len(parts) != 2:
                    raise ValueError(f"Invalid camera crop format: {crop_arg}")
                cam_name, crop_str = parts
                try:
                    x, y, w, h = map(int, crop_str.split(","))
                    camera_crops[cam_name] = (x, y, w, h)
                except ValueError:
                    raise ValueError(f"Invalid crop values for {cam_name}: {crop_str}")
        return camera_crops

    def set_additional_args(self, parser):
        super().set_additional_args(parser)
        parser.add_argument(
            "--camera_crops",
            type=str,
            nargs="*",
            default=None,
            help="""Define cropping regions for cameras.
            Format: camera_name1:x,y,w,h camera_name2:x,y,w,h.
            Example: --camera_crops front:10,20,224,224 side:0,0,224,224.
            All crops must result in the same final dimensions.""",
        )

    def setup_model_meta_info(self):
        super().setup_model_meta_info()
        self.model_meta_info["image"]["camera_crops"] = self._parse_camera_crops(
            self.args.camera_crops
        )

    def set_data_stats(self):
        """
        Overrides TrainBase.set_data_stats to apply cropping to the example images
        before they are stored in the model metadata.
        """
        camera_crops = self.model_meta_info.get("image", {}).get("camera_crops", {})

        all_state = []
        all_action = []
        rgb_image_example = None
        depth_image_example = None
        episode_len_list = []
        for filename in self.all_filenames:
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

                # Load image examples and apply cropping
                if rgb_image_example is None:
                    rgb_image_example = {}
                    for camera_name in self.args.camera_names:
                        key = DataKey.get_rgb_image_key(camera_name)
                        if key in rmb_data:
                            img = rmb_data[key][0]
                            if camera_name in camera_crops:
                                x, y, w, h = camera_crops[camera_name]
                                img = img[y : y + h, x : x + w]
                            rgb_image_example[camera_name] = img

                if depth_image_example is None:
                    depth_image_example = {}
                    for camera_name in self.args.camera_names:
                        key = DataKey.get_depth_image_key(camera_name)
                        if key in rmb_data:
                            img = rmb_data[key][0]
                            if camera_name in camera_crops:
                                x, y, w, h = camera_crops[camera_name]
                                img = img[y : y + h, x : x + w]
                            depth_image_example[camera_name] = img

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

    def make_dataloader(self, filenames, shuffle=True):
        """
        Overrides TrainBase.make_dataloader to inject cropping logic into the dataset's
        preprocessing pipeline by monkey-patching the pre_convert_data method.
        """
        dataset = self.DatasetClass(
            filenames, self.model_meta_info, self.args.enable_rmb_cache
        )

        camera_crops = self.model_meta_info.get("image", {}).get("camera_crops", {})
        camera_names = self.model_meta_info.get("image", {}).get("camera_names", [])

        if camera_crops:
            original_pre_convert = dataset.pre_convert_data

            def new_pre_convert_data(self_dataset, state, action, images):
                if images is not None:
                    # images is expected to be a numpy array of shape (num_cameras, H, W, C)
                    is_cropped = False
                    output_images = []
                    for i, cam_name in enumerate(camera_names):
                        if cam_name in camera_crops:
                            is_cropped = True
                            x, y, w, h = camera_crops[cam_name]
                            output_images.append(images[i, y : y + h, x : x + w, :])
                        else:
                            output_images.append(images[i])

                    if is_cropped:
                        try:
                            images = np.stack(output_images, axis=0)
                        except ValueError as e:
                            print(
                                "ERROR: All camera crops must result in the same image dimensions. "
                                "Could not stack cropped images."
                            )
                            raise e
                return original_pre_convert(state, action, images)

            # Bind the new method to the dataset instance
            dataset.pre_convert_data = new_pre_convert_data.__get__(
                dataset, type(dataset)
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
