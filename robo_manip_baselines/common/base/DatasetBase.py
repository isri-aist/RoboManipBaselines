import numpy as np
import torch
from torchvision.transforms import v2

from ..utils.DataUtils import normalize_data


class DatasetBase(torch.utils.data.Dataset):
    def __init__(
        self,
        filenames,
        model_meta_info,
        enable_rmb_cache=False,
    ):
        self.filenames = filenames
        self.model_meta_info = model_meta_info
        self.enable_rmb_cache = enable_rmb_cache

        if self.enable_rmb_cache:
            print(f"[{self.__class__.__name__}] Enable data caching in RmbData.")

        self.setup_image_transforms()

        self.setup_variables()

    def setup_image_transforms(self):
        """
        Setup image transforms.

        Image transforms should be responsible for converting the data type from uint8 to float32 with scale (255 -> 1.0).
        """
        image_transform_list = []

        if self.model_meta_info["image"]["aug_erasing_scale"] > 0.0:
            scale = self.model_meta_info["image"]["aug_erasing_scale"]
            image_transform_list.append(v2.RandomErasing(p=0.5 * scale))

        if self.model_meta_info["image"]["aug_color_scale"] > 0.0:
            scale = self.model_meta_info["image"]["aug_color_scale"]
            image_transform_list.append(
                v2.ColorJitter(
                    brightness=0.4 * scale,
                    contrast=0.4 * scale,
                    saturation=0.4 * scale,
                    hue=0.05 * scale,
                )
            )

        if self.model_meta_info["image"]["aug_affine_scale"] > 0.0:
            scale = self.model_meta_info["image"]["aug_affine_scale"]
            image_transform_list.append(
                v2.RandomAffine(
                    degrees=4.0 * scale,
                    translate=(0.05 * scale, 0.05 * scale),
                    scale=(1.0 - 0.1 * scale, 1.0 + 0.1 * scale),
                )
            )

        image_transform_list.append(v2.ToDtype(torch.float32, scale=True))

        if self.model_meta_info["image"]["aug_std"] > 0.0:
            image_transform_list.append(
                v2.GaussianNoise(sigma=self.model_meta_info["image"]["aug_std"])
            )

        self.image_transforms = v2.Compose(image_transform_list)

    def setup_variables(self):
        """Setup internal variables."""
        pass

    def pre_convert_data(self, state, action, images):
        """Pre-convert data. Arguments must be numpy arrays (not torch tensors)."""
        if state is not None:
            state = normalize_data(state, self.model_meta_info["state"])
        if action is not None:
            action = normalize_data(action, self.model_meta_info["action"])
        if images is not None:
            images = np.moveaxis(images, -1, -3)

        return state, action, images

    def augment_data(self, state, action, images):
        """Augment data. Arguments must be torch tensors (not numpy arrays)."""
        if state is not None:
            if self.model_meta_info["state"]["aug_std"] > 0.0:
                state += self.model_meta_info["state"]["aug_std"] * torch.randn_like(
                    state
                )
        if action is not None:
            if self.model_meta_info["action"]["aug_std"] > 0.0:
                action += self.model_meta_info["action"]["aug_std"] * torch.randn_like(
                    action
                )
        if images is not None:
            if images.ndim < 3:
                raise ValueError(
                    f"[{self.__class__.__name__}] Input must have at least 3 dimensions (C, H, W)"
                )
            elif images.ndim == 3:
                images = self.image_transforms(images)
            else:
                orig_prefix_shape = images.shape[:-3]
                orig_image_list = images.reshape(-1, *images.shape[-3:])
                transformed_image_list = [
                    self.image_transforms(img) for img in orig_image_list
                ]
                images = torch.stack(transformed_image_list)
                new_img_shape = images.shape[-3:]
                images = images.view(*orig_prefix_shape, *new_img_shape)

        return state, action, images
