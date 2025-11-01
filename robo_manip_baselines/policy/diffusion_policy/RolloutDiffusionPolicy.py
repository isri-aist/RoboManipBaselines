import os
import sys

import cv2
import matplotlib.pylab as plt
import numpy as np
import torch

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../third_party/diffusion_policy")
)
from robo_manip_baselines.common import (
    DataKey,
    RolloutBase,
    denormalize_data,
    normalize_data,
)


class RolloutDiffusionPolicy(RolloutBase):
    def setup_policy(self):
        # For backward compatibility
        if "backbone" not in self.model_meta_info["policy"]:
            self.model_meta_info["policy"]["backbone"] = "cnn"
        if "scheduler" not in self.model_meta_info["policy"]:
            self.model_meta_info["policy"]["scheduler"] = "ddpm"

        # Print policy information
        self.print_policy_info()
        print(
            f"  - use ema: {self.model_meta_info['policy']['use_ema']}, backbone: {self.model_meta_info['policy']['backbone']}, scheduler: {self.model_meta_info['policy']['scheduler']}"
        )
        print(
            f"  - horizon: {self.model_meta_info['data']['horizon']}, obs steps: {self.model_meta_info['data']['n_obs_steps']}, action steps: {self.model_meta_info['data']['n_action_steps']}"
        )
        print(
            f"  - image size: {self.model_meta_info['data']['image_size']}, image crop size: {self.model_meta_info['data']['image_crop_size']}"
        )

        # Construct scheduler
        if self.model_meta_info["policy"]["scheduler"] == "ddpm":
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

            noise_scheduler = DDPMScheduler(
                **self.model_meta_info["policy"]["noise_scheduler_args"]
            )
        elif self.model_meta_info["policy"]["scheduler"] == "ddim":
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler

            noise_scheduler = DDIMScheduler(
                **self.model_meta_info["policy"]["noise_scheduler_args"]
            )
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid scheduler: {self.model_meta_info['policy']['scheduler']}"
            )

        # Construct policy
        if self.model_meta_info["policy"]["backbone"] == "cnn":
            from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
                DiffusionUnetHybridImagePolicy,
            )

            PolicyClass = DiffusionUnetHybridImagePolicy
        elif self.model_meta_info["policy"]["backbone"] == "transformer":
            from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import (
                DiffusionTransformerHybridImagePolicy,
            )

            PolicyClass = DiffusionTransformerHybridImagePolicy
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid backbone: {self.model_meta_info['policy']['backbone']}"
            )
        self.policy = PolicyClass(
            noise_scheduler=noise_scheduler,
            **self.model_meta_info["policy"]["args"],
        )

        # Load checkpoint
        self.load_ckpt()

    def setup_plot(self):
        fig_ax = plt.subplots(
            2,
            len(self.camera_names),
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

    def infer_policy(self):
        # Update observation buffer
        if len(self.state_keys) > 0:
            self.update_state_buf()
        self.update_images_buf()

        # Infer
        if self.policy_action_buf is None or len(self.policy_action_buf) == 0:
            input_data = {}
            if len(self.state_keys) > 0:
                input_data["state"] = self.get_state()
            for camera_name, image in zip(self.camera_names, self.get_images()):
                input_data[DataKey.get_rgb_image_key(camera_name)] = image
            action = self.policy.predict_action(input_data)["action"][0]
            self.policy_action_buf = list(
                action.cpu().detach().numpy().astype(np.float64)
            )

        # Store action
        self.policy_action = denormalize_data(
            self.policy_action_buf.pop(0), self.model_meta_info["action"]
        )
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )

    def update_state_buf(self):
        state = np.concatenate(
            [
                self.motion_manager.get_data(state_key, self.obs)
                for state_key in self.state_keys
            ]
        )
        state = normalize_data(state, self.model_meta_info["state"])
        state = torch.tensor(state, dtype=torch.float32)

        if self.state_buf is None:
            self.state_buf = [
                state for _ in range(self.model_meta_info["data"]["n_obs_steps"])
            ]
        else:
            self.state_buf.pop(0)
            self.state_buf.append(state)

    def get_state(self):
        return torch.stack(self.state_buf, dim=0)[torch.newaxis].to(self.device)

    def update_images_buf(self):
        images = []
        for camera_name in self.camera_names:
            image = self.info["rgb_images"][camera_name]

            image = cv2.resize(image, self.model_meta_info["data"]["image_size"])

            image = np.moveaxis(image, -1, -3)
            image = torch.tensor(image, dtype=torch.uint8)
            image = self.image_transforms(image)
            # Adjust to a range from -1 to 1 to match the original implementation
            image = image * 2.0 - 1.0

            images.append(image)

        if self.images_buf is None:
            self.images_buf = [
                [image for _ in range(self.model_meta_info["data"]["n_obs_steps"])]
                for image in images
            ]
        else:
            for single_images_buf, image in zip(self.images_buf, images):
                single_images_buf.pop(0)
                single_images_buf.append(image)

    def get_images(self):
        return [
            torch.stack(single_images_buf, dim=0)[torch.newaxis].to(self.device)
            for single_images_buf in self.images_buf
        ]

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
