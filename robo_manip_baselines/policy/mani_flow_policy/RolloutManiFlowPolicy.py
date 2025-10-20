import os
import sys

import cv2
import matplotlib.pylab as plt
import numpy as np
import torch

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../../../third_party/ManiFlow_Policy/ManiFlow",
    )
)
from maniflow.policy.maniflow_image_policy import ManiFlowTransformerImagePolicy
from maniflow.policy.maniflow_pointcloud_policy import (
    ManiFlowTransformerPointcloudPolicy,
)

from robo_manip_baselines.common import (
    DataKey,
    RolloutBase,
    convert_depth_image_to_pointcloud,
    denormalize_data,
    euler_to_rotation_matrix,
    normalize_data,
)
from robo_manip_baselines.common.utils.Vision3dUtils import (
    crop_pointcloud_bb,
    downsample_pointcloud_fps,
    rotate_pointcloud,
)


class RolloutManiFlowPolicy(RolloutBase):
    def set_additional_args(self, parser):
        parser.add_argument(
            "--plot_colored_pointcloud",
            action="store_true",
            help="Whether to force plotting of colored point clouds",
        )

    def setup_variables(self):
        return super().setup_variables()

    def setup_policy(self):
        # Print policy information
        self.print_policy_info()
        print(f"  - policy type: {self.model_meta_info['policy']['policy_type']}")
        print(f"  - use ema: {self.model_meta_info['policy']['use_ema']}")
        print(
            f"  - horizon: {self.model_meta_info['data']['horizon']}, obs steps: {self.model_meta_info['data']['n_obs_steps']}, action steps: {self.model_meta_info['data']['n_action_steps']}"
        )
        data_info = self.model_meta_info["data"]
        if self.model_meta_info["policy"]["policy_type"] == "Pointcloud":
            print(
                f"  - with color: {data_info['use_pc_color']}, num points: {data_info['num_points']}, image size: {data_info['image_size']}, min bound: {data_info['min_bound']}, max bound: {data_info['max_bound']}, rpy_angle: {data_info['rpy_angle']}"
            )
        else:
            print(f"  - image size: {data_info['image_size']}")

        # Construct policy
        self.policy_type = self.model_meta_info["policy"]["policy_type"]
        if self.policy_type == "Pointcloud":
            PolicyClass = ManiFlowTransformerPointcloudPolicy
        else:
            PolicyClass = ManiFlowTransformerImagePolicy
        self.policy = PolicyClass(
            **self.model_meta_info["policy"]["args"],
        )

        # Load checkpoint
        self.load_ckpt()

    def setup_plot(self):
        fig_ax = plt.subplots(
            2,
            1,
            figsize=(13.5, 6.0),
            dpi=60,
            squeeze=False,
            constrained_layout=True,
        )

        super().setup_plot(fig_ax)

    def reset_variables(self):
        super().reset_variables()

        self.state_buf = None
        self.policy_action_buf = None

        if self.policy_type == "Pointcloud":
            self.pointcloud_buf = None
            self.pointcloud_plot = None
            self.pointcloud_scatter = None
        else:
            self.images_buf = None

    def infer_policy(self):
        # Update observation buffer
        if len(self.state_keys) > 0:
            self.update_state_buf()
        if self.policy_type == "Pointcloud":
            self.update_pointcloud_buf()
        else:
            self.update_images_buf()

        # Infer
        if self.policy_action_buf is None or len(self.policy_action_buf) == 0:
            input_data = {}
            if len(self.state_keys) > 0:
                input_data["state"] = self.get_state()
            if self.policy_type == "Pointcloud":
                input_data["point_cloud"] = self.get_pointcloud()
            else:
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

    def update_pointcloud_buf(self):
        # Get latest value
        camera_name = self.camera_names[0]
        if camera_name in self.env.unwrapped.pointcloud_camera_names:
            pointcloud = self.info["pointclouds"][camera_name]
        else:
            rgb_image = self.info["rgb_images"][camera_name]
            depth_image = self.info["depth_images"][camera_name]
            fovy = self.env.unwrapped.get_camera_fovy(camera_name)

            # Resize images
            image_size = self.model_meta_info["data"]["image_size"]
            rgb_image = cv2.resize(rgb_image, image_size)
            depth_image = cv2.resize(depth_image, image_size)

            # Convert to pointcloud
            pointcloud = np.concat(
                convert_depth_image_to_pointcloud(depth_image, fovy, rgb_image),
                axis=1,
            )
        # Crop and downsample pointcloud
        rotmat = euler_to_rotation_matrix(self.model_meta_info["data"]["rpy_angle"])
        pointcloud = rotate_pointcloud(pointcloud, rotmat)
        pointcloud = crop_pointcloud_bb(
            pointcloud,
            self.model_meta_info["data"]["min_bound"],
            self.model_meta_info["data"]["max_bound"],
        )
        pointcloud = downsample_pointcloud_fps(
            pointcloud,
            self.model_meta_info["data"]["num_points"],
        )
        self.pointcloud_plot = pointcloud.copy()
        pointcloud = normalize_data(pointcloud, self.model_meta_info["pointcloud"])
        pointcloud = torch.tensor(pointcloud, dtype=torch.float32)

        # Store to buffer
        if self.pointcloud_buf is None:
            self.pointcloud_buf = [
                pointcloud for _ in range(self.model_meta_info["data"]["n_obs_steps"])
            ]
        else:
            self.pointcloud_buf.pop(0)
            self.pointcloud_buf.append(pointcloud)

    def get_pointcloud(self):
        return torch.stack(self.pointcloud_buf, dim=0)[torch.newaxis].to(self.device)

    def plot_pointcloud(self, ax):
        xyz_array = self.pointcloud_plot[:, :3]
        if (
            self.model_meta_info["data"]["use_pc_color"]
            or self.args.plot_colored_pointcloud
        ):
            rgb_array = self.pointcloud_plot[:, 3:]
        else:
            rgb_array = "steelblue"

        if self.pointcloud_scatter is None:
            ax.remove()
            ax = self.fig.add_subplot(2, 1, 1, projection="3d")
            ax.view_init(elev=-90, azim=-90)
            ax.set_xlim(xyz_array[:, 0].min(), xyz_array[:, 0].max())
            ax.set_ylim(xyz_array[:, 1].min(), xyz_array[:, 1].max())
            ax.set_zlim(xyz_array[:, 2].min(), xyz_array[:, 2].max())
        else:
            self.pointcloud_scatter.remove()

        ax.axis("off")
        ax.set_box_aspect(np.ptp(xyz_array, axis=0))
        self.pointcloud_scatter = ax.scatter(
            xyz_array[:, 0], xyz_array[:, 1], xyz_array[:, 2], c=rgb_array
        )
        ax.set_title(self.camera_names[0], fontsize=20)

        # Reassign the updated axis
        return ax

    def draw_plot(self):
        # Clear plot
        for _ax in np.ravel(self.ax[1]):  # Do not reset 3D axis
            _ax.cla()
            _ax.axis("off")

        if self.policy_type == "Pointcloud":
            # Plot pointclouds
            self.ax[0, 0] = self.plot_pointcloud(self.ax[0, 0])
        else:
            self.plot_images(self.ax[0, 0 : len(self.camera_names)])

        # Plot action
        self.plot_action(self.ax[1, 0])

        # Finalize plot
        self.canvas.draw()
        cv2.imshow(
            self.policy_name,
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )
