import cv2
import matplotlib.pylab as plt
import numpy as np
import torch

from robo_manip_baselines.common import (
    DataKey,
    RolloutBase,
    denormalize_data,
    normalize_data,
)

from .PpoPolicy import PpoPolicy


def gripper_q_robomanip_to_maniskill(q_robomanip):
    """Convert RoboManip gripper scalar to ManiSkill scale."""

    return (q_robomanip - 840.0) / (-1000.0)


class RolloutPpo(RolloutBase):
    def setup_policy(self):
        # Print policy information
        self.print_policy_info()
        print(
            f"  - obs steps: {self.model_meta_info['data']['n_obs_steps']}, action steps: {self.model_meta_info['data']['n_action_steps']}"
        )

        # Construct policy
        self.policy = PpoPolicy(
            self.state_dim,
            self.action_dim,
            len(self.camera_names),
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

    def get_state(self):
        # Get latest value
        if len(self.state_keys) == 0:
            state = np.zeros(0, dtype=np.float32)
        else:
            state = np.concatenate(
                [
                    self.motion_manager.get_data(state_key, self.obs)
                    for state_key in self.state_keys
                ]
            )

        qpos = self.motion_manager.get_data(DataKey.MEASURED_JOINT_POS, self.obs)
        qvel = self.motion_manager.get_data(DataKey.MEASURED_JOINT_VEL, self.obs)
        target_qpos = np.array(
            [
                -0.00451699561347621,
                -0.4779577590016519,
                -0.0059982858387227518,
                0.8576097778375098,
                -0.032158957391327556,
                1.27989111575085,
                0.05005294852914137,
                0.0,
            ],
            dtype=np.float32,
        )
        qpos_ms = qpos.astype(np.float32).copy()
        qpos_ms[-1] = gripper_q_robomanip_to_maniskill(qpos_ms[-1])
        qvel_ms = qvel.astype(np.float32).copy()
        if qvel_ms.size > 0:
            qvel_ms[-1] = gripper_q_robomanip_to_maniskill(qvel_ms[-1])
        target_qpos_ms = target_qpos.copy()
        target_qpos_ms[-1] = gripper_q_robomanip_to_maniskill(target_qpos_ms[-1])

        self.state_for_ppo = np.concatenate([qpos_ms, qvel_ms, target_qpos_ms]).astype(
            np.float32
        )
        print(
            f"[rollout] state_for_ppo shape={self.state_for_ppo.shape}: {self.state_for_ppo}"
        )

        print(f"[rollout] raw state shape={state.shape}: {state}")

        norm_state = normalize_data(state, self.model_meta_info["state"])
        print(f"[rollout] norm state shape={norm_state.shape}: {norm_state}")

        state = torch.tensor(norm_state, dtype=torch.float32)

        # Store and return
        if self.state_buf is None:
            self.state_buf = [
                state for _ in range(self.model_meta_info["data"]["n_obs_steps"])
            ]
        else:
            self.state_buf.pop(0)
            self.state_buf.append(state)

        state = torch.stack(self.state_buf, dim=0)[torch.newaxis].to(self.device)

        return state

    def get_images(self):
        # Get latest value
        images = []
        for camera_name in self.camera_names:
            image = self.info["rgb_images"][camera_name]

            image = np.moveaxis(image, -1, -3)
            image = torch.tensor(image.copy(), dtype=torch.uint8)
            image = self.image_transforms(image)

            images.append(image)

        # Store and return
        if self.images_buf is None:
            self.images_buf = [
                [image for _ in range(self.model_meta_info["data"]["n_obs_steps"])]
                for image in images
            ]
        else:
            for single_images_buf, image in zip(self.images_buf, images):
                single_images_buf.pop(0)
                single_images_buf.append(image)

        images = torch.stack(
            [
                torch.stack(single_images_buf, dim=0)[torch.newaxis].to(self.device)
                for single_images_buf in self.images_buf
            ]
        )

        return images

    def infer_policy(self):
        # Infer
        if self.policy_action_buf is None or len(self.policy_action_buf) == 0:
            state = self.get_state()
            images = self.get_images()
            action = self.policy(state, images)[0]
            action_np = action.detach().cpu().numpy()
            print(
                f"[rollout] policy raw action shape={action_np.shape}: {action_np}"
            )
            self.policy_action_buf = list(
                action.cpu().detach().numpy().astype(np.float64)
            )

        # Store action
        self.policy_action = denormalize_data(
            self.policy_action_buf.pop(0), self.model_meta_info["action"]
        )
        print(
            f"[rollout] executed action shape={self.policy_action.shape}: {self.policy_action}"
        )
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )

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
