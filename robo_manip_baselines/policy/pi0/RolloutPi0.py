import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third_party/lerobot"))
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.control_utils import predict_action

from robo_manip_baselines.common import RolloutBase
from robo_manip_baselines.common.data.DataKey import DataKey


class RolloutPi0(RolloutBase):
    require_task_desc = True

    def set_additional_args(self, parser):
        # TODO: Disable rendering with matplotlib and cv2, as it causes the program to hang
        parser.set_defaults(no_plot=True)

    def setup_model_meta_info(self):
        self.state_keys = [DataKey.MEASURED_JOINT_POS]
        self.action_keys = [DataKey.COMMAND_JOINT_POS]

        if self.args.skip is None:
            self.args.skip = 1
        if self.args.skip_draw is None:
            self.args.skip_draw = self.args.skip

    def setup_policy(self):
        self.device = torch.device("cuda")

        self.policy = PI0Policy.from_pretrained(self.args.checkpoint)

        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            self.args.checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

        self.state_dim = self.policy.config.input_features["observation.state"].shape[0]
        self.action_dim = self.policy.config.output_features["action"].shape[0]
        self.camera_names = [
            key.replace("observation.images.", "").replace("_rgb", "")
            for key in self.policy.config.input_features
            if key.startswith("observation.images.")
        ]

        self.print_policy_info()

    def reset_variables(self):
        super().reset_variables()

        self.policy.reset()

    def get_state(self):
        if len(self.state_keys) == 0:
            return np.zeros(0, dtype=np.float32)

        return np.concatenate(
            [
                self.motion_manager.get_data(state_key, self.obs)
                for state_key in self.state_keys
            ]
        )

    def get_images(self):
        images = {}
        for camera_name in self.camera_names:
            images[camera_name] = self.info["rgb_images"][camera_name].copy()
        return images

    def infer_policy(self):
        state = self.get_state()
        images = self.get_images()

        observation = {
            "observation.state": state,
        }
        for camera_name in self.camera_names:
            observation[f"observation.images.{camera_name}_rgb"] = images[camera_name]

        action = predict_action(
            observation=observation,
            policy=self.policy,
            device=self.device,
            preprocessor=self.preprocess,
            postprocessor=self.postprocess,
            use_amp=self.policy.config.use_amp,
            task=self.args.task_desc,
        )[0]

        self.policy_action = action.detach().cpu().numpy().astype(np.float64)
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )

    def draw_plot(self):
        # TODO: Disable rendering with matplotlib and cv2, as it causes the program to hang
        pass
