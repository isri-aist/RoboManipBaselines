import numpy as np
import torch

from robo_manip_baselines.common import (
    DataKey,
    DatasetBase,
    RmbData,
    get_skipped_data_seq,
)


class MlpDataset(DatasetBase):
    """Dataset to train MLP policy."""

    def setup_variables(self):
        skip = self.model_meta_info["data"]["skip"]

        self.chunk_info_list = []
        for episode_idx, filename in enumerate(self.filenames):
            with RmbData(filename) as rmb_data:
                episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
                for start_time_idx in range(0, episode_len):
                    self.chunk_info_list.append((episode_idx, start_time_idx))

    def __len__(self):
        return len(self.chunk_info_list)

    def __getitem__(self, chunk_idx):
        skip = self.model_meta_info["data"]["skip"]
        n_obs_steps = self.model_meta_info["data"]["n_obs_steps"]
        n_action_steps = self.model_meta_info["data"]["n_action_steps"]
        episode_idx, start_time_idx = self.chunk_info_list[chunk_idx]

        with RmbData(self.filenames[episode_idx], self.enable_rmb_cache) as rmb_data:
            episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
            obs_time_idxes = np.clip(
                np.arange(start_time_idx - n_obs_steps + 1, start_time_idx + 1),
                0,
                episode_len - 1,
            )
            action_time_idxes = np.clip(
                np.arange(start_time_idx, start_time_idx + n_action_steps),
                0,
                episode_len - 1,
            )

            # Load state
            if len(self.model_meta_info["state"]["keys"]) == 0:
                state = np.zeros(0, dtype=np.float64)
            else:
                state = np.concatenate(
                    [
                        get_skipped_data_seq(rmb_data[key][:], key, skip)[
                            obs_time_idxes
                        ]
                        for key in self.model_meta_info["state"]["keys"]
                    ],
                    axis=1,
                )

            # Load action
            action = np.concatenate(
                [
                    get_skipped_data_seq(rmb_data[key][:], key, skip)[action_time_idxes]
                    for key in self.model_meta_info["action"]["keys"]
                ],
                axis=1,
            )

            # Load images
            images = np.stack(
                [
                    rmb_data[DataKey.get_rgb_image_key(camera_name)][::skip][
                        obs_time_idxes
                    ]
                    for camera_name in self.model_meta_info["image"]["camera_names"]
                ],
                axis=0,
            )

        # Pre-convert data
        state, action, images = self.pre_convert_data(state, action, images)

        # Convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        images_tensor = torch.tensor(images, dtype=torch.uint8)

        # Augment data
        state_tensor, action_tensor, images_tensor = self.augment_data(
            state_tensor, action_tensor, images_tensor
        )

        # Sort in the order of policy inputs and outputs
        return state_tensor, images_tensor, action_tensor
