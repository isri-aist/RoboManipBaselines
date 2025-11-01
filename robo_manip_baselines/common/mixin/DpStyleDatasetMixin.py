from robo_manip_baselines.common import DataKey, RmbData


class DpStyleDatasetMixin:
    "This mixin is intended to be used in classes that inherit from DatasetBase."

    def setup_dp_style_chunk(self):
        self.chunk_info_list = []
        skip = self.model_meta_info["data"]["skip"]
        horizon = self.model_meta_info["data"]["horizon"]
        # Set pad_before and pad_after to values one less than n_obs_steps and n_action_steps, respectively
        # Ref: https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/config/task/pusht_image.yaml#L36-L37
        pad_before = self.model_meta_info["data"]["n_obs_steps"] - 1
        pad_after = self.model_meta_info["data"]["n_action_steps"] - 1

        for episode_idx, filename in enumerate(self.filenames):
            with RmbData(filename) as rmb_data:
                episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
                for start_time_idx in range(
                    -1 * pad_before, episode_len - (horizon - 1) + pad_after
                ):
                    self.chunk_info_list.append((episode_idx, start_time_idx))
