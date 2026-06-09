import numpy as np
import torch

from robo_manip_baselines.common import (
    DataKey,
    DatasetBase,
    RmbData,
    convert_data_to_policy,
    get_skipped_data_seq,
    normalize_data,
)


class AsyncVlaDataset(DatasetBase):
    """Dataset to train the AsyncVLA Edge Adapter.

    Each sample provides:
      - the current observation (state, images I_t),
      - a delayed image I_{t-k} (random delay k) combined with I_t into a
        6-channel delta image (optical-flow token replacement),
      - the cached frozen-pi0 guidance action chunk at t-k (delay injection),
      - the ground-truth action chunk at t (imitation label),
      - a reweighting weight emphasizing delay-sensitive ("reactive") samples.

    The cached guidance is produced offline by ``misc/AddPi0GuidanceToRmbData.py``
    and stored per timestep in the RMB data under ``guidance_key``.
    """

    def setup_variables(self):
        skip = self.model_meta_info["data"]["skip"]
        guidance_key = self.model_meta_info["data"]["guidance_key"]
        n_action_steps = self.model_meta_info["data"]["n_action_steps"]

        self.chunk_info_list = []
        for episode_idx, filename in enumerate(self.filenames):
            with RmbData(filename) as rmb_data:
                if guidance_key not in rmb_data.keys():
                    raise ValueError(
                        f"[{self.__class__.__name__}] Cached guidance '{guidance_key}' "
                        f"not found in {filename}. Run misc/AddPi0GuidanceToRmbData.py first."
                    )
                if episode_idx == 0:
                    guidance_chunk_len = rmb_data[guidance_key].shape[1]
                    if guidance_chunk_len != n_action_steps:
                        raise ValueError(
                            f"[{self.__class__.__name__}] Cached guidance chunk length "
                            f"{guidance_chunk_len} != n_action_steps {n_action_steps}. "
                            f"Re-run AddPi0GuidanceToRmbData.py with "
                            f"--n_action_steps {n_action_steps}."
                        )
                episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
                for start_time_idx in range(0, episode_len):
                    self.chunk_info_list.append((episode_idx, start_time_idx))

    def __len__(self):
        return len(self.chunk_info_list)

    def get_rng(self):
        """Get a per-worker random generator for delay sampling."""
        if not hasattr(self, "_rng"):
            worker_info = torch.utils.data.get_worker_info()
            seed = 0 if worker_info is None else worker_info.seed % (2**32)
            self._rng = np.random.default_rng(seed)
        return self._rng

    def __getitem__(self, chunk_idx):
        data_meta_info = self.model_meta_info["data"]
        skip = data_meta_info["skip"]
        n_action_steps = data_meta_info["n_action_steps"]
        delay_min = data_meta_info["delay_min"]
        delay_max = data_meta_info["delay_max"]
        dth = data_meta_info["dth"]
        reweight_gain = data_meta_info["reweight_gain"]
        guidance_key = data_meta_info["guidance_key"]
        episode_idx, start_time_idx = self.chunk_info_list[chunk_idx]

        # Sample the communication delay k for this sample
        delay = int(self.get_rng().integers(delay_min, delay_max + 1))

        with RmbData(self.filenames[episode_idx], self.enable_rmb_cache) as rmb_data:
            episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
            time_idx = start_time_idx
            delayed_time_idx = int(np.clip(time_idx - delay, 0, episode_len - 1))
            action_time_idxes = np.clip(
                np.arange(time_idx, time_idx + n_action_steps), 0, episode_len - 1
            )

            # Load current state (single step)
            if len(self.model_meta_info["state"]["keys"]) == 0:
                state = np.zeros(0, dtype=np.float64)
            else:
                state = np.concatenate(
                    [
                        convert_data_to_policy(
                            get_skipped_data_seq(rmb_data[key][:], key, skip)[time_idx],
                            key,
                        )
                        for key in self.model_meta_info["state"]["keys"]
                    ]
                )

            # Load ground-truth action chunk (imitation label)
            action = np.concatenate(
                [
                    convert_data_to_policy(
                        get_skipped_data_seq(rmb_data[key][:], key, skip)[
                            action_time_idxes
                        ],
                        key,
                    )
                    for key in self.model_meta_info["action"]["keys"]
                ],
                axis=1,
            )

            # Load current and delayed images
            images = self._load_images(rmb_data, skip, time_idx)
            delayed_images = self._load_images(rmb_data, skip, delayed_time_idx)

            # Load cached frozen-pi0 guidance (raw joint space) at t and t-k
            guidance_seq = rmb_data[guidance_key][::skip]
            guidance = guidance_seq[time_idx].astype(np.float64)
            delayed_guidance = guidance_seq[delayed_time_idx].astype(np.float64)

        # Normalize (guidance shares the action normalization stats)
        state = normalize_data(state, self.model_meta_info["state"])
        action = normalize_data(action, self.model_meta_info["action"])
        guidance = normalize_data(guidance, self.model_meta_info["action"])
        delayed_guidance = normalize_data(
            delayed_guidance, self.model_meta_info["action"]
        )

        # Reweight: emphasize samples where stale vs current guidance diverge.
        # NOTE: This is an adaptation, not a faithful port. In the AsyncVLA paper
        # the reactivity is measured by the shift of the robot's moving local
        # frame between t and t-k; a fixed-base arm has no such shift, so we use
        # the divergence of pi0's prediction as a delay-sensitivity proxy.
        reactivity = float(np.linalg.norm(guidance - delayed_guidance))
        weight = 1.0 + reweight_gain * float(reactivity > dth)

        # Build the 6-channel delta image: concat(I_t, I_{t-k})
        images = np.moveaxis(images, -1, -3)  # (num_images, 3, H, W)
        delayed_images = np.moveaxis(delayed_images, -1, -3)
        delta_image = np.concatenate([images, delayed_images], axis=1)

        # Convert to tensors
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        guidance_tensor = torch.tensor(delayed_guidance, dtype=torch.float32)
        weight_tensor = torch.tensor(weight, dtype=torch.float32)
        images_tensor = torch.tensor(images.copy(), dtype=torch.uint8)
        delta_tensor = torch.tensor(delta_image.copy(), dtype=torch.uint8)

        # Apply image transforms.
        # Current images use the configured transforms (incl. augmentation);
        # the 6-channel delta image only gets scaled (3-channel-only ops such as
        # color jitter cannot be applied to a 6-channel tensor).
        images_tensor = torch.stack(
            [self.image_transforms(image) for image in images_tensor]
        )
        delta_tensor = delta_tensor.to(torch.float32) / 255.0

        # Augment state / action
        if self.model_meta_info["state"]["aug_std"] > 0.0:
            state_tensor += self.model_meta_info["state"]["aug_std"] * torch.randn_like(
                state_tensor
            )
        if self.model_meta_info["action"]["aug_std"] > 0.0:
            action_tensor += self.model_meta_info["action"][
                "aug_std"
            ] * torch.randn_like(action_tensor)

        return (
            state_tensor,
            images_tensor,
            delta_tensor,
            guidance_tensor,
            action_tensor,
            weight_tensor,
        )

    def _load_images(self, rmb_data, skip, time_idx):
        return np.stack(
            [
                rmb_data[DataKey.get_rgb_image_key(camera_name)][::skip][time_idx]
                for camera_name in self.model_meta_info["image"]["camera_names"]
            ],
            axis=0,
        )
