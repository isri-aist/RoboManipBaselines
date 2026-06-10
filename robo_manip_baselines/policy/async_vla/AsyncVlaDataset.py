import cv2
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
        # Expected guidance dims sized from the first file by TrainAsyncVla; every
        # episode must match exactly (see the per-file validation below).
        expected_tokens = self.model_meta_info["data"].get("n_guidance_tokens")
        expected_embed = self.model_meta_info["data"].get("guidance_embed_dim")
        tokens_attr = guidance_key + "_n_guidance_tokens"
        embed_attr = guidance_key + "_embed_dim"

        self.chunk_info_list = []
        for episode_idx, filename in enumerate(self.filenames):
            with RmbData(filename) as rmb_data:
                if guidance_key not in rmb_data.keys():
                    raise ValueError(
                        f"[{self.__class__.__name__}] Cached guidance '{guidance_key}' "
                        f"not found in {filename}. Run misc/AddPi0GuidanceToRmbData.py first."
                    )
                # Validate EVERY episode, not just the first one TrainAsyncVla sized
                # from: a mixed dataset (some files regenerated as hidden-state
                # guidance, some still holding pre-port action chunks under the same
                # key) would otherwise pass setup and then crash mid-epoch or, worse,
                # silently train on the wrong signal where shapes happen to collide.
                if (
                    tokens_attr not in rmb_data.attrs
                    or embed_attr not in rmb_data.attrs
                ):
                    raise ValueError(
                        f"[{self.__class__.__name__}] Cached guidance '{guidance_key}' in "
                        f"{filename} is missing the '{tokens_attr}' / '{embed_attr}' "
                        "attributes written by the hidden-state guidance cache. It was "
                        "likely produced by an older action-chunk implementation. Re-run "
                        "misc/AddPi0GuidanceToRmbData.py with --overwrite to regenerate it."
                    )
                guidance_shape = rmb_data[guidance_key].shape
                if (
                    len(guidance_shape) != 3
                    or (
                        expected_tokens is not None
                        and int(guidance_shape[1]) != expected_tokens
                    )
                    or (
                        expected_embed is not None
                        and int(guidance_shape[2]) != expected_embed
                    )
                ):
                    raise ValueError(
                        f"[{self.__class__.__name__}] Cached guidance '{guidance_key}' in "
                        f"{filename} has shape {tuple(guidance_shape)}, expected "
                        f"(T, {expected_tokens}, {expected_embed}). Regenerate it with "
                        "misc/AddPi0GuidanceToRmbData.py --overwrite so all episodes match."
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

    def _get_episode(self, episode_idx):
        """Per-worker resident episode cache.

        Reading from the RMB files per sample is prohibitively slow on network
        storage (h5py reopens the file every sample, and CIFS drops its page cache
        on close, so e.g. the (T, 16, 1024) guidance would be re-fetched over the
        network for EVERY sample). Instead, each DataLoader worker lazily loads an
        episode ONCE into RAM (images downscaled to the policy input size as uint8,
        guidance as float16) and serves all samples from memory. With
        ``persistent_workers=True`` the cache lives for the whole training run.

        Memory note: the cache is intentionally never evicted -- LRU eviction would
        re-introduce exactly the per-sample network reads this exists to avoid. The
        footprint is therefore bounded by (downscaled dataset size) x (num_workers):
        every worker eventually caches every episode it is handed. Because the images
        are stored at ``image_size`` (96x96 uint8 by default) the dominant per-episode
        cost is the float16 guidance, so a full dataset stays in the low GB per worker.
        If RAM is tight, reduce ``--num_workers`` rather than capping this cache.
        """
        if not hasattr(self, "_episode_cache"):
            self._episode_cache = {}
        if episode_idx in self._episode_cache:
            return self._episode_cache[episode_idx]

        data_meta_info = self.model_meta_info["data"]
        skip = data_meta_info["skip"]
        guidance_key = data_meta_info["guidance_key"]
        image_size = (
            self.model_meta_info.get("policy", {}).get("args", {}).get("image_size")
        )
        rmb_image_size = (image_size, image_size) if image_size else None

        with RmbData(
            self.filenames[episode_idx], False, image_size=rmb_image_size
        ) as rmb_data:
            episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
            if len(self.model_meta_info["state"]["keys"]) == 0:
                state_seq = np.zeros((episode_len, 0), dtype=np.float64)
            else:
                state_seq = np.concatenate(
                    [
                        convert_data_to_policy(
                            get_skipped_data_seq(rmb_data[key][:], key, skip), key
                        )
                        for key in self.model_meta_info["state"]["keys"]
                    ],
                    axis=1,
                )
            action_seq = np.concatenate(
                [
                    convert_data_to_policy(
                        get_skipped_data_seq(rmb_data[key][:], key, skip), key
                    )
                    for key in self.model_meta_info["action"]["keys"]
                ],
                axis=1,
            )
            # RmbData applies ``image_size`` only to the ``.rmb`` video format: for
            # single-HDF5 datasets ``__getitem__`` returns the raw full-resolution h5py
            # dataset and ignores it. The ``.rmb`` path therefore yields frames already
            # downscaled to image_size, so a single bulk strided read is fine. For HDF5
            # we must NOT bulk-read ``[::skip]`` -- that materializes the entire
            # full-resolution episode (per camera, per worker) before downscaling, a
            # transient spike that can OOM on large videos / network storage. Instead
            # read+resize one frame at a time into a preallocated downscaled buffer, so
            # the peak is a single frame and the resident cache holds only image_size
            # frames.
            images_seq = {}
            for camera_name in self.model_meta_info["image"]["camera_names"]:
                rgb = rmb_data[DataKey.get_rgb_image_key(camera_name)]
                if rmb_image_size is not None and rmb_data.is_single_hdf5:
                    src_indices = range(0, rgb.shape[0], skip)
                    frames = np.empty(
                        (len(src_indices), rmb_image_size[1], rmb_image_size[0], 3),
                        dtype=np.uint8,
                    )
                    for out_idx, src_idx in enumerate(src_indices):
                        frames[out_idx] = cv2.resize(
                            rgb[src_idx],
                            rmb_image_size,
                            interpolation=cv2.INTER_LINEAR,
                        )
                else:
                    frames = np.asarray(rgb[::skip])
                images_seq[camera_name] = frames
            guidance_seq = np.asarray(rmb_data[guidance_key][::skip]).astype(np.float16)

        episode = {
            "episode_len": episode_len,
            "state_seq": state_seq,
            "action_seq": action_seq,
            "images_seq": images_seq,
            "guidance_seq": guidance_seq,
        }
        self._episode_cache[episode_idx] = episode
        return episode

    def __getitem__(self, chunk_idx):
        data_meta_info = self.model_meta_info["data"]
        n_action_steps = data_meta_info["n_action_steps"]
        delay_min = data_meta_info["delay_min"]
        delay_max = data_meta_info["delay_max"]
        dth = data_meta_info["dth"]
        reweight_gain = data_meta_info["reweight_gain"]
        episode_idx, start_time_idx = self.chunk_info_list[chunk_idx]

        # Sample the communication delay k for this sample
        delay = int(self.get_rng().integers(delay_min, delay_max + 1))

        # All per-episode data comes from the per-worker resident cache (zero
        # filesystem access on cache hit; see _get_episode).
        episode = self._get_episode(episode_idx)
        episode_len = episode["episode_len"]
        time_idx = start_time_idx
        delayed_time_idx = int(np.clip(time_idx - delay, 0, episode_len - 1))
        action_time_idxes = np.clip(
            np.arange(time_idx, time_idx + n_action_steps), 0, episode_len - 1
        )

        # Current state (single step) and ground-truth action chunk
        state = episode["state_seq"][time_idx]
        action = episode["action_seq"][action_time_idxes]

        # Current and delayed images: (num_images, H, W, 3) uint8
        camera_names = self.model_meta_info["image"]["camera_names"]
        images = np.stack(
            [episode["images_seq"][c][time_idx] for c in camera_names], axis=0
        )
        delayed_images = np.stack(
            [episode["images_seq"][c][delayed_time_idx] for c in camera_names], axis=0
        )

        # Cached frozen-pi0 hidden-state guidance (chunk_size, embed_dim) at t and
        # t-k. These are pi0 action-expert embeddings, NOT actions. The cache is
        # float16 (memory); float32 is ample for both the reactivity norm below and
        # the model input (the guidance tensor is float32 anyway).
        guidance = episode["guidance_seq"][time_idx].astype(np.float32)
        delayed_guidance = episode["guidance_seq"][delayed_time_idx].astype(np.float32)

        # Normalize state/action only; guidance is raw pi0 hidden-state embeddings
        # (the Edge Adapter projects them with a learned Linear, so no action stats).
        state = normalize_data(state, self.model_meta_info["state"])
        action = normalize_data(action, self.model_meta_info["action"])

        # Reweight: emphasize samples where stale vs current guidance diverge.
        # NOTE: This is an adaptation, not a faithful port. In the AsyncVLA paper
        # the reactivity is measured by the shift of the robot's moving local
        # frame between t and t-k; a fixed-base arm has no such shift, so we use
        # the divergence of pi0's guidance embedding as a delay-sensitivity proxy.
        # Use a scale-invariant relative norm (embeddings are unnormalized), so the
        # threshold ``dth`` stays meaningful regardless of the embedding magnitude.
        reactivity = float(np.linalg.norm(guidance - delayed_guidance)) / (
            float(np.linalg.norm(guidance)) + 1e-6
        )
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
