import collections
import os
import sys
import threading
from contextlib import nullcontext

import numpy as np
import torch

from robo_manip_baselines.common import (
    DataKey,
    RolloutBase,
    denormalize_data,
    normalize_data,
)

from .AsyncVlaPolicy import AsyncVlaPolicy


class _Pi0GuidanceHook:
    """forward_pre_hook on pi0's ``action_out_proj`` that captures ``suffix_out``
    (the action-expert hidden states, ``(B, chunk_size, expert_width)``) on every
    denoise step; the last capture is the final denoise step used as guidance."""

    def __init__(self):
        self.captured = None

    def __call__(self, module, args):
        self.captured = args[0].detach()


class RolloutAsyncVla(RolloutBase):
    """Rollout of AsyncVLA with a threaded asynchronous base VLA (pi0).

    The frozen base VLA (pi0) runs in a background thread on the most recent
    observation snapshot, publishing a (possibly stale) action-chunk guidance.
    The lightweight Edge Adapter runs every control step, refining the latest
    guidance using the current observation.

    The control-step index (``rollout_time_idx``) is used as the timestamp, so
    the guidance staleness is expressed in control steps and matches the delay
    range injected during training.

    NOTE: All lerobot / pi0 imports are kept inside ``setup_policy`` so that the
    training entry point (which imports this package) does not require lerobot.
    """

    require_task_desc = True

    def set_additional_args(self, parser):
        # TODO: Disable rendering with matplotlib and cv2, as it causes the program to hang
        parser.set_defaults(no_plot=True)

        parser.add_argument(
            "--base_checkpoint",
            type=str,
            required=True,
            help="checkpoint directory of the frozen base VLA (pi0)",
        )
        parser.add_argument(
            "--image_ring_size",
            type=int,
            default=64,
            help="size of the timestamped image ring buffer for delayed-image lookup",
        )

    def setup_model_meta_info(self):
        super().setup_model_meta_info()

        # The Edge Adapter runs every control step
        self.args.skip = 1
        self.args.skip_draw = 1

        self.n_action_steps = self.model_meta_info["data"]["n_action_steps"]

    def setup_policy(self):
        # Construct and load the Edge Adapter (this is self.policy)
        self.print_policy_info()
        self.policy = AsyncVlaPolicy(
            self.state_dim,
            self.action_dim,
            len(self.camera_names),
            **self.model_meta_info["policy"]["args"],
        )
        self.load_ckpt()

        # Load the frozen base VLA (pi0)
        self.setup_base_vla()

        # AsyncVLA Stage-1 is joint-position only. The offline guidance cache
        # (misc/AddPi0GuidanceToRmbData.py) always feeds pi0 MEASURED_JOINT_POS and
        # stores a COMMAND_JOINT_POS chunk, and get_state_array() forwards the raw
        # env state to pi0 without convert_data_to_policy. Any other key -- even a
        # same-dimensional one (joint velocity, wrench, EEF pose, ...) -- would feed
        # pi0/the Edge Adapter a representation it was not trained on, so require the
        # joint-position keys exactly. This also guarantees state_dim == action_dim
        # for the hold-position guidance fallback.
        if self.state_keys != [DataKey.MEASURED_JOINT_POS]:
            raise ValueError(
                f"[{self.__class__.__name__}] AsyncVLA (Stage-1) requires "
                f"state_keys == ['{DataKey.MEASURED_JOINT_POS}'] (the base-VLA guidance "
                f"is cached from MEASURED_JOINT_POS), but got {self.state_keys}."
            )
        if self.action_keys != [DataKey.COMMAND_JOINT_POS]:
            raise ValueError(
                f"[{self.__class__.__name__}] AsyncVLA (Stage-1) requires "
                f"action_keys == ['{DataKey.COMMAND_JOINT_POS}'], but got {self.action_keys}."
            )

        # Setup threading shared state
        self.guidance_lock = threading.Lock()
        self.obs_lock = threading.Lock()
        self.latest_obs_snapshot = None
        self.latest_guidance = None
        self.image_ring = collections.deque(maxlen=self.args.image_ring_size)
        self.worker_stop = threading.Event()
        self.worker_thread = None
        self.worker_started = False
        self.episode_generation = 0

        # Async sanity statistics
        self.base_infer_count = 0
        self.staleness_list = []

        # Base-VLA worker health. The worker is the sole pi0 caller; if it crashes,
        # infer_policy would otherwise hold position forever and the run would be
        # misrecorded as a policy failure. Capture the worker exception and the
        # consecutive hold count so the control loop can fail loudly / warn instead.
        self.worker_exc = None
        self._hold_steps = 0
        # Diagnostic only (does NOT change the command): warn once per episode if no
        # guidance has arrived after this many consecutive hold steps. Normal warm-up
        # is a few steps, so a large count means the base VLA is stuck or far too slow.
        self._hold_warn_steps = 100

    def setup_base_vla(self):
        # Import lerobot lazily (see class docstring)
        sys.path.append(
            os.path.join(os.path.dirname(__file__), "../../../third_party/lerobot")
        )
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        from lerobot.policies.utils import prepare_observation_for_inference

        self.prepare_observation_for_inference = prepare_observation_for_inference
        self.base_policy = PI0Policy.from_pretrained(self.args.base_checkpoint)
        self.base_preprocess, self.base_postprocess = make_pre_post_processors(
            self.base_policy.config,
            self.args.base_checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        self.base_policy.eval()

        # Register the hidden-state guidance hook (faithful AsyncVLA guidance =
        # pi0 action-expert hidden states, captured at the final denoise step).
        self.guidance_hook = _Pi0GuidanceHook()
        self.base_policy.model.action_out_proj.register_forward_pre_hook(
            self.guidance_hook
        )
        self.guidance_embed_dim = self.base_policy.model.action_out_proj.in_features
        self.n_guidance_tokens = self.base_policy.config.chunk_size
        meta_n_guid = self.model_meta_info["data"].get("n_guidance_tokens")
        meta_embed = self.model_meta_info["data"].get("guidance_embed_dim")
        if meta_n_guid is not None and meta_n_guid != self.n_guidance_tokens:
            raise ValueError(
                f"[{self.__class__.__name__}] Base VLA chunk_size "
                f"{self.n_guidance_tokens} != trained n_guidance_tokens {meta_n_guid}. "
                "The rollout base VLA must match the one used for guidance caching."
            )
        if meta_embed is not None and meta_embed != self.guidance_embed_dim:
            raise ValueError(
                f"[{self.__class__.__name__}] Base VLA expert width "
                f"{self.guidance_embed_dim} != trained guidance_embed_dim {meta_embed}."
            )

        # Cameras expected by pi0 (must align with the Edge Adapter cameras)
        self.base_camera_names = [
            key.replace("observation.images.", "").replace("_rgb", "")
            for key in self.base_policy.config.input_features
            if key.startswith("observation.images.")
        ]
        # The base VLA reads images by base_camera_names from an images_dict built
        # only from the Edge Adapter cameras, so every base camera must be present
        # there; otherwise the first base inference raises KeyError. Fail fast.
        if not set(self.base_camera_names).issubset(self.camera_names):
            raise ValueError(
                f"[{self.__class__.__name__}] Base VLA cameras {self.base_camera_names} "
                f"are not a subset of the Edge Adapter cameras {self.camera_names}. "
                "Provide all base-VLA cameras via --camera_names."
            )

        # The base VLA must share the Edge Adapter's state/action space (the base
        # VLA must be trained on this task's COMMAND_JOINT_POS actions).
        base_state_dim = self.base_policy.config.input_features[
            "observation.state"
        ].shape[0]
        base_action_dim = self.base_policy.config.output_features["action"].shape[0]
        if (base_state_dim != self.state_dim) or (base_action_dim != self.action_dim):
            raise ValueError(
                f"[{self.__class__.__name__}] Base VLA state/action dims "
                f"({base_state_dim}, {base_action_dim}) do not match the Edge Adapter "
                f"({self.state_dim}, {self.action_dim}). The base VLA must be trained "
                "on the same task's measured_joint_pos / command_joint_pos."
            )

    def reset_variables(self):
        super().reset_variables()

        # Start a new episode generation so the persistent worker's in-flight result
        # (computed for the previous episode) is rejected once it is published.
        if hasattr(self, "guidance_lock"):
            self.episode_generation += 1
            with self.guidance_lock:
                self.latest_guidance = None
            with self.obs_lock:
                self.latest_obs_snapshot = None
            self.image_ring.clear()
            self._hold_steps = 0

    def run(self):
        try:
            super().run()
        finally:
            self.worker_stop.set()
            if self.worker_thread is not None:
                self.worker_thread.join(timeout=5.0)

    # ---- observation helpers (raw env units, as expected by pi0) ----
    def get_state_array(self):
        if len(self.state_keys) == 0:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(
            [
                self.motion_manager.get_data(state_key, self.obs)
                for state_key in self.state_keys
            ]
        )

    def get_images_dict(self):
        return {
            camera_name: self.info["rgb_images"][camera_name].copy()
            for camera_name in self.camera_names
        }

    # ---- base VLA inference (single integration point) ----
    def predict_base_guidance(self, state_array, images_dict):
        # Mirror lerobot's predict_action pipeline: prepare_observation_for_inference
        # converts the raw HWC uint8 images to CHW float[0,1] tensors, adds the batch
        # dimension, moves to the device and attaches the task BEFORE the preprocessor.
        # We run predict_action_chunk only to drive flow-matching denoising; the
        # predicted actions are ignored. The faithful AsyncVLA guidance is pi0's
        # action-expert hidden states (suffix_out), captured by the forward_pre_hook
        # on action_out_proj at the final denoise step.
        observation = {"observation.state": state_array}
        for camera_name in self.base_camera_names:
            observation[f"observation.images.{camera_name}_rgb"] = images_dict[
                camera_name
            ]

        use_amp = self.base_policy.config.use_amp
        autocast_ctx = (
            torch.autocast(device_type=self.device.type)
            if (self.device.type == "cuda" and use_amp)
            else nullcontext()
        )
        self.guidance_hook.captured = None
        with autocast_ctx:
            observation = self.prepare_observation_for_inference(
                observation, self.device, self.args.task_desc
            )
            batch = self.base_preprocess(observation)
            self.base_policy.predict_action_chunk(batch)
        captured = self.guidance_hook.captured
        if captured is None:
            raise RuntimeError(
                f"[{self.__class__.__name__}] pi0 guidance hook captured nothing; "
                "action_out_proj was not called during predict_action_chunk."
            )
        # (B, chunk_size, expert_width) at the final denoise step -> (chunk_size, width).
        # Keep it float32: the Edge Adapter consumes the guidance as a float32 tensor, so
        # a float64 round-trip would only add a CPU copy and double the guidance bandwidth
        # handed from the worker thread to the control loop.
        return captured[0].to(torch.float32).cpu().numpy()

    def compute_guidance(self, snapshot):
        chunk = self.predict_base_guidance(snapshot["state"], snapshot["images"])
        with self.guidance_lock:
            self.latest_guidance = {
                "chunk": chunk,
                "stamp": snapshot["stamp"],
                "gen": snapshot["gen"],
            }
        self.base_infer_count += 1

    def base_vla_worker(self):
        last_key = None
        try:
            while not self.worker_stop.is_set():
                with self.obs_lock:
                    snapshot = self.latest_obs_snapshot
                if snapshot is None:
                    self.worker_stop.wait(0.001)
                    continue
                # Skip recomputation when the observation has not advanced, so the base
                # VLA is not re-run on an identical snapshot (which would saturate the
                # GPU, starve the control loop, and inflate base_infer_count).
                key = (snapshot["gen"], snapshot["stamp"])
                if key == last_key:
                    self.worker_stop.wait(0.001)
                    continue
                last_key = key
                with torch.inference_mode():
                    self.compute_guidance(snapshot)
        except Exception as exc:
            # The worker is a daemon thread, so an uncaught exception would die
            # silently and leave the control loop holding position forever. Stash it
            # and stop; _check_base_worker_alive() re-raises it on the main thread.
            self.worker_exc = exc

    def _check_base_worker_alive(self):
        # Surface a dead/crashed base-VLA worker on the main thread instead of holding
        # position for the whole episode (which would be misrecorded as a policy
        # failure rather than the pi0 infrastructure failure it actually is).
        if self.worker_exc is not None:
            raise RuntimeError(
                f"[{self.__class__.__name__}] The base VLA worker thread crashed; no "
                "guidance can be produced (see the chained exception)."
            ) from self.worker_exc
        if (
            self.worker_started
            and not self.worker_stop.is_set()
            and self.worker_thread is not None
            and not self.worker_thread.is_alive()
        ):
            raise RuntimeError(
                f"[{self.__class__.__name__}] The base VLA worker thread exited "
                "unexpectedly; no guidance will ever arrive."
            )

    # ---- async bookkeeping ----
    def publish_snapshot(self, state_array, images_dict, stamp):
        snapshot = {
            "state": state_array.copy(),
            "images": {c: images_dict[c].copy() for c in self.camera_names},
            "stamp": stamp,
            "gen": self.episode_generation,
        }
        with self.obs_lock:
            self.latest_obs_snapshot = snapshot
        self.image_ring.append((stamp, snapshot["images"]))

    def lookup_delayed_images(self, target_stamp):
        # Nearest frame with stamp <= target_stamp (fallback: oldest available)
        best_images = None
        for stamp, images in self.image_ring:
            if stamp <= target_stamp:
                best_images = images
            else:
                break
        if best_images is None and len(self.image_ring) > 0:
            best_images = self.image_ring[0][1]
        return best_images

    def infer_policy(self):
        stamp = self.rollout_time_idx
        state_array = self.get_state_array()
        images_dict = self.get_images_dict()

        # Publish the current observation for the base VLA worker
        self.publish_snapshot(state_array, images_dict, stamp)

        # Start the base VLA worker once; it persists across episodes and is the
        # sole caller of pi0. Until it publishes the first guidance -- and at the
        # start of every episode, where reset_variables() clears the guidance --
        # infer_policy() uses the hold-position fallback below. Keeping all pi0
        # inference on the worker avoids main-thread/worker contention and makes
        # every episode start identically.
        if not self.worker_started:
            self.worker_stop.clear()
            self.worker_thread = threading.Thread(
                target=self.base_vla_worker, daemon=True
            )
            self.worker_thread.start()
            self.worker_started = True

        # Fail loudly if the sole pi0 caller has died, rather than holding forever.
        self._check_base_worker_alive()

        # Read the latest (stale) guidance, discarding any left over from a previous
        # episode (the worker persists across resets and may publish late).
        with self.guidance_lock:
            guidance = self.latest_guidance
        if guidance is None or guidance["gen"] != self.episode_generation:
            # No fresh base-VLA guidance yet (episode start / slow pi0 warm-up). The
            # Edge Adapter is trained ONLY on real (delayed) pi0 embeddings -- even the
            # smallest injected delay still indexes a real cached embedding, never a
            # zero one -- so feeding a zero embedding here is out-of-distribution and
            # can produce a lurching initial command. Hold the current joint position
            # until the worker publishes the first guidance of this episode (state_dim
            # == action_dim is enforced in setup_policy precisely for this fallback).
            self._hold_steps += 1
            if self._hold_steps == self._hold_warn_steps:
                print(
                    f"[{self.__class__.__name__}] WARNING: still no base-VLA guidance "
                    f"after {self._hold_steps} control steps; the base VLA may be far "
                    "too slow or stuck. Holding the current joint position meanwhile."
                )
            self.policy_action = state_array.copy()
            self.policy_action_list = np.concatenate(
                [self.policy_action_list, self.policy_action[np.newaxis]]
            )
            return

        self._hold_steps = 0
        guidance_chunk = guidance["chunk"]
        delayed_stamp = guidance["stamp"]
        self.staleness_list.append(stamp - delayed_stamp)

        # Retrieve I_{t-k} matching the guidance's observation timestamp
        delayed_images_dict = self.lookup_delayed_images(delayed_stamp)
        if delayed_images_dict is None:
            delayed_images_dict = images_dict

        # Run the Edge Adapter and command the first action of the refined chunk
        action_chunk = self.infer_edge_adapter(
            state_array, images_dict, delayed_images_dict, guidance_chunk
        )
        self.policy_action = denormalize_data(
            action_chunk[0], self.model_meta_info["action"]
        )
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )

    def infer_edge_adapter(
        self, state_array, images_dict, delayed_images_dict, guidance_chunk
    ):
        # State (joint-pos state keys make convert_data_to_policy an identity)
        state = normalize_data(state_array, self.model_meta_info["state"])
        state_tensor = torch.tensor(state[np.newaxis], dtype=torch.float32).to(
            self.device
        )

        # Current images (scaled to float via the rollout image transform)
        images = np.stack(
            [images_dict[c] for c in self.camera_names], axis=0
        )  # (num_images, H, W, 3)
        images = np.moveaxis(images, -1, -3)
        images_tensor = self.image_transforms(
            torch.tensor(images.copy(), dtype=torch.uint8)
        )[torch.newaxis].to(self.device)

        # Delta image: concat(I_t, I_{t-k}) along the channel axis, scaled
        delayed_images = np.stack(
            [delayed_images_dict[c] for c in self.camera_names], axis=0
        )
        delayed_images = np.moveaxis(delayed_images, -1, -3)
        delta = np.concatenate([images, delayed_images], axis=1).astype(np.float32)
        delta_tensor = (torch.tensor(delta, dtype=torch.float32) / 255.0)[
            torch.newaxis
        ].to(self.device)

        # Guidance: raw pi0 hidden-state embeddings (no action normalization)
        guidance_tensor = torch.tensor(
            guidance_chunk[np.newaxis], dtype=torch.float32
        ).to(self.device)

        action_seq = self.policy(
            state_tensor, images_tensor, delta_tensor, guidance_tensor
        )
        return action_seq[0].detach().to("cpu").numpy().astype(np.float64)

    def print_statistics(self):
        super().print_statistics()

        print(f"[{self.__class__.__name__}] Statistics on asynchronous base VLA")
        print(f"  - base VLA inference count | {self.base_infer_count}")
        print(f"  - control step count | {self.rollout_time_idx}")
        if len(self.staleness_list) > 0:
            staleness_arr = np.array(self.staleness_list)
            print(
                "  - guidance staleness [step] | "
                f"mean: {staleness_arr.mean():.2f}, max: {staleness_arr.max()}"
            )

    def draw_plot(self):
        # TODO: Disable rendering with matplotlib and cv2, as it causes the program to hang
        pass
