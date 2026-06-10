import argparse
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm

from robo_manip_baselines.common import DataKey, RmbData, find_rmb_files


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "path",
        type=str,
        help="path to data (*.hdf5 or *.rmb) or directory containing them",
    )
    parser.add_argument(
        "--base_checkpoint",
        type=str,
        required=True,
        help="checkpoint directory of the frozen base VLA (pi0)",
    )
    parser.add_argument(
        "--task_desc",
        type=str,
        required=True,
        help="task description passed to the base VLA",
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=["front", "hand"],
        help="camera names (must match the base VLA config)",
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=8,
        help="(metadata only) Edge Adapter output chunk length; the cached guidance "
        "always spans pi0's full chunk_size action tokens",
    )
    parser.add_argument(
        "--guidance_key",
        type=str,
        default="pi0_guidance",
        help="key under which the guidance is stored in the RMB data",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to run the base VLA on",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="whether to overwrite existing guidance if it exists",
    )

    return parser.parse_args()


class _Pi0GuidanceHook:
    """forward_pre_hook on pi0's ``action_out_proj``.

    During flow-matching denoising, ``denoise_step`` computes
    ``suffix_out = outputs_embeds[1][:, -chunk_size:].to(float32)`` (the action-expert
    hidden states, shape ``(B, chunk_size, expert_width)``) and feeds it to
    ``action_out_proj`` to predict the velocity. This pre-hook captures that
    ``suffix_out`` input on every denoise step; the last capture corresponds to the
    final denoise step, which is what AsyncVLA uses as the base-VLA "guidance" token
    sequence (NOT the predicted action chunk).
    """

    def __init__(self):
        self.captured = None

    def __call__(self, module, args):
        # args[0] == suffix_out, shape (B, chunk_size, expert_width)
        self.captured = args[0].detach()


class AddPi0GuidanceToRmbData:
    """Precompute and cache the frozen-pi0 *hidden-state* guidance per timestep.

    Faithful AsyncVLA port: the base-VLA guidance is the action-expert hidden states
    (``suffix_out``, the input to ``action_out_proj``) at the final denoise step, of
    shape ``(chunk_size, expert_width)`` (e.g. ``(16, 1024)`` for pi0 with a
    gemma_300m action expert), NOT the predicted action chunk. The Edge Adapter
    (``policy/async_vla``) consumes these embeddings as guidance tokens.

    For every timestep of every episode it runs the frozen base VLA (pi0) and stores
    the captured hidden states so that the Edge Adapter training can inject arbitrary
    delays by indexing into the cache.
    """

    def __init__(
        self,
        path,
        base_checkpoint,
        task_desc,
        camera_names,
        n_action_steps,
        guidance_key,
        device,
        overwrite=False,
    ):
        self.path = path
        self.base_checkpoint = base_checkpoint
        self.task_desc = task_desc
        self.camera_names = camera_names
        self.n_action_steps = n_action_steps
        self.guidance_key = guidance_key
        self.device = torch.device(device)
        self.overwrite = overwrite

        self.setup_base_vla()

    def setup_base_vla(self):
        sys.path.append(
            os.path.join(os.path.dirname(__file__), "../../third_party/lerobot")
        )
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        from lerobot.policies.utils import prepare_observation_for_inference

        self.prepare_observation_for_inference = prepare_observation_for_inference
        self.policy = PI0Policy.from_pretrained(self.base_checkpoint)
        self.preprocess, _ = make_pre_post_processors(
            self.policy.config,
            self.base_checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        self.policy.eval()

        # Register the hidden-state capture hook on pi0's action_out_proj.
        self.guidance_hook = _Pi0GuidanceHook()
        self.policy.model.action_out_proj.register_forward_pre_hook(self.guidance_hook)
        self.expert_width = self.policy.model.action_out_proj.in_features
        self.chunk_size = self.policy.config.chunk_size
        print(
            f"[{self.__class__.__name__}] pi0 guidance = action-expert hidden states "
            f"(chunk_size={self.chunk_size}, expert_width={self.expert_width})"
        )

    def capture_guidance(self, state_array, images_dict):
        # Mirror lerobot's predict_action pipeline: prepare_observation_for_inference
        # converts raw HWC uint8 images to CHW float[0,1] tensors, adds the batch
        # dimension, moves to the device and attaches the task BEFORE the preprocessor.
        # We run predict_action_chunk only to trigger denoising; the action output is
        # ignored and the guidance is the suffix_out captured by the hook (final step).
        observation = {"observation.state": state_array}
        for camera_name in self.camera_names:
            observation[f"observation.images.{camera_name}_rgb"] = images_dict[
                camera_name
            ]

        use_amp = self.policy.config.use_amp
        autocast_ctx = (
            torch.autocast(device_type=self.device.type)
            if (self.device.type == "cuda" and use_amp)
            else nullcontext()
        )
        self.guidance_hook.captured = None
        with autocast_ctx:
            observation = self.prepare_observation_for_inference(
                observation, self.device, self.task_desc
            )
            batch = self.preprocess(observation)
            self.policy.predict_action_chunk(batch)

        captured = self.guidance_hook.captured
        if captured is None:
            raise RuntimeError(
                f"[{self.__class__.__name__}] pi0 guidance hook captured nothing; "
                "action_out_proj was not called during predict_action_chunk."
            )
        # (B, chunk_size, expert_width) at the final denoise step -> (chunk_size, width).
        # Stored as float32: half the size/IO of float64 with ample precision for the
        # Edge Adapter (which consumes the guidance as a float32 tensor anyway).
        guidance = captured[0].to(torch.float32).cpu().numpy()
        return guidance

    def run(self):
        print(
            f"[{self.__class__.__name__}] Add base VLA hidden-state guidance "
            f"'{self.guidance_key}' generated by the frozen pi0 policy."
        )
        rmb_path_list = find_rmb_files(self.path)
        for rmb_path in tqdm(rmb_path_list):
            tqdm.write(f"[{self.__class__.__name__}] Open {rmb_path}")
            with RmbData(rmb_path, mode="r+") as rmb_data:
                if self.guidance_key in rmb_data.keys():
                    if self.overwrite:
                        del rmb_data.h5file[self.guidance_key]
                    else:
                        raise ValueError(
                            f"[{self.__class__.__name__}] Guidance already exists: "
                            f"{rmb_path} (use --overwrite to replace)"
                        )

                guidance_seq = self.get_guidance_seq(rmb_data)

                rmb_data.h5file[self.guidance_key] = guidance_seq
                rmb_data.attrs[self.guidance_key + "_n_guidance_tokens"] = (
                    guidance_seq.shape[1]
                )
                rmb_data.attrs[self.guidance_key + "_embed_dim"] = guidance_seq.shape[2]
                rmb_data.attrs[self.guidance_key + "_n_action_steps"] = (
                    self.n_action_steps
                )
                rmb_data.attrs[self.guidance_key + "_camera_names"] = list(
                    self.camera_names
                )
                rmb_data.attrs[self.guidance_key + "_task_desc"] = self.task_desc

    def get_guidance_seq(self, rmb_data):
        joint_pos_seq = rmb_data[DataKey.MEASURED_JOINT_POS][:]
        rgb_image_seqs = {
            camera_name: rmb_data[DataKey.get_rgb_image_key(camera_name)][:]
            for camera_name in self.camera_names
        }
        episode_len = len(joint_pos_seq)

        if hasattr(self.policy, "reset"):
            self.policy.reset()

        guidance_seq = []
        for time_idx in range(episode_len):
            state_array = joint_pos_seq[time_idx]
            images_dict = {
                camera_name: rgb_image_seqs[camera_name][time_idx]
                for camera_name in self.camera_names
            }
            with torch.inference_mode():
                guidance = self.capture_guidance(state_array, images_dict)
            guidance_seq.append(guidance)

        # (episode_len, chunk_size, expert_width), float32
        return np.asarray(guidance_seq, dtype=np.float32)


if __name__ == "__main__":
    add_pi0_guidance = AddPi0GuidanceToRmbData(**vars(parse_argument()))
    add_pi0_guidance.run()
