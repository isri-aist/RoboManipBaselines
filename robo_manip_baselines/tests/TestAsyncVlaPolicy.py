"""Standalone shape test for the AsyncVLA Edge Adapter policy (transformer port).

This does not require a dataset or a checkpoint; it only checks that the
``AsyncVlaPolicy`` transformer forward pass produces an action chunk of the
expected shape, given pi0 action-expert hidden-state embeddings as guidance.

Usage:
    uv run python ./tests/TestAsyncVlaPolicy.py
"""

import torch

from robo_manip_baselines.policy.async_vla import AsyncVlaPolicy


def main():
    batch_size = 4
    state_dim = 7
    action_dim = 7
    num_images = 2
    n_action_steps = 8
    n_guidance_tokens = 16  # pi0 chunk_size
    guidance_embed_dim = 1024  # pi0 gemma_300m action-expert width
    height = width = 96

    policy = AsyncVlaPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        num_images=num_images,
        n_action_steps=n_action_steps,
        guidance_embed_dim=guidance_embed_dim,
        n_guidance_tokens=n_guidance_tokens,
    )
    policy.eval()

    state = torch.randn(batch_size, state_dim)
    images = torch.rand(batch_size, num_images, 3, height, width)
    delta_image = torch.rand(batch_size, num_images, 6, height, width)
    # guidance is now pi0 action-expert hidden states (B, n_guidance_tokens, D_embed)
    guidance = torch.randn(batch_size, n_guidance_tokens, guidance_embed_dim)

    with torch.inference_mode():
        action_seq = policy(state, images, delta_image, guidance)

    expected_shape = (batch_size, n_action_steps, action_dim)
    assert (
        tuple(action_seq.shape) == expected_shape
    ), f"Unexpected output shape: {tuple(action_seq.shape)} != {expected_shape}"

    n_params = sum(p.numel() for p in policy.parameters())
    print(
        f"[TestAsyncVlaPolicy] OK: output shape {tuple(action_seq.shape)}, "
        f"params {n_params / 1e6:.2f}M"
    )


if __name__ == "__main__":
    main()
