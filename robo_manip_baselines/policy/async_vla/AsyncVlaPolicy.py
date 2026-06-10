import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (additive), matching the official
    AsyncVLA / MBRA ``PositionalEncoding`` (NHirose, self_attention.py)."""

    def __init__(self, d_model, max_seq_len):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_len, d_model)

    def forward(self, x):  # x: (B, S, d_model)
        return x + self.pe[:, : x.size(1), :]


def _build_efficientnet_b0_encoder(in_channels):
    """EfficientNet-B0 feature extractor -> (B, 1280) per image.

    Loads ImageNet-pretrained weights; for ``in_channels != 3`` the 3-channel stem
    conv is replaced by an ``in_channels``-channel conv (re-initialised), reusing the
    rest of the pretrained backbone (matches the official 6-channel ``cat_encoder``).
    """
    net = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    if in_channels != 3:
        stem = net.features[0][0]  # Conv2d(3, 32, k3, s2, p1, bias=False)
        net.features[0][0] = nn.Conv2d(
            in_channels,
            stem.out_channels,
            kernel_size=stem.kernel_size,
            stride=stem.stride,
            padding=stem.padding,
            bias=(stem.bias is not None),
        )
    feature_dim = net.classifier[1].in_features  # 1280
    backbone = nn.Sequential(net.features, net.avgpool, nn.Flatten(start_dim=1))
    return backbone, feature_dim


class AsyncVlaPolicy(nn.Module):
    """Edge Adapter for AsyncVLA — faithful transformer port (pi0 base VLA).

    Port of the official ``Edge_adapter`` (NHirose/AsyncVLA,
    ``prismatic/models/small_head.py``) onto RoboManipBaselines + pi0:

      - **EfficientNet-B0** encoders: ``obs`` (3-ch current image I_t) and ``cat``
        (6-ch ``concat(I_t, I_{t-k})`` motion cue). Per camera, concatenated into one
        ``obs`` token and one ``cat`` token.
      - **Guidance = pi0 action-expert hidden states** ``(B, n_guidance_tokens, D_embed)``
        (the ``suffix_out`` before ``action_out_proj``), projected per token to ``d_model``.
        NOT the predicted action chunk.
      - **Transformer** (``MultiLayerDecoder_trans`` equivalent: sinusoidal pos-enc +
        ``nn.TransformerEncoder``, pre-LN, GELU) over
        ``[guidance tokens, obs token, cat token]``.
      - Select the **I_t (obs) output token** (the official ``[:, -2:-1, :]``), then an
        **MLP action head** -> ``(B, n_action_steps, action_dim)``.
      - The robot proprioceptive ``state`` is injected at the head (manipulation
        adaptation, mirroring the official ``_v0``'s ``taskid`` concatenation).

    forward inputs:
      state:       (B, state_dim)
      images:      (B, num_images, 3, H, W)   current images I_t
      delta_image: (B, num_images, 6, H, W)   concat(I_t, I_{t-k})
      guidance:    (B, n_guidance_tokens, guidance_embed_dim)
    output:
      action_seq:  (B, n_action_steps, action_dim)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        num_images,
        n_action_steps,
        guidance_embed_dim=1024,
        n_guidance_tokens=16,
        d_model=512,
        n_heads=8,
        n_layers=4,
        ff_dim_factor=4,
        dropout=0.1,
        head_hidden_dim_list=(256, 128, 64),
        use_state=True,
        image_size=96,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_images = num_images
        self.n_action_steps = n_action_steps
        self.guidance_embed_dim = guidance_embed_dim
        self.n_guidance_tokens = n_guidance_tokens
        self.d_model = d_model
        self.use_state = use_state
        # Official Edge Adapter runs on 96x96; resize inside the policy so the
        # EfficientNet stays lightweight regardless of the env's native image size
        # (the base VLA pi0 still sees full-resolution images).
        self.image_size = image_size

        # --- EfficientNet-B0 image encoders (one token each, concat over cameras) ---
        self.obs_encoder, obs_feat = _build_efficientnet_b0_encoder(3)
        self.cat_encoder, cat_feat = _build_efficientnet_b0_encoder(6)
        self.obs_proj = nn.Linear(num_images * obs_feat, d_model)
        self.cat_proj = nn.Linear(num_images * cat_feat, d_model)

        # --- guidance token projector (per token): D_embed -> d_model ---
        self.guidance_proj = nn.Linear(guidance_embed_dim, d_model)

        # --- transformer (MultiLayerDecoder_trans equivalent) ---
        max_seq_len = n_guidance_tokens + 2
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim_factor * d_model,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- action head: obs(I_t) token (+ state) -> action chunk ---
        head_in = d_model + (state_dim if use_state else 0)
        dims = [head_in] + list(head_hidden_dim_list) + [n_action_steps * action_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.action_predictor = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialise the newly added Linear layers (EfficientNet keeps pretrained weights).
        for module in [self.obs_proj, self.cat_proj, self.guidance_proj]:
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            nn.init.zeros_(module.bias)
        for m in self.action_predictor:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        # Re-init the replaced 6-channel stem conv of the cat encoder.
        cat_stem = self.cat_encoder[0][0][0]
        nn.init.kaiming_normal_(cat_stem.weight, mode="fan_out", nonlinearity="relu")

    def _encode_multi_image(self, encoder, proj, images):
        # images: (B, num_images, C, H, W) -> (B, d_model)
        # Resize each camera image to image_size x image_size (official 96x96).
        batch_size, num_images = images.shape[:2]
        feats = []
        for i in range(num_images):
            img = images[:, i]
            if img.shape[-2:] != (self.image_size, self.image_size):
                img = F.interpolate(
                    img,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            feats.append(encoder(img).reshape(batch_size, -1))
        return proj(torch.cat(feats, dim=1))

    def forward(self, state, images, delta_image, guidance):
        batch_size = state.shape[0]

        obs_token = self._encode_multi_image(self.obs_encoder, self.obs_proj, images)
        cat_token = self._encode_multi_image(self.cat_encoder, self.cat_proj, delta_image)
        guidance_tokens = self.guidance_proj(guidance)  # (B, n_guid, d_model)

        # token order matches the official: [guidance tokens..., obs token, cat token]
        tokens = torch.cat(
            [guidance_tokens, obs_token.unsqueeze(1), cat_token.unsqueeze(1)], dim=1
        )
        tokens = self.positional_encoding(tokens)
        tokens = self.transformer(tokens)

        # select the I_t (obs) output token == official's [:, -2:-1, :]
        obs_out = tokens[:, -2, :]  # (B, d_model)

        if self.use_state:
            obs_out = torch.cat([obs_out, state], dim=1)
        action_seq = self.action_predictor(obs_out).reshape(
            batch_size, self.n_action_steps, self.action_dim
        )
        return action_seq
