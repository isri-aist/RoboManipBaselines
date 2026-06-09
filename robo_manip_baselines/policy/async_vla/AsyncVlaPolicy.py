import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.ops.misc import FrozenBatchNorm2d


class AsyncVlaPolicy(nn.Module):
    """Edge Adapter policy for AsyncVLA.

    Lightweight policy that refines a base-VLA (pi0) action chunk using the
    current observation (the onboard "Edge Adapter" of AsyncVLA).

    Inputs (all batched, B = batch size):
      - state:       (B, state_dim)             current proprioception
      - images:      (B, num_images, 3, H, W)   current images I_t
      - delta_image: (B, num_images, 6, H, W)   concat(I_t, I_{t-k}) motion cue
      - guidance:    (B, N, action_dim)         stale base-VLA action chunk (t-k)

    Output:
      - action_seq:  (B, N, action_dim)         refined action chunk

    Only the current observation feeds the action head's perception path, so
    the output is conditioned on the latest state rather than the stale guidance.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        num_images,
        n_action_steps,
        state_feature_dim=128,
        guidance_feature_dim=256,
        delta_feature_dim=128,
        hidden_dim_list=(512, 512),
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_images = num_images
        self.n_action_steps = n_action_steps

        # State feature extractor
        self.state_feature_extractor = nn.Sequential(
            nn.Linear(state_dim, state_feature_dim),
            nn.ReLU(),
        )

        # Current-image feature extractor (ResNet18 backbone, the main perception path)
        resnet_model = resnet18(
            weights=ResNet18_Weights.DEFAULT, norm_layer=FrozenBatchNorm2d
        )
        self.image_feature_extractor = nn.Sequential(
            *list(resnet_model.children())[:-1]
        )  # Remove last layer
        image_feature_dim = resnet_model.fc.in_features

        # Delta-image feature extractor (6-channel input; optical-flow token replacement)
        self.delta_feature_extractor = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, delta_feature_dim),
            nn.ReLU(),
        )

        # Guidance projector (the Stage-1 "token projector" operating on the action chunk)
        self.guidance_projector = nn.Sequential(
            nn.Linear(n_action_steps * action_dim, guidance_feature_dim),
            nn.ReLU(),
            nn.Linear(guidance_feature_dim, guidance_feature_dim),
            nn.ReLU(),
        )

        # Fusion + action head
        combined_feature_dim = (
            state_feature_dim
            + num_images * image_feature_dim
            + num_images * delta_feature_dim
            + guidance_feature_dim
        )
        linear_dim_list = (
            [combined_feature_dim]
            + list(hidden_dim_list)
            + [action_dim * n_action_steps]
        )
        linear_layers = []
        for linear_idx in range(len(linear_dim_list) - 1):
            input_dim = linear_dim_list[linear_idx]
            output_dim = linear_dim_list[linear_idx + 1]
            linear_layers.append(nn.Linear(input_dim, output_dim))
            if linear_idx < len(linear_dim_list) - 2:
                linear_layers.append(nn.ReLU())
        self.linear_layer_seq = nn.Sequential(*linear_layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _extract_multi_image_feature(self, extractor, images):
        # images: (B, num_images, C, H, W) -> (B, num_images * feature_dim)
        batch_size, num_images = images.shape[:2]
        features = []
        for i in range(num_images):
            feature = extractor(images[:, i])
            features.append(feature.reshape(batch_size, -1))
        return torch.cat(features, dim=1)

    def forward(self, state, images, delta_image, guidance):
        batch_size = state.shape[0]

        # Extract features
        state_feature = self.state_feature_extractor(state)
        image_feature = self._extract_multi_image_feature(
            self.image_feature_extractor, images
        )
        delta_feature = self._extract_multi_image_feature(
            self.delta_feature_extractor, delta_image
        )
        guidance_feature = self.guidance_projector(guidance.reshape(batch_size, -1))

        # Fuse features and predict the refined action chunk
        combined_feature = torch.cat(
            [state_feature, image_feature, delta_feature, guidance_feature], dim=1
        )
        action_seq = self.linear_layer_seq(combined_feature)
        action_seq = action_seq.reshape(
            batch_size, self.n_action_steps, self.action_dim
        )

        return action_seq
