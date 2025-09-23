import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.ops.misc import FrozenBatchNorm2d


class MlpPolicy(nn.Module):
    """MLP policy with ResNet backbone."""

    def __init__(
        self,
        state_dim,
        action_dim,
        num_images,
        n_obs_steps,
        n_action_steps,
        hidden_dim_list,
        state_feature_dim,
    ):
        super().__init__()

        # Setup Variable
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps

        # Instantiate state feature extractor
        self.state_feature_extractor = nn.Sequential(
            nn.Linear(state_dim * self.n_obs_steps, state_feature_dim),
            # nn.BatchNorm1d(state_feature_dim),
            nn.ReLU(),
        )

        # Instantiate image feature extractor
        resnet_model = resnet18(
            weights=ResNet18_Weights.DEFAULT, norm_layer=FrozenBatchNorm2d
        )
        self.image_feature_extractor = nn.Sequential(
            *list(resnet_model.children())[:-1]
        )  # Remove last layer
        image_feature_dim = resnet_model.fc.in_features

        # Instantiate linear layers
        combined_feature_dim = (
            state_feature_dim + num_images * n_obs_steps * image_feature_dim
        )
        linear_dim_list = (
            [combined_feature_dim]
            + hidden_dim_list
            + [action_dim * self.n_action_steps]
        )
        linear_layers = []
        for linear_idx in range(len(linear_dim_list) - 1):
            input_dim = linear_dim_list[linear_idx]
            output_dim = linear_dim_list[linear_idx + 1]
            linear_layers += [nn.Linear(input_dim, output_dim)]
            if linear_idx < len(linear_dim_list) - 2:
                linear_layers += [
                    # nn.BatchNorm1d(output_dim),
                    nn.ReLU(),
                ]
        self.linear_layer_seq = nn.Sequential(*linear_layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state_seq, images_seq):
        batch_size, _, _, C, H, W = images_seq.shape

        # Reshape state_seq and images_seq
        state_seq = state_seq.reshape(batch_size, -1)
        images_seq = images_seq.reshape(batch_size, -1, C, H, W)

        # Extract state feature
        state_feature = self.state_feature_extractor(
            state_seq
        )  # (batch_size, state_feature_dim)

        # Extract image feature
        image_features = []

        for i in range(images_seq.shape[1]):
            image_feature = self.image_feature_extractor(
                images_seq[:, i]
            )  # (batch_size, image_feature_dim, 1, 1)
            image_feature = image_feature.view(
                batch_size, -1
            )  # (batch_size, image_feature_dim)
            image_features.append(image_feature)
        image_features = torch.cat(
            image_features, dim=1
        )  # (batch_size, num_images * n_obs_steps * image_feature_dim)

        # Apply linear layers
        # FIXME: when state_dim is zero, all elements of state_feature are zero, so do not concatenate
        combined_feature = torch.cat(
            [state_feature, image_features], dim=1
        )  # (batch_size, combined_feature_dim)
        action_seq = self.linear_layer_seq(
            combined_feature
        )  # (batch_size, action_dim * n_action_steps)

        # Reshape action_seq
        action_seq = action_seq.reshape(
            batch_size, self.n_action_steps, -1
        )  # (batch_size, n_action_steps, action_dim)

        return action_seq
