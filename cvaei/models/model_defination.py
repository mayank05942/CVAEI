import torch
from torch import nn
from typing import List


class Encoder(nn.Module):
    """
    Implements the Encoder, learning expressive features
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = None,
        activation_fn: nn.Module = nn.ReLU(),
    ):
        super(Encoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256, 256]

        # Dynamically create layers based on hidden_dims
        modules = []
        for h_dim in hidden_dims:
            modules.append(nn.Linear(input_dim, h_dim))
            modules.append(activation_fn)
            input_dim = h_dim  # Set input dimension for the next layer

        self.layers = nn.Sequential(*modules)

        # Output layers for mean and log variance
        self.fc_mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        h = self.layers(x)
        return self.fc_mean(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """
    Implements the Decoder, mapping features to data
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        conditional_dim: int,
        hidden_dims: List[int] = None,
        activation_fn: nn.Module = nn.ReLU(),
    ):
        super(Decoder, self).__init__()
        self.conditional_dim = conditional_dim

        # Adjust latent_dim to account for concatenated conditional data
        adjusted_latent_dim = latent_dim + conditional_dim

        if hidden_dims is None:
            hidden_dims = [256, 256, 256]

        # Dynamically create layers based on hidden_dims
        modules = []
        # Initialize the input dimension to the adjusted latent dimension
        input_dim = adjusted_latent_dim
        for h_dim in hidden_dims:
            modules.append(nn.Linear(input_dim, h_dim))
            modules.append(activation_fn)
            input_dim = h_dim  # Update input dimension for the next layer

        self.layers = nn.Sequential(*modules)
        self.final_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # Concatenate latent variable z with conditional data
        z_cond = torch.cat([z, condition], dim=1)
        h = self.layers(z_cond)
        return self.final_layer(h)
