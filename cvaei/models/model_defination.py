import torch
from torch import nn
from typing import List
from typing import Tuple
import torch.nn.functional as F
from typing import List, Callable


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


class MultiTaskDecoder(nn.Module):
    """
    Implements the Decoder for MultiTaskCVAE, mapping features to data for two separate tasks
    with output dimensions matching the dimensions of the input and the condition respectively.
    """

    def __init__(
        self,
        latent_dim: int,
        input_dim: int,  # Now explicitly taking input_dim for output_dim_1
        conditional_dim: int,  # Directly using conditional_dim for output_dim_2
        hidden_dims: List[int] = None,
        activation_fn: nn.Module = nn.ReLU(),
    ):
        super(MultiTaskDecoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256, 256]

        # Adjust latent_dim to account for concatenated conditional data
        adjusted_latent_dim = latent_dim + conditional_dim

        # Dynamically create layers based on hidden_dims
        modules = []
        input_dim_current = adjusted_latent_dim  # Current input dimension starts with adjusted latent dimension
        for h_dim in hidden_dims:
            modules.append(nn.Linear(input_dim_current, h_dim))
            modules.append(activation_fn)
            input_dim_current = h_dim  # Update input dimension for the next layer

        self.layers = nn.Sequential(*modules)

        # Task-specific output layers, directly using provided dimensions
        self.output_layer_1 = nn.Linear(hidden_dims[-1], input_dim)
        self.output_layer_2 = nn.Linear(hidden_dims[-1], conditional_dim)

    def forward(
        self, z: torch.Tensor, condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Concatenate latent variable z with conditional data
        z_cond = torch.cat([z, condition], dim=1)
        h = self.layers(z_cond)
        # Produce outputs for both tasks, matching the dimensions of the input and condition
        output_1 = self.output_layer_1(h)
        output_2 = self.output_layer_2(h)
        return output_1, output_2


class CNN_Decoder(nn.Module):
    """
    CNN_Decoder takes a latent variable (NxL) and a conditional input (NxSxT) to reconstruct
    both the original input (Nx input_dim) and the conditional data (NxSxT).
    """

    def __init__(
        self,
        latent_dim: int,
        conditional_dim: Tuple[int, int],  # tuple (S, T)
        output_dim_1: int,  # input dim
        conv_output_channels: List[int],
        kernel_sizes: List[int],
        activation_fn: nn.Module = nn.ReLU(),
    ):
        super(CNN_Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.conditional_channels, self.sequence_length = conditional_dim
        self.conditional_dim = conditional_dim
        self.output_dim_1 = output_dim_1
        self.output_channels = self.conditional_channels

        self.activation_fn = activation_fn

        # The latent variable will be concatenated with the conditional input, thus needing adjustment
        self.fc_expand = nn.Linear(latent_dim, self.sequence_length)

        # Initialize Sequential container for Conv1D, BatchNorm1d, and Activation layers
        modules = []
        in_channels = (
            self.conditional_channels + 1
        )  # Adjust for concatenated latent variable
        for out_channels, kernel_size in zip(conv_output_channels, kernel_sizes):
            modules.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=1,
                )
            )
            modules.append(nn.BatchNorm1d(out_channels))
            modules.append(activation_fn)
            in_channels = out_channels

        self.convolutional_sequence = nn.Sequential(
            *modules
        )  # Bundle the layers together

        # Calculating the flattened size for the linear layer
        flattened_size = conv_output_channels[-1] * self.sequence_length

        self.output_layer_1 = nn.Linear(flattened_size, output_dim_1)
        self.output_layer_2 = nn.Conv1d(
            conv_output_channels[-1], self.output_channels, kernel_size=1, stride=1
        )

    def forward(self, z: torch.Tensor, condition: torch.Tensor):
        # Expand and reshape latent variable for concatenation with the condition
        z_expanded = self.fc_expand(z).view(-1, 1, self.sequence_length)
        combined_input = torch.cat(
            (z_expanded, condition), dim=1
        )  # Concatenating along the channel dimension to get Nx4x200

        # Process the combined input through Conv1D layers
        x = self.convolutional_sequence(combined_input)

        x_flattened = x.view(x.size(0), -1)
        reconstructed_input = self.output_layer_1(x_flattened)

        # Pass through the final Conv1d layer to match the conditional data dimension
        reconstructed_condition = self.output_layer_2(x)

        return reconstructed_input, reconstructed_condition
