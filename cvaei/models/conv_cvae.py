from .model_base import ModelBase
from .model_defination import CNN_Decoder, Encoder
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim


class CNN_CVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        conditional_dim,
        encoder_hidden_dims: List[int],
        sequence_length: int,  # Added parameter for sequence length
        conv_output_channels: List[int],  # Added parameter for Conv1D output channels
        kernel_sizes: List[int],  # Added parameter for Conv1D kernel sizes
        activation_fn: nn.Module = nn.ReLU(),
        device=None,
        w_recon=1.0,
        w_misfit=1.0,
        kld=1.0,
        **kwargs,
    ):
        super(CNN_CVAE, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.conditional_dim = conditional_dim
        self.w_recon = w_recon
        self.w_misfit = w_misfit
        self.kld = kld

        self.training_losses = {
            "beta": [],
            "total_loss": [],
            "recon_loss": [],
            "misfit_loss": [],
            "kl_div": [],
        }
        self.validation_losses = {
            "total_loss": [],
            "recon_loss": [],
            "misfit_loss": [],
            "kl_div": [],
        }

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the encoder
        self.encoder = Encoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            activation_fn=activation_fn,
        )

        # Initialize the decoder
        self.decoder = CNN_Decoder(
            latent_dim=latent_dim,
            conditional_dim=conditional_dim,
            output_dim_1=input_dim,
            sequence_length=sequence_length,
            conv_output_channels=conv_output_channels,
            kernel_sizes=kernel_sizes,
            output_channels=conditional_dim,  # Assuming output_channels is meant to match conditional_dim
        )

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        # Directly return the two outputs from the MultiTaskDecoder
        recon_x1, recon_x2 = self.decoder(z, cond)
        return recon_x1, recon_x2

    def forward(self, x, cond):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x1, recon_x2 = self.decode(z, cond)
        return recon_x1, recon_x2, mu, logvar

    def loss_function(self, x, x_hat, y, y_hat, mean, logvar, beta):
        # Reconstruction loss compares the input x to its reconstruction x_hat
        recon_loss = F.mse_loss(x_hat, x, reduction="sum") * self.w_recon
        # Misfit loss compares the actual y to the predicted y_hat
        misfit_loss = F.mse_loss(y_hat, y, reduction="sum") * self.w_misfit
        # KL divergence loss
        kl_div = (
            -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) * beta * self.kld
        )
        # Total loss
        total_loss = recon_loss + misfit_loss + kl_div
        return total_loss, recon_loss, misfit_loss, kl_div, beta
