import torch
from torch import nn, optim
from torch.nn import functional as F
from typing import List, Any, Tuple
import numpy as np
from .model_base import ModelBase
import matplotlib.pyplot as plt



class Encoder(nn.Module):
    """
    Implements the Encoder, learning expressive features
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int] = None,
                 activation_fn: nn.Module = nn.ReLU()):
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
    def __init__(self, latent_dim: int, output_dim: int, conditional_dim: int,
                 hidden_dims: List[int] = None, activation_fn: nn.Module = nn.ReLU()):
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


class CVAE(ModelBase):
    """
    Defines the Conditional Variational Autoencoder.
    Implements the definition, training and prediction pipeline.
    """
    def __init__(self, input_dim, latent_dim, output_dim, conditional_dim,
                 encoder_hidden_dims: List[int], decoder_hidden_dims: List[int],
                 activation_fn: nn.Module = nn.ReLU()):

        super(CVAE, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.conditional_dim = conditional_dim

        self.losses = {'total_loss': [], 'recon_loss': [], 'misfit_loss': [], 'kl_div': []}


        self.encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim,
                               hidden_dims=encoder_hidden_dims, activation_fn=activation_fn)
        self.decoder = Decoder(latent_dim=latent_dim, output_dim=output_dim,
                               conditional_dim=conditional_dim, hidden_dims=decoder_hidden_dims,
                               activation_fn=activation_fn)

    def encode(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(input)

    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, condition)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input_theta: torch.Tensor, conditional_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(input_theta)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, conditional_data), mu, logvar

    def loss_function(self, x, x_hat, y, y_hat, mean, logvar, beta):
        # Reconstruction loss compares the input x to its reconstruction x_hat
        recon_loss = F.mse_loss(x_hat, x, reduction='sum')

        # Misfit loss compares the actual y to the predicted y_hat
        misfit_loss = F.mse_loss(y_hat, y, reduction='sum')

        # KL divergence loss
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        # Total loss
        total_loss = recon_loss + misfit_loss + beta * kl_div

        return total_loss, recon_loss, misfit_loss, kl_div, beta

    def train_model(self, train_loader, optimizer, epochs=10,
                    cycle_length=10, num_cycles=1, device=None,
                    theta_normalizer=None, data_normalizer=None, forward_model=None):

        if device.type.startswith('cuda'):
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        else:
            print("Using CPU")

        self.to(device)
        self.train()  # Set the model to training mode

        if theta_normalizer is None or data_normalizer is None or forward_model is None:
            raise ValueError(
                "theta_normalizer, data_normalizer, and forward_model must be provided")

        for epoch in range(epochs):

            epoch_losses = {'total_loss': 0, 'recon_loss': 0, 'misfit_loss': 0, 'kl_div': 0}

            epoch_beta = self.calculate_beta(
                epoch, epochs, cycle_length, num_cycles)

            for batch_idx, (data, theta) in enumerate(train_loader):
                data, theta = data.to(device), theta.to(device)

                optimizer.zero_grad()

                # Forward pass
                theta_pred, mu, logvar = self(theta, data)

                # de-normalize theta predicted
                theta_pred_unnorm = theta_normalizer.inverse_transform(
                    theta_pred)

                # Pass through the model

                y_pred_unnorm = torch.stack(
                    [forward_model(t) for t in theta_pred_unnorm])
                
                # transform to get in the scale of data
                y_pred_norm = data_normalizer.transform(
                    y_pred_unnorm).to(device)
                
                #y_pred_norm = data

                # Compute the loss
                loss, recon_loss, misfit_loss, kl_div, beta = self.loss_function(theta_pred, theta, y_pred_norm, data,
                                                                                 mu, logvar, beta=epoch_beta)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_losses['total_loss'] += loss.item()
                epoch_losses['recon_loss'] += recon_loss.item()
                epoch_losses['misfit_loss'] += misfit_loss.item()
                epoch_losses['kl_div'] += kl_div.item()

            # Log epoch-wise average loss
            for key in epoch_losses:
                self.losses[key].append(epoch_losses[key] / len(train_loader.dataset))
            
            print(f'Epoch {epoch+1}/{epochs}, Beta: {epoch_beta:.1f}, Total Loss: {epoch_losses["total_loss"]:.4f}, ' +
                  f'Recon Loss: {epoch_losses["recon_loss"]:.4f}, Misfit Loss: {epoch_losses["misfit_loss"]:.4f}, ' +
                  f'KL Div: {epoch_losses["kl_div"]:.4f}')

 
    @staticmethod
    def calculate_beta(epoch, total_epochs, cycle_length, num_cycles):
        """
        Calculate the beta value for KL divergence regularization based on the current epoch.

        Parameters:
        epoch (int): Current training epoch.
        total_epochs (int): Total number of epochs for training.
        cycle_length (int): Length of a beta cycle.
        num_cycles (int): Number of cycles in the total epochs.

        Returns:
        float: Calculated beta value.
        """
        cycle_epoch = epoch % cycle_length
        if cycle_epoch < (cycle_length / num_cycles):
            # Linearly increase
            return cycle_epoch / (cycle_length / num_cycles)
        return 1  # Remain at 1 for the rest of the cycle

    def get_posterior(self, observed_data, num_samples=10000, latent_dim=None, device=None):
        """
        Get samples from the posterior distribution given observed data.

        Parameters:
        observed_data (torch.Tensor): Observed data to condition the generation on.
        num_samples (int): Number of samples to generate from the posterior.
        latent_dim (int, optional): Dimension of the latent space. Defaults to None, in which case it will be inferred from the model.
        device (str, optional): Device to run the computations on. Defaults to None, in which case the current device is used.

        Returns:
        torch.Tensor: Samples from the posterior distribution.
        """
        if device is None:
            device = next(self.parameters()).device

        if latent_dim is None:
            latent_dim = self.latent_dim

        # Set the network to evaluation mode
        self.eval()

        with torch.no_grad():
            observed_data = observed_data.to(device)

            mean = torch.zeros(latent_dim).to(device)
            covariance = torch.eye(latent_dim).to(device)
            m = torch.distributions.MultivariateNormal(
                mean, covariance_matrix=covariance)

            z = m.sample((num_samples,)).to(device)

            y = observed_data.repeat(num_samples, 1)

            # zy = torch.cat((z, y), dim=1)
            posterior_samples = self.decoder(z, y)

            # posterior_samples = theta_scaler.inverse_transform(posterior_samples)

        return posterior_samples
    
    def plot_loss(self):
        """Plot the training losses in separate plots for detailed analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # Create a 2x2 grid of plots
        axes = axes.flatten()  # Flatten the grid to make it easier to iterate
        loss_titles = ['Total Loss', 'Reconstruction Loss', 'Misfit Loss', 'KL Divergence']
        
        for i, key in enumerate(['total_loss', 'recon_loss', 'misfit_loss', 'kl_div']):
            axes[i].plot(self.losses[key], label=key, linewidth=2)
            axes[i].set_title(loss_titles[i])
            axes[i].set_xlabel('Epochs')
            axes[i].set_ylabel('Loss')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.show()
