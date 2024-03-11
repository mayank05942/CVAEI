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
            output_channels=conditional_dim,
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

    # def loss_function(self, x, x_hat, y, y_hat, mean, logvar, beta):
    #     # Reconstruction loss compares the input x to its reconstruction x_hat
    #     recon_loss = F.mse_loss(x_hat, x, reduction="sum") * self.w_recon
    #     # Misfit loss compares the actual y to the predicted y_hat
    #     misfit_loss = F.mse_loss(y_hat, y, reduction="sum") * self.w_misfit
    #     # KL divergence loss
    #     kl_div = (
    #         -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) * beta * self.kld
    #     )
    #     # Total loss
    #     total_loss = recon_loss + misfit_loss + kl_div
    #     return total_loss, recon_loss, misfit_loss, kl_div, beta

    def loss_function(self, x, x_hat, y, y_hat, mean, logvar, beta):
        # Reconstruction loss compares the input x to its reconstruction x_hat
        recon_loss = F.mse_loss(x_hat, x, reduction="sum") * self.w_recon
        # Misfit loss compares the actual y to the predicted y_hat
        misfit_loss = F.mse_loss(y_hat, y, reduction="sum") * self.w_misfit
        # KL divergence loss
        kl_div = (
            -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) * beta * self.kld
        )

        # Use geometric mean for the total loss
        epsilon = 1e-8  # To ensure numerical stability and avoid log(0)
        total_loss = (
            torch.pow(recon_loss + epsilon, 1 / 3)
            * torch.pow(misfit_loss + epsilon, 1 / 3)
            * torch.pow(torch.abs(kl_div) + epsilon, 1 / 3)
        )

        return total_loss, recon_loss, misfit_loss, kl_div, beta

    def process_batch(
        self,
        data,
        theta,
        device,
        epoch_beta,
        theta_normalizer,
        data_normalizer,
        forward_model,
        validation=True,
    ):
        data, theta = data.to(device), theta.to(device)
        theta_pred, data_pred, mu, logvar = self(
            theta, data
        )  # Forward pass through the model

        # Compute the loss
        total_loss, recon_loss, misfit_loss, kl_div, beta = self.loss_function(
            theta, theta_pred, data, data_pred, mu, logvar, epoch_beta
        )

        return total_loss, recon_loss, misfit_loss, kl_div, beta

    def update_losses(self, epoch_losses, loader, loss_dict):
        for key in epoch_losses:
            epoch_losses[key] /= len(loader.dataset)
            loss_dict[key].append(epoch_losses[key])

    def train_epoch(
        self,
        train_loader,
        optimizer,
        device,
        epoch,
        epochs,
        num_cycles,
        theta_normalizer,
        data_normalizer,
        forward_model,
        max_grad_norm=1.0,
    ):
        self.train()
        epoch_losses = {
            "total_loss": 0,
            "recon_loss": 0,
            "misfit_loss": 0,
            "kl_div": 0,
        }
        epoch_beta = self.calculate_beta(
            epoch, epochs, num_cycles, ratio=0.5, start=0, stop=1
        )
        # epoch_beta = self.calculate_beta(epoch, epochs, cycle_length, num_cycles)
        self.training_losses["beta"].append(epoch_beta)

        for data, theta in train_loader:
            optimizer.zero_grad()
            loss, recon_loss, misfit_loss, kl_div, beta = self.process_batch(
                data,
                theta,
                device,
                epoch_beta,
                theta_normalizer,
                data_normalizer,
                forward_model,
                validation=False,
            )
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

            optimizer.step()

            # epoch_losses["beta"] += epoch_beta
            epoch_losses["total_loss"] += loss.item()
            epoch_losses["recon_loss"] += recon_loss.item()
            epoch_losses["misfit_loss"] += misfit_loss.item()
            epoch_losses["kl_div"] += kl_div.item()

        self.update_losses(epoch_losses, train_loader, self.training_losses)

        epoch_msg = f"Epoch {epoch+1}/{epochs}:"
        # Exclude beta from the losses message, format it separately
        losses_msg = ", ".join(
            f"{k}: {v[-1]:.4f}" for k, v in self.training_losses.items() if k != "beta"
        )
        beta_msg = f"Beta: {self.training_losses['beta'][-1]:.1f}"
        print(epoch_msg + " " + beta_msg + ", " + losses_msg)

    def validate_epoch(
        self,
        validation_loader,
        device,
        epoch,
        epochs,
        num_cycles,
        theta_normalizer,
        data_normalizer,
        forward_model,
    ):
        self.eval()
        val_epoch_losses = {
            "total_loss": 0,
            "recon_loss": 0,
            "misfit_loss": 0,
            "kl_div": 0,
        }
        # epoch_beta = self.calculate_beta(epoch, epochs, cycle_length, num_cycles)
        epoch_beta = self.calculate_beta(
            epoch, epochs, num_cycles, ratio=0.5, start=0, stop=1
        )

        with torch.no_grad():
            for data, theta in validation_loader:
                loss, recon_loss, misfit_loss, kl_div, beta = self.process_batch(
                    data,
                    theta,
                    device,
                    epoch_beta,
                    theta_normalizer,
                    data_normalizer,
                    forward_model,
                    validation=True,
                )

                val_epoch_losses["total_loss"] += loss.item()
                val_epoch_losses["recon_loss"] += recon_loss.item()
                val_epoch_losses["misfit_loss"] += misfit_loss.item()
                val_epoch_losses["kl_div"] += kl_div.item()

        self.update_losses(val_epoch_losses, validation_loader, self.validation_losses)

        epoch_msg = f"Epoch {epoch+1}/{epochs} Validation: "
        losses_msg = ", ".join(f"{k}: {v:.4f}" for k, v in val_epoch_losses.items())
        print(epoch_msg + losses_msg)
        print()

        return val_epoch_losses["total_loss"] / len(validation_loader.dataset)

    def train_model(
        self,
        train_loader,
        validation_loader,
        optimizer,
        epochs=10,
        num_cycles=1,
        theta_normalizer=None,
        data_normalizer=None,
        forward_model=None,
        patience=5,
    ):

        self.to(self.device)

        print(
            f"Using {'GPU: ' + torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else 'CPU'} for training."
        )

        best_val_loss = float("inf")
        no_improve_epoch = 0

        for epoch in range(epochs):
            # Train for one epoch
            self.train_epoch(
                train_loader,
                optimizer,
                self.device,
                epoch,
                epochs,
                num_cycles,
                theta_normalizer,
                data_normalizer,
                forward_model,
            )

            # Validate the performance on validation dataset
            avg_val_loss = self.validate_epoch(
                validation_loader,
                self.device,
                epoch,
                epochs,
                num_cycles,
                theta_normalizer,
                data_normalizer,
                forward_model,
            )

            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve_epoch = 0
                print(
                    f"Epoch {epoch+1}: Validation loss improved to {best_val_loss:.8f}"
                )
            else:
                no_improve_epoch += 1
                print(
                    f"Epoch {epoch+1}: No improvement in validation loss for {no_improve_epoch} epochs."
                )

            # Early Stopping Check
            if no_improve_epoch >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch+1} with validation loss: {best_val_loss:.4f}"
                )
                break

        print("Training completed.")

    @staticmethod
    def calculate_beta(epoch, total_epochs, n_cycle, ratio=0.5, start=0, stop=1):
        """
        Calculate the cyclical beta value for KL divergence regularization based on the current epoch.

        Parameters:
        epoch (int): Current training epoch.
        total_epochs (int): Total number of epochs for training.
        n_cycle (int): Number of cycles over total epochs.
        ratio (float): Portion of the cycle for beta to increase.
        start (float): Starting value of beta.
        stop (float): Stopping value of beta.

        Returns:
        float: Calculated beta value.
        """
        # Calculate the period of each cycle and the step for linear increase
        period = total_epochs / n_cycle
        step = (stop - start) / (period * ratio)

        # Determine the current cycle
        cycle = int(epoch // period)
        cycle_epoch = epoch % period

        # Increase beta linearly for a portion of the cycle, then hold
        if cycle_epoch < (period * ratio):
            beta = start + cycle_epoch * step
        else:
            beta = stop

        return beta

    # def get_posterior(
    #     self, observed_data, num_samples=10000, latent_dim=None, device=None
    # ):
    #     """
    #     Get samples from the posterior distribution given observed data.

    #     Parameters:
    #     observed_data (torch.Tensor): Observed data to condition the generation on.
    #     num_samples (int): Number of samples to generate from the posterior.
    #     latent_dim (int, optional): Dimension of the latent space. Defaults to None, in which case it will be inferred from the model.
    #     device (str, optional): Device to run the computations on. Defaults to None, in which case the current device is used.

    #     Returns:
    #     torch.Tensor: Samples from the posterior distribution.
    #     """
    #     if device is None:
    #         device = next(self.parameters()).device

    #     if latent_dim is None:
    #         latent_dim = self.latent_dim

    #     if observed_data.dim() == 3:
    #         # Reshape y_pred to match y, case when [N, 1, 1000]
    #         observed_data = observed_data.squeeze(1)

    #     # Set the network to evaluation mode
    #     self.eval()

    #     with torch.no_grad():
    #         observed_data = observed_data.to(device)

    #         mean = torch.zeros(latent_dim).to(device)
    #         covariance = torch.eye(latent_dim).to(device)
    #         m = torch.distributions.MultivariateNormal(
    #             mean, covariance_matrix=covariance
    #         )

    #         z = m.sample((num_samples,)).to(device)

    #         y = observed_data.repeat(num_samples, 1)

    #         # zy = torch.cat((z, y), dim=1)
    #         posterior_samples, _ = self.decoder(z, y)

    #         # posterior_samples = theta_scaler.inverse_transform(posterior_samples)

    #     return posterior_samples

    def get_posterior(
        self, observed_data, num_samples=10000, latent_dim=None, device=None
    ):
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

        # Ensure the network is in evaluation mode
        self.eval()

        with torch.no_grad():
            observed_data = observed_data.to(device)

            # Ensure observed_data has the shape expected by the model
            # If observed_data shape is (1, 3, 200) and you need to generate samples based on it:
            if observed_data.dim() == 3 and observed_data.shape[0] == 1:
                # Repeat the observed data to match num_samples
                observed_data = observed_data.repeat(
                    num_samples, 1, 1
                )  # Shape becomes (num_samples, 3, 200)

            mean = torch.zeros(latent_dim).to(device)
            covariance = torch.eye(latent_dim).to(device)
            m = torch.distributions.MultivariateNormal(
                mean, covariance_matrix=covariance
            )

            z = m.sample((num_samples,)).to(device)

            # Decode z and the repeated observed_data to generate posterior samples
            posterior_samples, _ = self.decode(
                z, observed_data
            )  # Adjust according to your decode method's signature

        return posterior_samples

    def plot_loss(self):
        """Plot the training and validation losses in separate plots for detailed analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # Create a 2x2 grid of plots
        axes = axes.flatten()  # Flatten the grid to make it easier to iterate
        loss_titles = [
            "Total Loss",
            "Reconstruction Loss",
            "Misfit Loss",
            "KL Divergence",
        ]

        for i, key in enumerate(["total_loss", "recon_loss", "misfit_loss", "kl_div"]):
            axes[i].plot(
                self.training_losses[key],
                label=f"Training {key}",
                linewidth=2,
                marker="o",
                markersize=4,
            )
            axes[i].plot(
                self.validation_losses[key],
                label=f"Validation {key}",
                linewidth=2,
                linestyle="--",
                marker="x",
                markersize=4,
            )
            axes[i].set_title(loss_titles[i])
            axes[i].set_xlabel("Epochs")
            axes[i].set_ylabel("Loss")
            axes[i].legend()
            axes[i].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_beta(self):
        """Plot the beta value across training epochs."""

        plt.figure(figsize=(12, 6))
        beta_values = self.training_losses["beta"]
        plt.plot(beta_values, label="Cyclical Beta", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Beta Value")
        plt.title("Cyclical Beta Schedule Across Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()
