from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim


from .model_base import ModelBase
from .model_defination import Decoder, Encoder

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, rand


class CVAE(ModelBase):
    """
    Defines the Conditional Variational Autoencoder.
    Implements the definition, training and prediction pipeline.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        output_dim: int,
        conditional_dim: int,
        encoder_hidden_dims: List[int],
        decoder_hidden_dims: List[int],
        activation_fn: nn.Module = nn.ReLU(),
        **kwargs,  # Accept additional keyword arguments
    ):
        super(CVAE, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.conditional_dim = conditional_dim
        self.w_recon = kwargs.get("w_recon", 1.0)
        self.w_misfit = kwargs.get("w_misfit", 1.0)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.training_losses = {
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

        self.encoder = Encoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            activation_fn=activation_fn,
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            output_dim=output_dim,
            conditional_dim=conditional_dim,
            hidden_dims=decoder_hidden_dims,
            activation_fn=activation_fn,
        )

    def encode(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(input)

    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, condition)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, input_theta: torch.Tensor, conditional_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(input_theta)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, conditional_data), mu, logvar

    def loss_function(self, x, x_hat, y, y_hat, mean, logvar, beta):
        # Reconstruction loss compares the input x to its reconstruction x_hat
        recon_loss = F.mse_loss(x_hat, x, reduction="sum") * self.w_recon
        # Misfit loss compares the actual y to the predicted y_hat
        misfit_loss = F.mse_loss(y_hat, y, reduction="sum") * self.w_misfit
        # KL divergence loss
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) * beta
        # Total loss
        total_loss = recon_loss + misfit_loss + kl_div
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
        theta_pred, mu, logvar = self(theta, data)  # Forward pass through the model
        theta_pred_unnorm = theta_normalizer.inverse_transform(theta_pred)
        y_pred_unnorm = forward_model(theta_pred_unnorm)
        y_pred_norm = data_normalizer.transform(y_pred_unnorm).to(device)

        if y_pred_norm.dim() == 3:
            y_pred_norm = y_pred_norm.squeeze(1)

        # Compute the loss
        total_loss, recon_loss, misfit_loss, kl_div, beta = self.loss_function(
            theta, theta_pred, data, y_pred_norm, mu, logvar, epoch_beta
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
        cycle_length,
        num_cycles,
        theta_normalizer,
        data_normalizer,
        forward_model,
    ):
        self.train()
        epoch_losses = {"total_loss": 0, "recon_loss": 0, "misfit_loss": 0, "kl_div": 0}
        epoch_beta = self.calculate_beta(epoch, epochs, cycle_length, num_cycles)

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
            optimizer.step()

            epoch_losses["total_loss"] += loss.item()
            epoch_losses["recon_loss"] += recon_loss.item()
            epoch_losses["misfit_loss"] += misfit_loss.item()
            epoch_losses["kl_div"] += kl_div.item()

        self.update_losses(epoch_losses, train_loader, self.training_losses)
        epoch_msg = f"Epoch {epoch+1}/{epochs}: Beta: {epoch_beta:.1f}, "
        losses_msg = ", ".join(
            f"{k}: {v[-1]:.4f}" for k, v in self.training_losses.items()
        )
        print(epoch_msg + losses_msg)

    def validate_epoch(
        self,
        validation_loader,
        device,
        epoch,
        epochs,
        cycle_length,
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
        epoch_beta = self.calculate_beta(epoch, epochs, cycle_length, num_cycles)

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

        epoch_msg = f"Epoch {epoch+1}/{epochs} Validation: Beta: {epoch_beta:.1f}, "
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
        cycle_length=10,
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
                cycle_length,
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
                cycle_length,
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
                    f"Epoch {epoch+1}: Validation loss improved to {best_val_loss:.4f}"
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

        if observed_data.dim() == 3:
            # Reshape y_pred to match y, case when [N, 1, 1000]
            observed_data = observed_data.squeeze(1)

        # Set the network to evaluation mode
        self.eval()

        with torch.no_grad():
            observed_data = observed_data.to(device)

            mean = torch.zeros(latent_dim).to(device)
            covariance = torch.eye(latent_dim).to(device)
            m = torch.distributions.MultivariateNormal(
                mean, covariance_matrix=covariance
            )

            z = m.sample((num_samples,)).to(device)

            y = observed_data.repeat(num_samples, 1)

            # zy = torch.cat((z, y), dim=1)
            posterior_samples = self.decoder(z, y)

            # posterior_samples = theta_scaler.inverse_transform(posterior_samples)

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

    def tune_model(
        self,
        train_loader,
        validation_loader,
        theta_normalizer,
        data_normalizer,
        forward_model,
        n_trials=10,
    ):
        """
        Optimize the hyperparameters of the CVAE model using Hyperopt within the CVAE class.

        Parameters:
        - train_loader: DataLoader for training data.
        - validation_loader: DataLoader for validation data.
        - n_trials: The number of optimization trials to run.
        """

        def objective(hyperparams):
            # Hyperparameter to be tuned
            latent_dim_trial = int(hyperparams["latent_dim"])

            # Move model to the correct device at the beginning of each trial
            self.to(self.device)

            # Temporarily adjust the model's hyperparameters
            original_latent_dim = self.latent_dim
            self.latent_dim = latent_dim_trial

            # Use a fixed learning rate
            fixed_lr = 1e-3  # Example fixed learning rate, adjust as needed
            optimizer = optim.AdamW(self.parameters(), lr=fixed_lr)

            # Training and validation process
            best_val_loss = float("inf")
            for epoch in range(10):  # Assuming a fixed number of epochs for simplicity
                self.train_epoch(
                    train_loader=train_loader,
                    optimizer=optimizer,
                    device=self.device,
                    epoch=epoch,
                    epochs=10,
                    cycle_length=10,
                    num_cycles=1,
                    theta_normalizer=theta_normalizer,
                    data_normalizer=data_normalizer,
                    forward_model=forward_model,
                )
                avg_val_loss = self.validate_epoch(
                    validation_loader=validation_loader,
                    device=self.device,
                    epoch=epoch,
                    epochs=10,
                    cycle_length=10,
                    num_cycles=1,
                    theta_normalizer=theta_normalizer,
                    data_normalizer=data_normalizer,
                    forward_model=forward_model,
                )
                best_val_loss = min(best_val_loss, avg_val_loss)

            # Restore original latent dimension before the trial ends
            self.latent_dim = original_latent_dim

            # Return the loss
            return {"loss": best_val_loss, "status": STATUS_OK}

        # Define the search space for latent_dim only
        search_space = {
            "latent_dim": hp.quniform(
                "latent_dim", 10, 20, 1
            )  # Example range from 10 to 20
        }

        # Run the optimization
        trials = Trials()
        best = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=n_trials,
            trials=trials,
        )

        best_latent_dim = int(best["latent_dim"])  # Convert to integer
        print(f"Optimization finished. Best latent_dim: {best_latent_dim}")
