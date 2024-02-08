import torch
from torch import nn, optim
from torch.nn import functional as F
from typing import List, Any, Tuple
import numpy as np
from .model_base import ModelBase

class Encoder(nn.Module):
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
    def __init__(self, input_dim, latent_dim, output_dim, conditional_dim, 
                 encoder_hidden_dims: List[int], decoder_hidden_dims: List[int], 
                 activation_fn: nn.Module = nn.ReLU()):
        
        super(CVAE, self).__init__()

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
        #misfit_loss = F.mse_loss(y_hat, y, reduction='sum')
        misfit_loss = recon_loss
        
        # KL divergence loss
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + misfit_loss + beta * kl_div
        
        return total_loss, recon_loss, misfit_loss, kl_div, beta


    def train_model(self, train_loader, optimizer, epochs=10, cycle_length=10, num_cycles=1, device=None):

        if device is None: 
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(device)
        self.train()  # Set the model to training mode

        for epoch in range(epochs):
            total_loss = 0
            total_recon_loss = 0
            total_misfit_loss = 0
            total_kl_div = 0
            epoch_beta = 0

            epoch_beta = self.calculate_beta(epoch, epochs, cycle_length, num_cycles)

            # Calculate beta for this epoch
            for batch_idx, (data, theta) in enumerate(train_loader):
                data, theta = data.to(device), theta.to(device)

                optimizer.zero_grad()  # Clear the gradients of all optimized tensors
                
                # Forward pass
                theta_pred, mu, logvar = self(theta, data)
                
                # Compute the loss
                loss, recon_loss, misfit_loss, kl_div, beta = self.loss_function(theta_pred, theta, data, data , mu, logvar, 
                                                    beta= epoch_beta)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_misfit_loss += misfit_loss.item()
                total_kl_div += kl_div.item()

            # Print average loss for the epoch
            avg_loss = total_loss / len(train_loader.dataset)
            avg_recon_loss = total_recon_loss / len(train_loader.dataset)
            avg_misfit_loss = total_misfit_loss / len(train_loader.dataset)
            avg_kl_div = total_kl_div / len(train_loader.dataset)
            print(f'Epoch [{epoch+1}/{epochs}], Beta: {epoch_beta:.4f}, Average Loss: {avg_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, Misfit Loss: {avg_misfit_loss:.4f}, KL Div: {avg_kl_div:.4f}')
            

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
            return cycle_epoch / (cycle_length / num_cycles)  # Linearly increase
        return 1  # Remain at 1 for the rest of the cycle