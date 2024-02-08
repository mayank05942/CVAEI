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

    def loss_function(self, *inputs: Any, **kwargs) -> torch.Tensor:
        recons = inputs[0]
        input = inputs[1]
        mu = inputs[2]
        logvar = inputs[3]
        recon_loss = F.mse_loss(recons, input, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kwargs.get('beta', 1) * kld_loss
    
    def train_model(self, train_loader, optimizer, epochs=10, beta=0.1, device = None):
        if device is None: 
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(device)
        self.train()  # Set the model to training mode

        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (data, theta) in enumerate(train_loader):
                data, theta = data.to(device), theta.to(device)

                optimizer.zero_grad()  # Clear the gradients of all optimized tensors
                
                # Forward pass: Compute predicted output by passing inputs to the model
                theta_pred, mu, logvar =self(theta, data)
                
                # Compute the loss
                loss = self.loss_function(theta_pred, theta, mu, logvar, M_N=1.0/len(train_loader), beta=beta)
                
                # Backward pass: Compute gradient of the loss with respect to model parameters
                loss.backward()
                
                # Perform a single optimization step (parameter update)
                optimizer.step()
                
                total_loss += loss.item()

            # Print average loss for the epoch
            avg_loss = total_loss / len(train_loader.dataset)
            print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')



