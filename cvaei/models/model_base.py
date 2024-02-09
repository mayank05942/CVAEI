import torch
from torch import nn
from typing import List, Any, Tuple


class ModelBase(nn.Module):
    """
    Base class for Conditional Variational Autoencoder (CVAE).

    This class provides the structure for the CVAE and includes placeholders
    for the main methods that need to be implemented by a subclass.
    """

    def __init__(self):
        super(ModelBase, self).__init__()

    def encode(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input parameters into a latent representation by outputting mean and log variance.

        Parameters:
            input (torch.Tensor): The input data tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the mean and log variance of the latent distribution.
        """
        raise NotImplementedError(
            "The encode method must be implemented by the subclass.")

    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent variable z along with conditional data to reconstruct the input.

        Parameters:
            z (torch.Tensor): The latent variable.
            condition (torch.Tensor): Conditional data that influences the decoding process.

        Returns:
            torch.Tensor: The output parameters theta for the distribution of the decoded data.
        """
        raise NotImplementedError(
            "The decode method must be implemented by the subclass.")

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        Must be implemented by subclasses if they use stochastic latent variables.
        """
        raise NotImplementedError(
            "The reparameterize method must be implemented by the subclass.")

    def forward(self, *inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        Must be implemented by subclasses to define the computation performed at every call.
        Should return the reconstructed input, the mean and log variance of the latent distribution.
        """
        raise NotImplementedError(
            "The forward method must be implemented by the subclass.")

    def loss_function(self, *inputs: Any, **kwargs) -> torch.Tensor:
        """
        Computes the VAE loss function.
        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "The loss_function method must be implemented by the subclass.")

    def inference(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Generates new data conditioned on input x.
        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "The generate method must be implemented by the subclass.")

    # @abstractmethod
    def train_model(self, *inputs: Any, **kwargs) -> torch.Tensor:
        """
        Sub-classable method for training the cave model. Each derived class must implement.
        """
        raise NotImplementedError(
            "The generate method must be implemented by the subclass.")
