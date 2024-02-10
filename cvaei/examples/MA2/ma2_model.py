import torch
import numpy as np
from cvaei.helper import DataNormalizer
import matplotlib.pyplot as plt


class MovingAverage2:
    def __init__(self, true_params=None):
        """
        Initialize the MA2 model with optional true parameters.

        Parameters:
        - true_params (torch.Tensor, optional): The true parameters for the MA2 model.
        """
        if true_params is None:
            self.true_params = torch.tensor([0.6, 0.2])
        else:
            self.true_params = true_params

        self.theta_normalizer = None
        self.data_normalizer = None


    def simulator(self, param, seed=42):
        """
        Simulate data using the MA2 model.

        Parameters:
        - param (torch.Tensor): The parameters for the MA2 model.
        - seed (int): Seed for random number generation to ensure reproducibility.

        Returns:
        - torch.Tensor: Simulated data based on the MA2 model.
        """
        torch.manual_seed(seed)
        n = 100
        m = len(param)
        g = torch.randn(n)
        gy = torch.randn(n) * 0.3
        y = torch.zeros(n)
        x = torch.zeros(n)
        for t in range(n):
            x[t] += g[t]
            for p in range(min(t, m)):
                x[t] += g[t - 1 - p] * param[p]
            y[t] = x[t] + gy[t]
        return y

    @staticmethod
    def prior(num_samples):
        """
        Sample parameters from the prior distribution using PyTorch tensors.

        Parameters:
        - num_samples (int): Number of samples to draw.

        Returns:
        - torch.Tensor: Sampled parameters from the prior distribution.
        """
        p = []
        acc = 0
        while acc < num_samples:
            r = torch.rand(2) * torch.tensor([4, 2]) + torch.tensor([-2, -1])
            if r[1] + r[0] >= -1 and r[1] - r[0] >= -1:
                p.append(r)
                acc += 1
        return torch.stack(p)

    def generate_data(self, num_samples=1000):
        """
        Generate data samples based on the prior and simulator.

        Parameters:
        - num_samples (int): Number of samples to generate.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Sampled parameters and corresponding simulated data.
        """
        theta = self.prior(num_samples=num_samples)
        data = torch.stack([self.simulator(t) for t in theta])
        return theta, data

    def prepare_data(self, num_samples=1000, scale=True):
        """
        Generate, (optionally) normalize data and parameters, and return them with their normalizers.

        Parameters:
        - num_samples (int): Number of samples to generate and (optionally) normalize.
        - scale (bool): If True, return normalized data; otherwise, return unnormalized data.

        Returns:
        - Tuple containing (optionally normalized) theta, (optionally normalized) data, 
        theta normalizer, and data normalizer.
        """
        # Generate data
        theta, data = self.generate_data(num_samples=num_samples)

        # Initialize normalizers
        self.theta_normalizer = DataNormalizer()
        self.data_normalizer = DataNormalizer()

        if scale:
            # Fit and transform thetas and data if scale is True
            self.theta_normalizer.fit(theta)
            theta_norm = self.theta_normalizer.transform(theta)

            self.data_normalizer.fit(data)
            data_norm = self.data_normalizer.transform(data)

            return theta_norm, data_norm, self.theta_normalizer, self.data_normalizer
        else:
            # Return the original data without transformation if scale is False
            return theta, data

    def observed_data(self, true_params=None):
        """
        Generate observed data based on true parameters and return the normalized observed data.

        Parameters:
        - true_params (torch.Tensor, optional): True parameters to simulate the observed data. If not provided, use the class's true_params attribute.

        Returns:
        - torch.Tensor: Normalized observed data.
        """
        if true_params is None:
            true_params = self.true_params

        # Ensure that normalizers are available
        if self.theta_normalizer is None or self.data_normalizer is None:
            raise ValueError("Normalizers have not been initialized. Call prepare_data first.")

        # Simulate observed data with true parameters
        observed_data = self.simulator(true_params)

        # Normalize the observed data using the previously fit data normalizer
        observed_data_norm = self.data_normalizer.transform(observed_data.unsqueeze(0))

        return observed_data_norm
    
    def plot_prior(self, params):
        """
        Plot a scatter plot of parameters.

        Parameters:
        - params (torch.Tensor): Tensor of parameters, where each row is a set of parameters.
        """
        params = params.cpu().numpy()
        plt.figure(figsize=(10, 5))
        plt.scatter(params[:, 0], params[:, 1], alpha=0.5)
        plt.title('Scatter Plot of Parameters')
        plt.xlabel('Parameter 1')
        plt.ylabel('Parameter 2')
        plt.grid(True)
        plt.show()

    def plot_observation(self, observations, num_samples=5):
        """
        Plot overlapping time series of observations.

        Parameters:
        - observations (torch.Tensor): Tensor of observed data, each row is a time series.
        - num_samples (int): Number of samples to consider for plotting.
        """
        observations = observations.cpu().numpy()
        num_samples = min(num_samples, observations.shape[0])

        plt.figure(figsize=(10, 5))
        for i in range(num_samples):
            plt.plot(observations[i], alpha=0.7)

        plt.title(f'Overlapping Time Series of Observed Data for {num_samples} Samples')
        plt.xlabel('Time Step')
        plt.ylabel('Observed Value')
        plt.grid(True)
        plt.show()

    def check_normalizer(self):
        """
        Checks if the normalizer properly normalizes and denormalizes the data.
        """
        # Sample 100 points from the prior
        sampled_params = self.prior(num_samples=100)
        
        # Generate observed data using the simulator
        observed_data = torch.stack([self.simulator(params) for params in sampled_params])

        # Normalize the sampled parameters and observed data
        sampled_params_norm = self.theta_normalizer.transform(sampled_params)
        observed_data_norm = self.data_normalizer.transform(observed_data)

        # Denormalize the normalized data
        sampled_params_denorm = self.theta_normalizer.inverse_transform(sampled_params_norm)
        observed_data_denorm = self.data_normalizer.inverse_transform(observed_data_norm)

        # Compare the original and denormalized data
        params_check = torch.allclose(sampled_params, sampled_params_denorm, atol=1e-5)
        data_check = torch.allclose(observed_data, observed_data_denorm, atol=1e-5)

        if params_check and data_check:
            print("Normalization and denormalization process is consistent for both parameters and observed data.")
        else:
            print("There is a discrepancy in the normalization and denormalization process.")

    
    def plot_posterior(self, posterior):
        """
        Creates a scatter plot with a triangular prior indicated by dotted lines and scatter points for estimates.

        :param posterior: A tensor with shape [n_samples, 2] for posterior samples.
        """
        # Convert the tensor to a NumPy array for plotting
        data = posterior.cpu().numpy()

        # Define the corners of the triangular prior
        triangle_corners = np.array([[-2, 1], [2, 1], [0, -1]])

        # Create the scatter plot
        plt.figure(figsize=(10, 5))
        plt.scatter(data[:, 0], data[:, 1], alpha=1,  s=100.5, label='Estimated Posterior')

        # Draw the triangle with dashed lines
        plt.plot([triangle_corners[0][0], triangle_corners[1][0]], [triangle_corners[0][1], triangle_corners[1][1]], 'k--', label='Prior')
        plt.plot([triangle_corners[1][0], triangle_corners[2][0]], [triangle_corners[1][1], triangle_corners[2][1]], 'k--')
        plt.plot([triangle_corners[2][0], triangle_corners[0][0]], [triangle_corners[2][1], triangle_corners[0][1]], 'k--')

        plt.scatter([0.6], [0.2], color='red', s=100, label='True Value', zorder=5)

        # Set the axes limits
        plt.xlim(-2, 2)
        plt.ylim(-1, 1)

        # Add labels and title
        plt.xlabel('Theta 1')
        plt.ylabel('Theta 2')
        plt.title('VAE: Posterior MA2')
        
        # Add legend
        plt.legend()

        # Show the plot
        plt.show()

    def posterior_hist(self, posterior):
        """
        Plots histograms of the posterior parameters.

        :param posterior: A tensor with shape [n_samples, 2] for posterior samples.
        """
        # Convert the tensor to a NumPy array for plotting
        data = posterior.numpy()

        # Create histograms for each parameter
        plt.figure(figsize=(12, 5))

        # Theta 1
        plt.subplot(1, 2, 1)
        plt.hist(data[:, 0], bins=30, alpha=0.7)
        plt.title('Histogram of Theta 1')
        plt.xlabel('Theta 1')
        plt.ylabel('Frequency')

        # Theta 2
        plt.subplot(1, 2, 2)
        plt.hist(data[:, 1], bins=30, alpha=0.7)
        plt.title('Histogram of Theta 2')
        plt.xlabel('Theta 2')

        # Show the plot
        plt.tight_layout()
        plt.show()

