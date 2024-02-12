import torch
import numpy as np
from cvaei.helper import DataNormalizer
import matplotlib.pyplot as plt


class GKDistribution:
    def __init__(self, true_params=None):
        """
        Initialize the MA2 model with optional true parameters.

        Parameters:
        - true_params (torch.Tensor, optional): The true parameters for the MA2 model.
        """
        if true_params is None:
            self.true_params = torch.tensor([3, 1, 2, 0.5], dtype=torch.float)
        else:
            self.true_params = true_params

        self.theta_normalizer = None
        self.data_normalizer = None


    def simulator(self, params, c=0.8, n_obs =1000, seed=42):
        """
        Vectorized sampling from the g-and-k distribution using provided parameters.
        
        Parameters:
        - params (torch.Tensor): The parameters for the GK model [A, B, g, k] for each sample.
        - c (float, optional): Overall asymmetry parameter, default is 0.8.
        - seed (int, optional): Seed for random number generation to ensure reproducibility.
        
        Returns:
        - torch.Tensor: Simulated data based on the GK model.
        """
        if seed is not None:
            torch.manual_seed(seed)


        if params.ndim == 1:
            params = params.unsqueeze(0)

        A, B, g, k = params.t()

        # Generate standard normal variates for each set of parameters, shape [N, n_obs]
        z = torch.randn((len(params), n_obs))

        # Evaluate the quantile function Q_{gnk} for each set of parameters, expanded for each observation
        term = 1 + c * ((1 - torch.exp(-g.unsqueeze(1) * z)) / (1 + torch.exp(-g.unsqueeze(1) * z)))
        y = A.unsqueeze(1) + B.unsqueeze(1) * term * (1 + z.pow(2)).pow(k.unsqueeze(1)) * z
        return y
    

    @staticmethod
    def prior(num_samples):
        """
        Sample parameters from the prior distribution using PyTorch.
        """
        A = torch.rand(num_samples) * 4.9 + 0.1  # Uniformly distributed between 0.1 and 5
        B = torch.rand(num_samples) * 4.9 + 0.1
        g = torch.rand(num_samples) * 4.9 + 0.1
        k = torch.rand(num_samples) * 4.9 + 0.1
        return torch.stack((A, B, g, k), dim=1)
    
    @staticmethod
    def clean_data(data, lower_bound=-10, upper_bound=50, seed=None):
        """
        Clean the data by replacing outliers with random values within specified bounds.
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        mask = (data < lower_bound) | (data > upper_bound)
        random_replacements = lower_bound + (upper_bound - lower_bound) * torch.rand(mask.sum())
        data[mask] = random_replacements
        
        return data

    def generate_data(self, num_samples=1000, seed=42):
        """
        Generate data samples based on the prior and vectorized simulator.
        """
        theta = self.prior(num_samples=num_samples)
        data = self.simulator(theta, seed=seed)  # Vectorized simulation
        data = self.clean_data(data)
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
        N, num_vars = params.shape
        if num_vars != 4:
            raise ValueError("Data must have 4 variables for histograms.")
        
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        
        for i in range(num_vars):
            axs[i].hist(params[:, i], bins=20, alpha=0.7, label=f'Variable {i+1}')
            axs[i].set_title(f'Histogram of Variable {i+1}')
            axs[i].set_xlabel('Value')
            axs[i].set_ylabel('Frequency')
            axs[i].legend(loc='upper right')
        
        plt.tight_layout()
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

        plt.figure(figsize=(10, 4))
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


    def plot_posterior(self, posterior, true_params=None):
        """
        Create a scatter plot of posterior samples for each parameter.
        
        Parameters:
        - posterior (torch.Tensor): A tensor with shape [n_samples, n_params] for posterior samples.
        - true_params (list or array, optional): The true parameter values. If None, uses stored true_params.
        """
        if true_params is None:
            true_params = self.true_params.numpy()
        else:
            true_params = np.array(true_params)

        # Convert tensor to NumPy array for plotting
        data = posterior.cpu().numpy()

        num_vars = data.shape[1]  # Number of parameters
        fig, axes = plt.subplots(1, num_vars, figsize=(num_vars * 4, 4))

        # Generate index for each sample
        index = np.arange(data.shape[0])

        for i in range(num_vars):
            axes[i].scatter(index, data[:, i], alpha=0.5, color='blue', label=f'Param {i+1} Samples')
            axes[i].axhline(y=true_params[i], color='red', linestyle='--', label='True Value')
            axes[i].set_title(f'Parameter {i+1}')
            axes[i].set_xlabel('Sample Index')
            axes[i].set_ylabel(f'Value')
            axes[i].legend()

        plt.tight_layout()
        plt.show()



    def posterior_hist(self, posterior, true_params=None):
        """
        Plots histograms of the posterior parameters.

        :param posterior: A tensor with shape [n_samples, 2] for posterior samples.
        """

        if true_params is None:
            true_params = self.true_params.numpy()

        # Convert tensor to NumPy array for plotting
        data = posterior.cpu().numpy()

        num_vars = data.shape[1]
        fig, axes = plt.subplots(1, num_vars, figsize=(num_vars * 4, 4))
    
        labels = [f'Dim {i}' for i in range(num_vars)]
        
        for i in range(num_vars):
            axes[i].hist(data[:, i], bins=30, alpha=0.5, color='blue')
            axes[i].axvline(x=true_params[i], color='red', linestyle='--')
            axes[i].set_title(labels[i])

        plt.tight_layout()
        plt.show()
