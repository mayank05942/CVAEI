import torch
import numpy as np
from cvaei.helper import DataNormalizer
import matplotlib.pyplot as plt
import seaborn as sns


class TwoMoons:
    def __init__(self, obs_data=None):
        """
        Initialize the Two Moon model with optional observed data.

        Parameters:
        - obs_data (torch.Tensor, optional): Observed data for the Two Moons model.
        """
        if obs_data is None:
            self.obs_data = torch.tensor([0.0, 0.0],dtype=torch.float)
        else:
            self.obs_data = obs_data

        self.theta_normalizer = None
        self.data_normalizer = None


    def simulator(self, param, seed=42, device = None):
        """
        Simulate data using the Two moons model.

        Parameters:
        - param (torch.Tensor): The parameters for the two moons model.
        - seed (int): Seed for random number generation to ensure reproducibility.

        Returns:
        - torch.Tensor: Simulated data based on the two moons model.
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Set the device for computations
        device = device or torch.device("cpu")

        param = param.to(device)

        # Generate noise
        alpha = torch.empty(1, device=device).uniform_(-0.5 * np.pi, 0.5 * np.pi)
        r = torch.empty(1, device=device).normal_(mean=0.1, std=0.01)

        # Forward process
        rhs1 = torch.tensor([r * torch.cos(alpha) + 0.25, r * torch.sin(alpha)], device=device)
        rhs2 = torch.tensor(
            [
                -torch.abs(param[0] + param[1]) / torch.sqrt(torch.tensor(2.0, device=device)),
                (-param[0] + param[1]) / torch.sqrt(torch.tensor(2.0, device=device)),
            ],
            device=device
        )

        return rhs1 + rhs2

    @staticmethod
    def prior(num_samples, device = None):
        """
        Sample parameters from the prior distribution using PyTorch tensors.

        Parameters:
        - num_samples (int): Number of samples to draw.

        Returns:
        - torch.Tensor: Sampled parameters from the prior distribution.
        """
        "Prior ~ U(-1,1)"
        device = device or torch.device("cpu")

        return torch.FloatTensor(num_samples, 2).uniform_(-1, 1).to(device)

    def generate_data(self, num_samples=1000, device = None):
        """
        Generate data samples based on the prior and simulator.

        Parameters:
        - num_samples (int): Number of samples to generate.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Sampled parameters and corresponding simulated data.
        """
        device = device or torch.device("cpu")

        theta = self.prior(num_samples=num_samples, device = device)
        data = torch.stack([self.simulator(t, seed =42, device = device) for t in theta])
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

    def observed_data(self, obs_data=None):
        # Ensure that normalizers are available
        if self.theta_normalizer is None or self.data_normalizer is None:
            raise ValueError("Normalizers have not been initialized. Call prepare_data first.")

        if obs_data is not None:
            self.obs_data = obs_data

        # Normalize the observed data using the previously fit data normalizer
        observed_data_norm = self.data_normalizer.transform(self.obs_data.unsqueeze(0))

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

    def plot_observation(self, observations):
        """
        2D Scatter plot of simulated data

        Parameters:
        - observations (torch.Tensor): Tensor of observed data (2D)
        - num_samples (int): Number of samples to consider for plotting.
        """
        observations = observations.cpu().numpy()
        plt.figure(figsize=(10, 5))
        plt.scatter(observations[:, 0], observations[:, 1], alpha=0.5)
        plt.title('Scatter Plot of observed data')
        plt.xlabel('Observation 1')
        plt.ylabel('Observation 2')
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

    
    def analytic_posterior(self, x_o = None, n_samples=10000, device = None):
        """
        Compute the analytic posterior for given observed data.
        """
        x_o = self.obs_data if x_o is None else x_o

        device = device or torch.device("cpu")
            
        ang = torch.tensor(-torch.pi / 4.0)
        c = torch.cos(-ang)
        s = torch.sin(-ang)

        theta = torch.zeros((n_samples, 2))

        for i in range(n_samples):
            p = self.simulator(torch.zeros(2), seed = None)
            q = torch.zeros(2)

            q[0] = p[0] - x_o[0]
            q[1] = x_o[1] - p[1]

            if torch.rand(1) < 0.5:
                q[0] = -q[0]

            theta[i, 0] = c * q[0] - s * q[1]
            theta[i, 1] = s * q[0] + c * q[1]

        return self.plot_posterior(theta)


    def plot_posterior(self, posterior):
        """
        Creates a scatter plot with density contours for posterior samples.

        :param posterior: A tensor with shape [n_samples, 2] for posterior samples.
        :param true_params: A list or array with the true parameter values [Theta 1, Theta 2].
        """
        posterior = posterior.cpu().numpy()
        plt.figure(figsize=(8, 6))
        
        sns.scatterplot(x=posterior[:, 0], y=posterior[:, 1], color='blue', alpha=1, marker='+')
        plt.title('Scatter Plot with Density Contours')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.show()


    def posterior_hist(self, posterior):
        """
        Plots histograms of the posterior parameters.

        :param posterior: A tensor with shape [n_samples, 2] for posterior samples.
        """
        # Convert tensor to NumPy array for plotting
        data = posterior.cpu().numpy()

        # Create histograms for each parameter
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram of Theta 1
        axs[0].hist(data[:, 0], bins=30, color='skyblue', edgecolor='black', range=(0, 1))
        axs[0].set_title('Histogram of Theta 1')
        axs[0].set_xlabel('Theta 1')
        axs[0].set_ylabel('Frequency')

        # Histogram of Theta 2
        axs[1].hist(data[:, 1], bins=30, color='skyblue', edgecolor='black', range=(0, 1))
        axs[1].set_title('Histogram of Theta 2')
        axs[1].set_xlabel('Theta 2')

        plt.tight_layout()
        plt.show()
