import torch
import numpy as np
from cvaei.helper import DataNormalizer
import matplotlib.pyplot as plt
import seaborn as sns
import math


class TwoMoons:
    def __init__(self, obs_data=None):
        """
        Initialize the Two Moon model with optional observed data.

        Parameters:
        - obs_data (torch.Tensor, optional): Observed data for the Two Moons model.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if obs_data is None:
            self.obs_data = torch.tensor(
                [0.0, 0.0], dtype=torch.float32, device=self.device
            )
        else:
            self.obs_data = obs_data.to(self.device)

        self.theta_normalizer = None
        self.data_normalizer = None

    # def simulator(self, theta, seed=1000):
    #     """
    #     Simulate data using the Two moons model for a batch of parameters.

    #     Parameters:
    #     - params (torch.Tensor): The batch of parameters for the two moons model. Shape: [batch_size, param_dim]
    #     - seed (int): Seed for random number generation to ensure reproducibility.
    #     - device (torch.device): The device to perform computations on.

    #     Returns:
    #     - torch.Tensor: Simulated data based on the two moons model for each set of parameters in the batch.

    #     """
    #     theta = theta.to(self.device)

    #     # Set the random seed for reproducibility if provided
    #     if seed is not None:
    #         torch.manual_seed(seed)

    #     # Ensure theta is two-dimensional for batch processing
    #     if theta.dim() == 1:
    #         theta = theta.unsqueeze(0)  # Convert to shape [1, 2]

    #     batch_size = theta.size(0)

    #     mean_radius = 1.0
    #     sd_radius = 0.1
    #     baseoffset = 1.0
    #     torch.pi = torch.acos(torch.zeros(1)) * 2

    #     a = torch.pi * (torch.rand(batch_size, device=self.device) - 0.5)
    #     r = mean_radius + torch.randn(batch_size, device=self.device) * sd_radius
    #     p = torch.stack(
    #         [r * torch.cos(a) + baseoffset, r * torch.sin(a)], dim=-1
    #     )  # p shape will be [batch_size, 2]

    #     # Fixed angle for rotation
    #     ang = torch.tensor(-torch.pi / 4.0, device=self.device)
    #     c = torch.cos(ang)
    #     s = torch.sin(ang)

    #     # Apply rotation to theta
    #     z0 = c * theta[:, 0] - s * theta[:, 1]
    #     z1 = s * theta[:, 0] + c * theta[:, 1]

    #     # Combine p and rotated theta for final output
    #     transformed = p + torch.stack(
    #         [-torch.abs(z0), z1], dim=-1
    #     )  # Output shape [batch_size, 2]

    #     return transformed

    def simulator(self, thetas, seed=42):

        if seed is not None:
            torch.manual_seed(seed)

        # Ensure theta is 2-dimensional
        if thetas.dim() == 1:
            thetas = thetas.unsqueeze(0)

        # Batch size
        N = thetas.size(0)

        # Generate random noise
        alpha = torch.rand(N, device=self.device) * torch.tensor(np.pi) - torch.tensor(
            0.5 * np.pi, device=self.device
        )
        r = 0.1 + 0.01 * torch.randn(N, device=self.device)

        # Calculate positions
        x1 = r * torch.cos(alpha) + 0.25
        x2 = r * torch.sin(alpha)
        y1 = -torch.abs(thetas[:, 0] + thetas[:, 1]) / torch.sqrt(
            torch.tensor(2.0, device=self.device)
        )
        y2 = (-thetas[:, 0] + thetas[:, 1]) / torch.sqrt(
            torch.tensor(2.0, device=self.device)
        )

        # Output
        return torch.stack([x1 + y1, x2 + y2], dim=1)

    def prior(self, num_samples):
        """
        Sample parameters from the prior distribution using PyTorch tensors.

        Parameters:
        - num_samples (int): Number of samples to draw.

        Returns:
        - torch.Tensor: Sampled parameters from the prior distribution.
        """
        "Prior ~ U(-1,1)"
        return torch.FloatTensor(num_samples, 2).uniform_(-1, 1).to(self.device)

    def generate_data(self, num_samples=1000):
        """
        Generate data samples based on the prior and simulator.

        Parameters:
        - num_samples (int): Number of samples to generate.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Sampled parameters and corresponding simulated data.
        """

        theta = self.prior(num_samples=num_samples)
        data = self.simulator(theta)
        return theta, data

    def prepare_data(self, num_samples=1000, scale=True, validation=True):
        """
        Generate, (optionally) normalize data and parameters, and return them with their normalizers.
        Optionally generates validation data of size 10,000. Prints the shape of all generated data.

        Parameters:
        - num_samples (int): Number of samples to generate and (optionally) normalize for training.
        - scale (bool): If True, return normalized data; otherwise, return unnormalized data.
        - validation (bool): If True, also generate and return validation data of size 10,000.

        Returns:
        - Tuple containing (optionally normalized) training theta, training data,
        theta normalizer, and data normalizer. If validation is True, also returns
        (optionally normalized) validation theta and validation data.
        """
        # Generate training data
        train_theta, train_data = self.generate_data(num_samples=num_samples)

        # Initialize normalizers
        self.theta_normalizer = DataNormalizer()
        self.data_normalizer = DataNormalizer()

        if scale:
            # Normalize training data
            self.theta_normalizer.fit(train_theta)
            train_theta_norm = self.theta_normalizer.transform(train_theta)
            self.data_normalizer.fit(train_data)
            train_data_norm = self.data_normalizer.transform(train_data)
        else:
            # Use unnormalized training data
            train_theta_norm = train_theta
            train_data_norm = train_data

        print(f"Training Theta Shape: {train_theta_norm.shape}")
        print(f"Training Data Shape: {train_data_norm.shape}")

        return_values = (
            train_theta_norm,
            train_data_norm,
            self.theta_normalizer,
            self.data_normalizer,
        )

        if validation:
            # Generate validation data
            val_theta, val_data = self.generate_data(num_samples=10000)

            if scale:
                # Normalize validation data using the same normalizers as for the training data
                val_theta_norm = self.theta_normalizer.transform(val_theta)
                val_data_norm = self.data_normalizer.transform(val_data)
            else:
                # Use unnormalized validation data
                val_theta_norm = val_theta
                val_data_norm = val_data

            print(f"Validation Theta Shape: {val_theta_norm.shape}")
            print(f"Validation Data Shape: {val_data_norm.shape}")

            # Extend return values to include validation data
            return_values += (val_theta_norm, val_data_norm)

        return return_values

    def observed_data(self, obs_data=None):
        # Ensure that normalizers are available
        if self.theta_normalizer is None or self.data_normalizer is None:
            raise ValueError(
                "Normalizers have not been initialized. Call prepare_data first."
            )

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
        plt.title("Scatter Plot of Parameters")
        plt.xlabel("Parameter 1")
        plt.ylabel("Parameter 2")
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
        plt.title("Scatter Plot of observed data")
        plt.xlabel("Observation 1")
        plt.ylabel("Observation 2")
        plt.grid(True)
        plt.show()

    def check_normalizer(self):
        """
        Checks if the normalizer properly normalizes and denormalizes the data.
        """
        # Sample 100 points from the prior
        sampled_params = self.prior(num_samples=100)

        # Generate observed data using the simulator
        observed_data = self.simulator(sampled_params)

        # Normalize the sampled parameters and observed data
        sampled_params_norm = self.theta_normalizer.transform(sampled_params)
        observed_data_norm = self.data_normalizer.transform(observed_data)

        # Denormalize the normalized data
        sampled_params_denorm = self.theta_normalizer.inverse_transform(
            sampled_params_norm
        )
        observed_data_denorm = self.data_normalizer.inverse_transform(
            observed_data_norm
        )

        # Compare the original and denormalized data
        params_check = torch.allclose(sampled_params, sampled_params_denorm, atol=1e-5)
        data_check = torch.allclose(observed_data, observed_data_denorm, atol=1e-5)

        if params_check and data_check:
            print(
                "Normalization and denormalization process is consistent for both parameters and observed data."
            )
        else:
            print(
                "There is a discrepancy in the normalization and denormalization process."
            )

    def analytic_posterior(self, x_o=None, n_samples=10000):
        """
        Compute the analytic posterior for given observed data.
        """

        x_o = self.obs_data
        ang = torch.tensor(-torch.pi / 4.0, device=self.device)
        c = torch.cos(-ang)
        s = torch.sin(-ang)

        theta = torch.zeros((n_samples, 2), device=self.device)

        for i in range(n_samples):
            p = self.simulator(torch.zeros(2), seed=None).squeeze(0)
            q = torch.zeros(2, device=self.device)

            # Adjust subtraction to ensure compatibility
            q[0] = p[0] - x_o[0]
            q[1] = x_o[1] - p[1]

            if torch.rand(1, device=self.device) < 0.5:
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

        sns.scatterplot(
            x=posterior[:, 0], y=posterior[:, 1], color="blue", alpha=1, marker="+"
        )
        plt.title("Scatter Plot with Density Contours")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.show()

    def posterior_hist(self, posterior, kde=False):
        """
        Plots histograms or KDE of the posterior parameters based on the kde flag.

        :param posterior: A tensor with shape [n_samples, 2] for posterior samples.
        :param true_params: True parameters to plot as vertical lines for comparison.
        :param kde: If True, plots KDE instead of histogram.
        """

        # Convert tensor to NumPy array for plotting
        data = posterior.cpu().numpy()

        # Create plots for each parameter
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        titles = ["Theta 1", "Theta 2"]
        for i in range(2):
            if kde:
                # KDE plot
                sns.kdeplot(
                    data[:, i], ax=axs[i], fill=True, color="skyblue", edgecolor="black"
                )
                axs[i].set_ylabel("Density")
            else:
                # Histogram
                axs[i].hist(
                    data[:, i],
                    bins=30,
                    color="skyblue",
                    edgecolor="black",
                    range=(0, 1),
                )
                axs[i].set_ylabel("Frequency")

            # Common configurations for both plot types
            axs[i].set_title(f"{titles[i]} Distribution")
            axs[i].set_xlabel(titles[i])

        plt.tight_layout()
        plt.show()

    def get_info(self):
        # Directly check the device of tensor attributes
        tensor_attributes = [
            "train_theta_norm",
            "train_data_norm",
            "val_theta_norm",
            "val_data_norm",
            "observed_data",
        ]
        for attr in tensor_attributes:
            if hasattr(self, attr):
                tensor = getattr(self, attr)
                if isinstance(tensor, torch.Tensor):
                    print(f"{attr} is on device: {tensor.device}")

        # Assuming normalizers store tensors or provide a method to check their device
        normalizer_attributes = ["theta_normalizer", "data_normalizer"]
        for attr in normalizer_attributes:
            if hasattr(self, attr):
                normalizer = getattr(self, attr)
                # Example check, adjust based on your implementation of DataNormalizer
                if hasattr(
                    normalizer, "device"
                ):  # If your normalizer has a 'device' attribute
                    print(f"{attr} uses device: {normalizer.device}")
                elif hasattr(
                    normalizer, "get_device"
                ):  # Or if it has a method to get the device
                    print(f"{attr} uses device: {normalizer.get_device()}")

        # Additional checks for observed_data if it's stored differently
        if hasattr(self, "observed_data"):
            observed_data = getattr(self, "observed_data")
            if isinstance(observed_data, torch.Tensor):
                print(f"observed_data is on device: {observed_data.device}")
