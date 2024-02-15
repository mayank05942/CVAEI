import torch
import numpy as np
from cvaei.helper import DataNormalizer
import matplotlib.pyplot as plt
import seaborn as sns


class MovingAverage2:
    def __init__(self, true_params=None):
        """
        Initialize the MA2 model with optional true parameters.

        Parameters:
        - true_params (torch.Tensor, optional): The true parameters for the MA2 model.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if true_params is None:
            self.true_params = torch.tensor(
                [0.6, 0.2], dtype=torch.float32, device=self.device
            )
        else:
            self.true_params = true_params.to(self.device)

        self.theta_normalizer = None
        self.data_normalizer = None

    def simulator(self, params, seed=42, n=100):
        """
        Simulate data using the MA2 model for a batch of parameters.

        Parameters:
        - params (torch.Tensor): The batch of parameters for the MA2 model.
        - seed (int): Seed for random number generation to ensure reproducibility.
        - device (torch.device): The device to perform computations on.
        - n: Size of time series

        Returns:
        - torch.Tensor: Simulated data based on the MA2 model for each set of parameters in the batch.

        """

        torch.manual_seed(seed)

        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        params = params.to(self.device)

        if params.ndimension() == 1:
            params = params.unsqueeze(0)

        # Preparing the random inputs
        g = torch.randn(n, device=self.device)
        gy = torch.randn(n, device=self.device) * 0.3

        # Expand g to match the batch size and perform batch operations
        g_expanded = g.expand(params.size(0), n)
        gy_expanded = gy.expand(params.size(0), n)

        # Initialize the output tensor
        y = torch.zeros((params.size(0), n), device=self.device)

        # Apply MA2 model in a vectorized manner
        for p in range(params.size(1)):
            if p == 0:
                y[:, 1:] += g_expanded[:, :-1] * params[:, p : p + 1]
            else:
                y[:, p + 1 :] += g_expanded[:, : -p - 1] * params[:, p : p + 1]

        # Add the original noise and the additional noise term
        y += g_expanded + gy_expanded

        return y

    def prior(self, num_samples):
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
            r = torch.rand(2, device=self.device) * torch.tensor(
                [4, 2], device=self.device
            ) + torch.tensor([-2, -1], device=self.device)
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
        data = self.simulator(theta)
        # data = torch.stack([self.simulator(t) for t in theta])
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

    def observed_data(self, true_params=None):
        """
        Generate observed data based on true parameters and return the normalized observed data.

        Parameters:
        - true_params (torch.Tensor, optional): True parameters to simulate the observed data.
           If not provided, use the class's true_params attribute.

        Returns:
        - torch.Tensor: Normalized observed data.
        """

        if true_params is None:
            true_params = self.true_params

        # Ensure that normalizers are available
        if self.theta_normalizer is None or self.data_normalizer is None:
            raise ValueError(
                "Normalizers have not been initialized. Call prepare_data first."
            )

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
        plt.title("Scatter Plot of Parameters")
        plt.xlabel("Parameter 1")
        plt.ylabel("Parameter 2")
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

        plt.title(f"Overlapping Time Series of Observed Data for {num_samples} Samples")
        plt.xlabel("Time Step")
        plt.ylabel("Observed Value")
        plt.grid(True)
        plt.show()

    def check_normalizer(self):
        """
        Checks if the normalizer properly normalizes and denormalizes the data.
        """
        # Sample 100 points from the prior
        sampled_params = self.prior(num_samples=100)

        # Generate observed data using the simulator
        observed_data = torch.stack(
            [self.simulator(params) for params in sampled_params]
        )

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

    def plot_posterior(self, posterior, true_params=None):
        """
        Creates a scatter plot with a triangular prior indicated by dotted lines and scatter points for estimates.
        Adds dotted lines to indicate the true parameter values.

        :param posterior: A tensor with shape [n_samples, 2] for posterior samples.
        :param true_params: A list or array with the true parameter values [Theta 1, Theta 2].
        """

        if true_params is None:
            true_params = self.true_params.numpy()

        # Convert tensor to NumPy array for plotting
        data = posterior.cpu().numpy()

        # Create the scatter plot
        plt.figure(figsize=(10, 5))
        plt.scatter(data[:, 0], data[:, 1], alpha=0.5, label="Estimated Posterior")

        # Draw the triangular prior
        triangle_corners = np.array([[-2, 1], [2, 1], [0, -1]])
        plt.plot(
            [triangle_corners[0][0], triangle_corners[1][0]],
            [triangle_corners[0][1], triangle_corners[1][1]],
            "k--",
        )
        plt.plot(
            [triangle_corners[1][0], triangle_corners[2][0]],
            [triangle_corners[1][1], triangle_corners[2][1]],
            "k--",
        )
        plt.plot(
            [triangle_corners[2][0], triangle_corners[0][0]],
            [triangle_corners[2][1], triangle_corners[0][1]],
            "k--",
        )

        # Plot the true value with dotted lines indicating its position
        plt.scatter(
            [true_params[0]], [true_params[1]], color="red", s=50, label="True Value"
        )
        plt.axvline(x=true_params[0], color="red", linestyle="--", linewidth=1)
        plt.axhline(y=true_params[1], color="red", linestyle="--", linewidth=1)

        # Set the axes limits and labels
        plt.xlim(-2, 2)
        plt.ylim(-1, 1)
        plt.xlabel("Theta 1")
        plt.ylabel("Theta 2")
        plt.title("VAE: Posterior MA2")
        plt.legend()
        # plt.grid(True)
        plt.show()

    def posterior_hist(self, posterior, true_params=None, kde=False):
        """
        Plots histograms or KDE of the posterior parameters based on the kde flag.

        :param posterior: A tensor with shape [n_samples, 2] for posterior samples.
        :param true_params: True parameters to plot as vertical lines for comparison.
        :param kde: If True, plots KDE instead of histogram.
        """
        if true_params is None:
            true_params = self.true_params.cpu().numpy()

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
            axs[i].axvline(x=true_params[i], color="red", linestyle="--", linewidth=1)
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
