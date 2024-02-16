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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if true_params is None:
            self.true_params = torch.tensor(
                [3, 1, 2, 0.5], dtype=torch.float32, device=self.device
            )
        else:
            self.true_params = true_params.to(self.device)

        self.theta_normalizer = None
        self.data_normalizer = None

    def simulator(self, params, c=0.8, n_obs=1000, seed=42):
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

        params = params.to(device=self.device)

        A, B, g, k = params.t()

        # Generate standard normal variates for each set of parameters, shape [N, n_obs]
        z = torch.randn((len(params), n_obs), device=self.device)

        # Evaluate the quantile function Q_{gnk} for each set of parameters, expanded for each observation
        term = 1 + c * (
            (1 - torch.exp(-g.unsqueeze(1) * z)) / (1 + torch.exp(-g.unsqueeze(1) * z))
        )
        y = (
            A.unsqueeze(1)
            + B.unsqueeze(1) * term * (1 + z.pow(2)).pow(k.unsqueeze(1)) * z
        )

        y = self.clean_data(y)

        return y

    def prior(self, num_samples):
        """
        Sample parameters from the prior distribution using PyTorch.
        """
        A = (
            torch.rand(num_samples, device=self.device) * 4.9 + 0.1
        )  # Uniformly distributed between 0.1 and 5
        B = torch.rand(num_samples, device=self.device) * 4.9 + 0.1
        g = torch.rand(num_samples, device=self.device) * 4.9 + 0.1
        k = torch.rand(num_samples, device=self.device) * 4.9 + 0.1
        return torch.stack((A, B, g, k), dim=1)

    def clean_data(self, data, lower_bound=-10, upper_bound=50, seed=55):
        """
        Clean the data by replacing outliers with random values within specified bounds, using self.device.
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Ensure data is on the specified device
        data = data.to(self.device)

        # Generate a mask for values outside the bounds
        mask = (data < lower_bound) | (data > upper_bound)

        # Generate random replacements on the correct device
        random_replacements = lower_bound + (upper_bound - lower_bound) * torch.rand(
            mask.sum(), device=self.device
        )

        # Apply replacements
        data[mask] = random_replacements

        return data

    def generate_data(self, num_samples=1000, seed=42):
        """
        Generate data samples based on the prior and vectorized simulator.
        """
        theta = self.prior(num_samples=num_samples)
        data = self.simulator(theta, seed=seed)  # Vectorized simulation
        # data = self.clean_data(data)
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
        N, num_vars = params.shape
        if num_vars != 4:
            raise ValueError("Data must have 4 variables for histograms.")

        fig, axs = plt.subplots(1, 4, figsize=(20, 5))

        for i in range(num_vars):
            axs[i].hist(params[:, i], bins=20, alpha=0.7, label=f"Variable {i+1}")
            axs[i].set_title(f"Histogram of Variable {i+1}")
            axs[i].set_xlabel("Value")
            axs[i].set_ylabel("Frequency")
            axs[i].legend(loc="upper right")

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
        Create a scatter plot of posterior samples for each parameter.

        Parameters:
        - posterior (torch.Tensor): A tensor with shape [n_samples, n_params] for posterior samples.
        - true_params (list or array, optional): The true parameter values. If None, uses stored true_params.
        """
        if true_params is None:
            true_params = self.true_params.cpu().numpy()
        else:
            true_params = np.array(true_params)

        # Convert tensor to NumPy array for plotting
        data = posterior.cpu().numpy()

        num_vars = data.shape[1]  # Number of parameters
        fig, axes = plt.subplots(1, num_vars, figsize=(num_vars * 4, 4))

        # Generate index for each sample
        index = np.arange(data.shape[0])

        for i in range(num_vars):
            axes[i].scatter(
                index, data[:, i], alpha=0.5, color="blue", label=f"Param {i+1} Samples"
            )
            axes[i].axhline(
                y=true_params[i], color="red", linestyle="--", label="True Value"
            )
            axes[i].set_title(f"Parameter {i+1}")
            # axes[i].set_xlabel("Sample Index")
            # axes[i].set_ylabel(f"Value")
            axes[i].legend()

        plt.tight_layout()
        plt.show()

    def posterior_hist(self, posterior, true_params=None):
        """
        Plots histograms of the posterior parameters.

        :param posterior: A tensor with shape [n_samples, 2] for posterior samples.
        """

        if true_params is None:
            true_params = self.true_params.cpu().numpy()

        # Convert tensor to NumPy array for plotting
        data = posterior.cpu().numpy()

        num_vars = data.shape[1]
        fig, axes = plt.subplots(1, num_vars, figsize=(num_vars * 4, 4))

        labels = [f"Dim {i}" for i in range(num_vars)]

        for i in range(num_vars):
            axes[i].hist(data[:, i], bins=30, alpha=0.5, color="blue")
            axes[i].axvline(x=true_params[i], color="red", linestyle="--")
            axes[i].set_title(labels[i])

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
