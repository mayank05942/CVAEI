import torch
import numpy as np
from cvaei.helper import DataNormalizer, NormaliseTimeSeries
import matplotlib.pyplot as plt
import seaborn as sns
from .gillespy2_model_villar import Vilar_Oscillator
from gillespy2 import SSACSolver
import multiprocessing as mp


class Villar:
    def __init__(self, true_params=None):
        """
        Initialize the MA2 model with optional true parameters.

        Parameters:
        - true_params (torch.Tensor, optional): The true parameters for the MA2 model.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if true_params is None:
            np.array(
                [
                    50.0,
                    500.0,
                    0.01,
                    50.0,
                    50.0,
                    5.0,
                    10.0,
                    0.5,
                    1.0,
                    0.2,
                    1.0,
                    1.0,
                    2.0,
                    50.0,
                    100.0,
                ],
                dtype=np.float32,
            )
        else:
            self.true_params = true_params

        self.theta_normalizer = None
        self.data_normalizer = None

        self.parameter_names = [
            "alpha_a",
            "alpha_a_prime",
            "alpha_r",
            "alpha_r_prime",
            "beta_a",
            "beta_r",
            "delta_ma",
            "delta_mr",
            "delta_a",
            "delta_r",
            "gamma_a",
            "gamma_r",
            "gamma_c",
            "theta_a",
            "theta_r",
        ]

        self.dmin = np.array(
            [0, 100, 0, 20, 10, 1, 1, 0, 0, 0, 0.5, 0, 0, 0, 0], dtype=np.float32
        )
        self.dmax = np.array(
            [80, 600, 4, 60, 60, 7, 12, 2, 3, 0.7, 2.5, 4, 3, 70, 300], dtype=np.float32
        )

        self.model = Vilar_Oscillator()

    def simulator(self, params, transform=True):

        solver = SSACSolver(self.model)
        params = params.ravel()
        res = self.model.run(
            solver=solver,
            timeout=0.33,
            variables={
                self.parameter_names[i]: params[i]
                for i in range(len(self.parameter_names))
            },
        )

        if res.rc == 33:
            return np.ones((1, 3, 200))
        if transform:
            sp_C = res["C"]
            sp_A = res["A"]
            sp_R = res["R"]
            return np.vstack([sp_C, sp_A, sp_R])[np.newaxis, :, :]

        else:
            return res

    def prior(self, num_samples):
        ranges = self.dmax - self.dmin
        samples = np.random.rand(num_samples, len(self.dmin)) * ranges + self.dmin
        return samples

    def generate_data(self, num_samples=1000):
        """
        Generate data samples based on the prior and simulator.

        Parameters:
        - num_samples (int): Number of samples to generate.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Sampled parameters and corresponding simulated data.
        """
        theta = self.prior(num_samples=num_samples)

        with mp.Pool(processes=4) as pool:
            data = pool.map(self.simulator, theta)
        data = np.asarray(data)

        data = torch.tensor(data, dtype=torch.float32, device=self.device)
        theta = torch.tensor(theta, dtype=torch.float32, device=self.device)
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
        villar_instance = Villar()
        train_theta, train_data = villar_instance.generate_data(num_samples=num_samples)

        # Initialize normalizers
        self.theta_normalizer = DataNormalizer()
        self.data_normalizer = NormaliseTimeSeries()

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
            val_theta, val_data = villar_instance.generate_data(num_samples=4)

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
        Plot scatter plots for the first two parameters out of 15.

        Parameters:
        - params (torch.Tensor): Tensor of parameters, shape Nx15, where each row is a set of parameters.
        """
        params = params.cpu().numpy()
        plt.figure(figsize=(10, 7))
        # Plotting only the first two parameters for simplicity
        for i in range(1, 15):
            plt.subplot(3, 5, i)  # Adjust subplot grid as needed
            plt.scatter(params[:, 0], params[:, i], alpha=0.5)
            plt.title(f"Param 1 vs Param {i+1}")
            plt.xlabel("Parameter 1")
            plt.ylabel(f"Parameter {i+1}")
            plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_observation(self, observations, num_samples=5):
        """
        Plot time series of observations for each feature in separate subplots.

        Parameters:
        - observations (torch.Tensor): Tensor of observed data, shape Nx3x200.
        - num_samples (int): Number of samples to plot.
        """
        observations = observations.cpu().numpy()
        num_samples = min(num_samples, observations.shape[0])

        plt.figure(figsize=(15, 10))
        for f in range(3):  # Assuming 3 features/channels
            plt.subplot(3, 1, f + 1)  # Create a subplot for each feature
            for i in range(num_samples):
                plt.plot(
                    observations[i, f, :],
                    label=f"Sample {i+1}" if f == 0 else "",
                    alpha=0.7,
                )
            plt.title(f"Feature {f+1} Time Series for {num_samples} Samples")
            plt.xlabel("Time Step")
            plt.ylabel(f"Feature {f+1} Value")
            plt.grid(True)
            if f == 0:
                plt.legend()
        plt.tight_layout()
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
