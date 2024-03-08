import torch
import warnings
import numpy as np
from cvaei.helper import DataNormalizer, NormaliseTimeSeries
import matplotlib.pyplot as plt
import seaborn as sns
from .gillespy2_model_villar import Vilar_Oscillator
from gillespy2 import SSACSolver
from functools import partial
from tqdm import tqdm
from joblib import Parallel, delayed
from functools import partial
from joblib import Parallel, delayed

from gillespy2.core.events import *

# import multiprocessing as mp


from torch.multiprocessing import Pool, set_start_method

try:
    # Set the start method to 'spawn' to work correctly with CUDA in subprocesses if necessary
    set_start_method("spawn")
except RuntimeError:
    # The start method can only be set once per program, so ignore the error if it's already been set
    pass


class Villar:
    def __init__(self, model=None, true_params=None):
        """
        Initialize the MA2 model with optional true parameters.

        Parameters:
        - true_params (torch.Tensor, optional): The true parameters for the MA2 model.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model if model is not None else Vilar_Oscillator()

        if true_params is None:
            self.true_params = np.array(
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

        self.solver = SSACSolver(model=self.model)

    def simulator(self, params, model, solver):
        params_dict = {self.parameter_names[i]: param for i, param in enumerate(params)}
        result = model.run(solver=solver, timeout=0.7, variables=params_dict)

        if result.rc == 33:  # Timeout or error
            return np.full((3, 200), np.inf)
        else:
            return np.array([result[species] for species in ["C", "A", "R"]])

    def prior(self, num_samples):
        return np.random.uniform(low=self.dmin, high=self.dmax, size=(num_samples, 15))

    def generate_data(self, num_samples=1000, resample_failed=True):
        print("Generating data...")

        theta = self.prior(num_samples)

        model = Vilar_Oscillator()
        solver = SSACSolver(model=model)

        # Use partial to fix the model and solver arguments for the simulator function
        simulator_with_fixed_args = partial(self.simulator, model=model, solver=solver)

        # Use joblib's Parallel and delayed with the partial function
        series = Parallel(n_jobs=90)(
            delayed(simulator_with_fixed_args)(theta_i) for theta_i in theta
        )
        series = np.array(series)

        # Convert numpy arrays to torch tensors
        series = torch.from_numpy(series).to(dtype=torch.float32, device=self.device)
        theta = torch.from_numpy(theta).to(dtype=torch.float32, device=self.device)

        return theta, series

    # def generate_data(self, num_samples=1000, resample_failed=True):
    #     """
    #     Optimized data generation process with efficient error handling using joblib for parallel processing.
    #     """
    #     print("Generating data...")

    #     # No need to explicitly calculate num_processes for joblib, it intelligently uses available CPUs
    #     theta = self.prior(num_samples)

    #     model = Vilar_Oscillator()
    #     solver = SSACSolver(model=model)

    #     # Replace the multiprocessing part with joblib's Parallel and delayed
    #     series = Parallel(n_jobs=90)(
    #         delayed(self.simulator)(theta_i) for theta_i in theta
    #     )
    #     series = np.array(series)

    #     # Efficient handling of failed simulations (remaining unchanged)
    #     # failed_indices = np.where(
    #     #     np.isinf(series).any(axis=(1, 2)) | np.isnan(series).any(axis=(1, 2))
    #     # )[0]

    #     # while failed_indices.size > 0:
    #     #     print("Resampling for failed simulations...")
    #     #     new_theta = self.prior(len(failed_indices))
    #     #     new_series = Parallel(n_jobs=-1)(
    #     #         delayed(self.simulator)(theta_i) for theta_i in new_theta
    #     #     )
    #     #     new_series = np.array(new_series)

    #     #     for idx, new_idx in enumerate(failed_indices):
    #     #         theta[new_idx] = new_theta[idx]
    #     #         series[new_idx] = new_series[idx]

    #     #     failed_indices = np.where(
    #     #         np.isinf(series).any(axis=(1, 2)) | np.isnan(series).any(axis=(1, 2))
    #     #     )[0]

    #     # Convert numpy arrays to torch tensors
    #     series = torch.from_numpy(series).to(dtype=torch.float32, device=self.device)
    #     theta = torch.from_numpy(theta).to(dtype=torch.float32, device=self.device)

    #     return theta, series

    # def generate_data(self, num_samples=1000, resample_failed=True):
    #     """
    #     Optimized data generation process with efficient error handling.
    #     """
    #     print("Generating data...")
    #     print(mp.cpu_count())
    #     theta = self.prior(num_samples)
    #     if mp.cpu_count() == 8:
    #         num_processes = 6
    #     else:
    #         num_processes = 96

    #     chunksize = int(num_samples / (num_processes * 10))

    #     with mp.Pool(processes=num_processes) as pool:
    #         series = pool.map(self.simulator, theta)
    #     series = np.array(series)

    #     # Efficient handling of failed simulations
    #     failed_indices = np.where(
    #         np.isinf(series).any(axis=(1, 2)) | np.isnan(series).any(axis=(1, 2))
    #     )[0]

    #     while failed_indices.size > 0:
    #         print("there are failed sim")
    #         new_theta = self.prior(len(failed_indices))
    #         with mp.Pool(processes=num_processes) as pool:
    #             new_series = pool.map(self.simulator, new_theta)
    #         new_series = np.array(new_series)

    #         for idx, new_idx in enumerate(failed_indices):
    #             theta[new_idx] = new_theta[idx]
    #             series[new_idx] = new_series[idx]

    #         failed_indices = np.where(
    #             np.isinf(series).any(axis=(1, 2)) | np.isnan(series).any(axis=(1, 2))
    #         )[0]

    #     # series = np.squeeze(series, axis=1)
    #     series = torch.from_numpy(series).to(dtype=torch.float32, device=self.device)
    #     theta = torch.from_numpy(theta).to(dtype=torch.float32, device=self.device)

    #     return theta, series

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

        train_theta, train_data = self.generate_data(num_samples=num_samples)

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
            val_theta, val_data = self.generate_data(num_samples=4)

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

        observed_data = torch.tensor(
            observed_data, dtype=torch.float32, device=self.device
        )
        print(observed_data.shape)

        # Normalize the observed data using the previously fit data normalizer
        observed_data_norm = self.data_normalizer.transform(observed_data)

        return observed_data_norm

    def plot_prior(self, params):
        """
        Plot histogram plots for each of the 15 parameters.

        Parameters:
        - params (torch.Tensor): Tensor of parameters, shape Nx15, where each row is a set of parameters.
        """
        params = params.cpu().numpy()
        plt.figure(figsize=(20, 10))  # Adjust figure size for better visualization

        for i in range(15):
            plt.subplot(3, 5, i + 1)  # Arrange subplots in a 3x5 grid
            plt.hist(params[:, i], bins=30, alpha=0.75)
            plt.title(f"Histogram of Parameter {i+1}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.grid(True)

        plt.tight_layout()  # Adjust layout to prevent overlap
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
            # if f == 0:
            #     plt.legend()
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

    def posterior_hist(self, posterior, kde=False):
        """
        Plots histograms or KDE of the posterior parameters based on the kde flag.

        Parameters:
        - posterior (torch.Tensor): A tensor for posterior samples.
        - kde (bool): If True, plots KDE instead of histogram.
        """
        data = posterior.cpu().numpy()

        # Plot setup
        plt.figure(figsize=(15, 10))
        for i in range(posterior.shape[1]):
            plt.subplot(3, 5, i + 1)
            if kde:
                sns.kdeplot(
                    data[:, i],
                    fill=True,
                    color="skyblue",
                    edgecolor="black",
                    label="Posterior",
                )
                plt.ylabel("Density")
            else:
                plt.hist(
                    data[:, i],
                    bins=30,
                    alpha=0.5,
                    label="Posterior",
                    color="skyblue",
                    edgecolor="black",
                )
                plt.ylabel("Frequency")

            plt.axvline(
                x=self.true_params[i], color="r", linestyle="--", label="True Value"
            )
            plt.title(f"{self.parameter_names[i]}")
            plt.xlim([self.dmin[i], self.dmax[i]])
            plt.legend()

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

    # def simulator(self, params, transform=True):

    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore", category=UserWarning)

    #         gillespy2_logger = logging.getLogger("GillesPy2")
    #         original_level = gillespy2_logger.getEffectiveLevel()
    #         gillespy2_logger.setLevel(logging.ERROR)

    #         try:

    #             solver = SSACSolver(self.model)
    #             params = params.ravel()

    #             # GillesPy2 simulation execution
    #             res = self.model.run(
    #                 solver=solver,
    #                 timeout=0.33,  # Adjust timeout as necessary
    #                 variables={
    #                     self.parameter_names[i]: params[i]
    #                     for i in range(len(self.parameter_names))
    #                 },
    #             )

    #             # Process simulation result
    #             if res.rc == 33:
    #                 # Handling the case where simulation exceeded timeout
    #                 return np.ones((1, 3, 200))  # Return a default or error value

    #             if transform:

    #                 sp_C = res["C"]
    #                 sp_A = res["A"]
    #                 sp_R = res["R"]
    #                 simulation_result = np.vstack([sp_C, sp_A, sp_R])[np.newaxis, :, :]
    #                 return simulation_result
    #             else:
    #                 return res
    #         finally:
    #             # Reset GillesPy2 logger to its original level after the simulation
    #             gillespy2_logger.setLevel(original_level)

    # def simulator(self, params, transform=True):

    # local_solver = SSACSolver(model=self.model)
    # params = params.ravel()
    # res = self.model.run(
    #     solver=local_solver,
    #     timeout=0.33,
    #     variables={
    #         self.parameter_names[i]: params[i]
    #         for i in range(len(self.parameter_names))
    #     },
    # )

    # if res.rc == 33:
    #     return None
    # if transform:
    #     sp_C = res["C"]
    #     sp_A = res["A"]
    #     sp_R = res["R"]
    #     return np.vstack([sp_C, sp_A, sp_R])[np.newaxis, :, :]

    # else:
    #     return res


# def generate_data(self, num_samples=1000, resample_failed=True):
#         print("Generating data...")
#         print(mp.cpu_count())
#         theta = self.prior(num_samples)

#         with mp.Pool(processes=mp.cpu_count() - 2) as pool:
#             series = pool.map(self.simulator, theta)
#         series = np.array(series)

#         # Check for divergence
#         while True:
#             inf_inds = [
#                 i
#                 for i, ts in enumerate(series)
#                 if np.isinf(ts).any() or np.isnan(ts).any()
#             ]
#             if len(inf_inds) == 0:
#                 series = np.squeeze(series, axis=1)
#                 series = torch.from_numpy(series).to(
#                     dtype=torch.float32, device=self.device
#                 )
#                 theta = torch.from_numpy(theta).to(
#                     dtype=torch.float32, device=self.device
#                 )

#                 return theta, series
#             else:
#                 new_params = self.prior(num_samples)

#                 with mp.Pool(processes=mp.cpu_count() - 2) as pool:
#                     new_tst = pool.map(self.simulator, new_params)
#                     new_tst = np.asarray(new_tst)

#                 for i, ind in enumerate(inf_inds):
#                     theta[ind] = new_params[i]
#                     series[ind] = new_tst[i]
