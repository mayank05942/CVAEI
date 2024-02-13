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
        if true_params is None:
            self.true_params = torch.tensor([0.6, 0.2])
        else:
            self.true_params = true_params

        self.theta_normalizer = None
        self.data_normalizer = None


    # def simulator(self, param, seed=42):
    #     """
    #     Simulate data using the MA2 model.

    #     Parameters:
    #     - param (torch.Tensor): The parameters for the MA2 model.
    #     - seed (int): Seed for random number generation to ensure reproducibility.

    #     Returns:
    #     - torch.Tensor: Simulated data based on the MA2 model.
    #     """
    #     torch.manual_seed(seed)
    #     n = 100
    #     m = len(param)
    #     g = torch.randn(n)
    #     gy = torch.randn(n) * 0.3
    #     y = torch.zeros(n)
    #     x = torch.zeros(n)
    #     for t in range(n):
    #         x[t] += g[t]
    #         for p in range(min(t, m)):
    #             x[t] += g[t - 1 - p] * param[p]
    #         y[t] = x[t] + gy[t]
    #     return y

    def simulator(self, params, seed=42, n = 100, device=None):
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
        # Set random seed for reproducibility
        torch.manual_seed(seed)

        # Ensure params has two dimensions [batch_size, param_dim]
        if params.ndimension() == 1:
            params = params.unsqueeze(0)
        
        # Set the device for computations
        device = device or torch.device("cpu")
        params = params.to(device)

        # Get batch size and sequence length
        batch_size, param_dim = params.size(0), params.size(1)
        

        # Generate random noise for all batches
        g = torch.randn(batch_size, n, device=device)
        gy = torch.randn(batch_size, n, device=device) * 0.3

        # Initialize x and y for all batches
        x = torch.zeros(batch_size, n, device=device)
        y = torch.zeros(batch_size, n, device=device)

        # Simulate the MA2 process in a vectorized form
        for t in range(n):
            x[:, t] += g[:, t]
            for p in range(1, min(t + 1, param_dim + 1)):  
                x[:, t] += g[:, t - p] * params[:, p - 1] if t - p >= 0 else 0  
            y[:, t] = x[:, t] + gy[:, t]

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
        data = self.simulator(theta)
        #data = torch.stack([self.simulator(t) for t in theta])
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

        return_values = (train_theta_norm, train_data_norm, self.theta_normalizer, self.data_normalizer)

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
        plt.scatter(data[:, 0], data[:, 1], alpha=0.5, label='Estimated Posterior')

        # Draw the triangular prior
        triangle_corners = np.array([[-2, 1], [2, 1], [0, -1]])
        plt.plot([triangle_corners[0][0], triangle_corners[1][0]], [triangle_corners[0][1], triangle_corners[1][1]], 'k--')
        plt.plot([triangle_corners[1][0], triangle_corners[2][0]], [triangle_corners[1][1], triangle_corners[2][1]], 'k--')
        plt.plot([triangle_corners[2][0], triangle_corners[0][0]], [triangle_corners[2][1], triangle_corners[0][1]], 'k--')

        # Plot the true value with dotted lines indicating its position
        plt.scatter([true_params[0]], [true_params[1]], color='red', s=50, label='True Value')
        plt.axvline(x=true_params[0], color='red', linestyle='--', linewidth=1)
        plt.axhline(y=true_params[1], color='red', linestyle='--', linewidth=1)

        # Set the axes limits and labels
        plt.xlim(-2, 2)
        plt.ylim(-1, 1)
        plt.xlabel('Theta 1')
        plt.ylabel('Theta 2')
        plt.title('VAE: Posterior MA2')
        plt.legend()
        #plt.grid(True)
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

        # Create histograms for each parameter
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram of Theta 1
        axs[0].hist(data[:, 0], bins=30, color='skyblue', edgecolor='black', range=(0, 1))
        axs[0].axvline(x=true_params[0], color='red', linestyle='--', linewidth=1)
        axs[0].set_title('Histogram of Theta 1')
        axs[0].set_xlabel('Theta 1')
        axs[0].set_ylabel('Frequency')

        # Histogram of Theta 2
        axs[1].hist(data[:, 1], bins=30, color='skyblue', edgecolor='black', range=(0, 1))
        axs[1].axvline(x=true_params[1], color='red', linestyle='--', linewidth=1)
        axs[1].set_title('Histogram of Theta 2')
        axs[1].set_xlabel('Theta 2')

        plt.tight_layout()
        plt.show()


    def posterior_kde(self, posterior, true_params=None):
        """
        Plots KDE of the posterior parameters.

        :param posterior: A tensor with shape [n_samples, 2] for posterior samples.
        """
        if true_params is None:
            true_params = self.true_params.numpy()

        # Convert tensor to NumPy array for plotting
        data = posterior.cpu().numpy()

        # Create KDE plots for each parameter
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # KDE of Theta 1
        sns.kdeplot(data[:, 0], ax=axs[0], fill=True, color='skyblue', edgecolor='black')
        axs[0].axvline(x=true_params[0], color='red', linestyle='--', linewidth=1)
        axs[0].set_title('KDE of Theta 1')
        axs[0].set_xlabel('Theta 1')
        axs[0].set_ylabel('Density')

        # KDE of Theta 2
        sns.kdeplot(data[:, 1], ax=axs[1], fill=True, color='skyblue', edgecolor='black')
        axs[1].axvline(x=true_params[1], color='red', linestyle='--', linewidth=1)
        axs[1].set_title('KDE of Theta 2')
        axs[1].set_xlabel('Theta 2')

        plt.tight_layout()
        plt.show()