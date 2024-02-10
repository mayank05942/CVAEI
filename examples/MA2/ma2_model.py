import torch
from cvaei.helper import DataNormalizer

class MovingAverage2:
    def __init__(self, true_params=None):
        """
        Initialize the MA2 model with optional true parameters.

        Parameters:
        - true_params (torch.Tensor, optional): The true parameters for the MA2 model.
        """
        if true_params is None:
            self.true_params = torch.tensor([0.6, 0.5]) 
        else:
            self.true_params = true_params

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
    
    def prepare_data(self, num_samples=1000):
        """
        Generate, normalize data and parameters, and return them with their normalizers.

        Parameters:
        - num_samples (int): Number of samples to generate and normalize.

        Returns:
        - Tuple containing normalized theta, normalized data, theta normalizer, and data normalizer.
        """
        # Generate data
        theta, data = self.generate_data(num_samples=num_samples)

        # Initialize normalizers
        theta_normalizer = DataNormalizer()
        data_normalizer = DataNormalizer()

        # Fit and transform thetas and data
        theta_normalizer.fit(theta)
        train_theta_norm = theta_normalizer.transform(theta)

        data_normalizer.fit(data)
        train_data_norm = data_normalizer.transform(data)

        return train_theta_norm, train_data_norm, theta_normalizer, data_normalizer