import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_squared_error
import torch
from sklearn import metrics


class InferenceMetrics:
    def __init__(
        self, true_param, observed_data, posterior_params, posterior_sims, **kwargs
    ):

        self.true_param = self._repeat_to_match(
            self._ensure_numpy(true_param, posterior_params)
        )
        self.observed_data = self._repeat_to_match(
            self._ensure_numpy(observed_data, posterior_sims)
        )
        self.posterior_params = self._ensure_numpy(posterior_params)
        self.posterior_sims = self._ensure_numpy(posterior_sims)
        self.method = kwargs.get("method", "linear")

    def _ensure_numpy(self, data):
        """
        Ensure the data is a numpy array, converting it if it's a torch tensor.

        :param data: Data to be checked and potentially converted.
        :return: Numpy array of the data.
        """
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, (list, tuple)):
            return np.array(data)
        return data

    def _repeat_to_match(self, data, reference_data):
        """
        Repeat data to match the size of the first dimension of reference_data.

        :param data: Data to be repeated (true_param or observed_data).
        :param reference_data: The reference data (posterior_params or posterior_sims) to match the size.
        :return: Numpy array of repeated data.
        """
        repeat_times = reference_data.shape[0]  # Number of samples in reference_data
        repeated_data = np.tile(data, (repeat_times, 1))
        return repeated_data

    def mmd_linear(self, X, Y):
        """MMD using linear kernel (i.e., k(x,y) = <x,y>)
        Note that this is not the original linear MMD, only the reformulated and faster version.
        The original version is:
            def mmd_linear(X, Y):
                XX = np.dot(X, X.T)
                YY = np.dot(Y, Y.T)
                XY = np.dot(X, Y.T)
                return XX.mean() + YY.mean() - 2 * XY.mean()

        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]

        Returns:
            [scalar] -- [MMD value]
        """
        delta = X.mean(0) - Y.mean(0)
        return delta.dot(delta.T)

    def mmd_rbf(self, X, Y, gamma=1.0):
        """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]

        Keyword Arguments:
            gamma {float} -- [kernel parameter] (default: {1.0})

        Returns:
            [scalar] -- [MMD value]
        """
        XX = metrics.pairwise.rbf_kernel(X, X, gamma)
        YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
        XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
        return XX.mean() + YY.mean() - 2 * XY.mean()

    def mmd_poly(self, X, Y, degree=2, gamma=1, coef0=0):
        """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)

        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]

        Keyword Arguments:
            degree {int} -- [degree] (default: {2})
            gamma {int} -- [gamma] (default: {1})
            coef0 {int} -- [constant item] (default: {0})

        Returns:
            [scalar] -- [MMD value]
        """
        XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
        YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
        XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
        return XX.mean() + YY.mean() - 2 * XY.mean()

    def mse_loss(self, x, y):
        """
        Calculate the Mean Squared Error (MSE) loss between two numpy arrays, x and y,
        accommodating both data formats. The data can be in N x S x T format (3D) or N x L format (2D).

        :param x: Numpy array of observed data.
        :param y: Numpy array of simulated or predicted data.
        :return: float or array of MSE loss.
        """
        if x.ndim == 3:  # For N x S x T format
            N, S, T = x.shape
            mse_values = np.zeros(S)

            for s in range(S):  # Loop over each feature set
                for t in range(T):  # Compute MSE across T for each S
                    mse_value = np.mean((x[:, s, t] - y[:, s, t]) ** 2)
                    mse_values[s] += mse_value
                mse_values[s] /= T  # Average over T

            return np.mean(mse_values)  # Return mean MSE across all features
        else:  # For N x L format
            return np.mean((x - y) ** 2)

    def log_mmd(self, x, y, method=None):
        """
        Compute Maximum Mean Discrepancy (MMD) between two dist

        :param x: numpy array of first sample distribution.
        :param y: numpy array of second sample distribution.
        :return: float MMD value.
        """
        if self.method == "linear":
            return self.mmd_linear(x, y)
        elif self.method == "rbf":
            return self.mmd_rbf(x, y)
        elif self.method == "poly":
            return self.mmd_poly(x, y)
        else:
            raise ValueError(
                f"Unsupported method '{method}'. Supported methods are 'linear', 'rbf', and 'poly'."
            )

    def compute_wasserstein_distances(self, x, y):
        """
        Compute the Wasserstein distance for each feature or feature set, accommodating both data formats.

        :param x: numpy array of first distribution, in N x L or N x S x T format.
        :param y: numpy array of second distribution, in N x L or N x S x T format.
        :return: mean of the Wasserstein distances for all features or feature sets.
        """
        if x.ndim == 3:
            # If data is in N x S x T format, compute Wasserstein distance per feature set
            N, S, T = x.shape
            wasserstein_distances = np.zeros(S)
            for s in range(S):
                for t in range(T):
                    wasserstein_distances[s] += wasserstein_distance(
                        x[:, s, t], y[:, s, t]
                    )
            wasserstein_distances /= T  # Average over T
        else:
            # If data is in N x L format, compute Wasserstein distance per feature
            N, L = x.shape
            wasserstein_distances = np.zeros(L)
            for l in range(L):
                wasserstein_distances[l] = wasserstein_distance(x[:, l], y[:, l])
        return wasserstein_distances.mean()

    def posterior_eval_test(self):
        """
        Perform evaluation tests including MSE loss, log MMD, and log EMD between posterior parameters and simulated data.

        :return: dict of evaluation metrics.
        """
        metrics = {}
        metrics["MSE Loss"] = self.mse_loss(self.true_param, self.posterior_params)
        metrics["MMD"] = self.log_mmd(self.true_param, self.posterior_params)
        metrics["EMD"] = self.compute_wasserstein_distances(
            self.true_param, self.posterior_params
        )

        return metrics

    def posterior_predictive_check(self):
        metrics = {}

        metrics["MSE Loss"] = self.mse_loss(self.observed_data, self.posterior_sims)
        metrics["MMD"] = self.log_mmd(self.observed_data, self.posterior_sims)
        metrics["EMD"] = self.compute_wasserstein_distances(
            self.observed_data, self.posterior_sims
        )

        return metrics
