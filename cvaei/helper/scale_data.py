import torch

from sklearn.preprocessing import MinMaxScaler


class DataNormalizer:
    """
    The DataNormalizer class implements a custom MinMax scaler
    """

    def __init__(self):
        self.min = None
        self.range = None
        self.scale = None
        self.device = None

    def fit(self, tensor):
        """
        Compute the minimum and range of each feature from the training data tensor.
        """
        self.device = tensor.device
        self.min = torch.min(tensor, dim=0, keepdim=True)[0]
        self.range = torch.max(tensor, dim=0, keepdim=True)[0] - self.min

        # Prevent division by zero for features with constant value
        self.scale = torch.where(
            self.range == 0, torch.ones_like(self.range), self.range
        )

    def transform(self, tensor):
        """
        Apply min-max scaling to the tensor based on the min and range from the fit.
        """
        if self.min is None or self.range is None:
            raise RuntimeError(
                "DataNormalizer instance needs to be fitted before calling transform."
            )
        tensor = tensor.to(self.device)
        return (tensor - self.min) / self.scale

    def inverse_transform(self, tensor):
        """
        Scale back the data to the original representation.
        """
        if self.min is None or self.range is None:
            raise RuntimeError(
                "DataNormalizer instance needs to be fitted before calling inverse_transform."
            )

        tensor = tensor.to(self.device)

        # Rescale only the features that were not constant
        rescaled_tensor = tensor * self.scale + self.min

        # Handle constant features (where range is zero)
        constant_features = self.range == 0

        # Ensure the mask has the same number of dimensions as the tensor being indexed
        constant_features_expanded = constant_features.expand_as(rescaled_tensor)
        rescaled_tensor[constant_features_expanded] = self.min[
            constant_features
        ].expand_as(rescaled_tensor[constant_features_expanded])
        return rescaled_tensor


class NormaliseTimeSeries:
    def __init__(self):
        self.scalers = []

    def fit(self, train_data):
        _, C, _ = train_data.shape  # Get the number of channels/features
        self.scalers = [MinMaxScaler() for _ in range(C)]

        for i in range(C):
            # Reshape and fit scaler on each feature across all samples
            feature = train_data[:, i, :].reshape(-1, 1).cpu().numpy()
            self.scalers[i].fit(feature)

    def transform(self, data):
        N, C, T = data.shape
        transformed_data = torch.zeros_like(data)

        for i in range(C):
            # Flatten, transform, and reshape back for each feature
            feature = data[:, i, :].reshape(-1, 1).cpu().numpy()
            transformed_feature = self.scalers[i].transform(feature).reshape(N, T)
            transformed_data[:, i, :] = torch.tensor(
                transformed_feature, dtype=data.dtype, device=data.device
            )

        return transformed_data

    def inverse_transform(self, data):
        N, C, T = data.shape
        inv_transformed_data = torch.zeros_like(data)

        for i in range(C):
            # Flatten, inverse transform, and reshape back for each feature
            feature = data[:, i, :].reshape(-1, 1).cpu().numpy()
            inv_transformed_feature = (
                self.scalers[i].inverse_transform(feature).reshape(N, T)
            )
            inv_transformed_data[:, i, :] = torch.tensor(
                inv_transformed_feature, dtype=data.dtype, device=data.device
            )

        return inv_transformed_data
