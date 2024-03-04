import torch

# class DataNormalizer:
#     """
#     The DataNormalizer class implements a custom MinMax scaler
#     """
#     def __init__(self):
#         self.min = None
#         self.range = None
#         self.scale = None
#         self.device = None

#     def fit(self, tensor):
#         """
#         Compute the minimum and range of each feature from the training data tensor.
#         """
#         self.device = tensor.device
#         self.min = torch.min(tensor, dim=0, keepdim=True)[0]
#         self.range = torch.max(tensor, dim=0, keepdim=True)[0] - self.min

#         # Prevent division by zero for features with constant value
#         self.scale = torch.where(
#             self.range == 0, torch.ones_like(self.range), self.range)

#     def transform(self, tensor):
#         """
#         Apply min-max scaling to the tensor based on the min and range from the fit.
#         """
#         if self.min is None or self.range is None:
#             raise RuntimeError(
#                 "DataNormalizer instance needs to be fitted before calling transform.")
#         tensor = tensor.to(self.device)
#         return (tensor - self.min) / self.scale

#     def inverse_transform(self, tensor):
#         """
#         Scale back the data to the original representation.
#         """
#         if self.min is None or self.range is None:
#             raise RuntimeError(
#                 "DataNormalizer instance needs to be fitted before calling inverse_transform.")

#         tensor = tensor.to(self.device)

#         # Rescale only the features that were not constant
#         rescaled_tensor = tensor * self.scale + self.min

#         # Handle constant features (where range is zero)
#         constant_features = self.range == 0

#         # Ensure the mask has the same number of dimensions as the tensor being indexed
#         constant_features_expanded = constant_features.expand_as(
#             rescaled_tensor)
#         rescaled_tensor[constant_features_expanded] = self.min[constant_features].expand_as(
#             rescaled_tensor[constant_features_expanded])
#         return rescaled_tensor


class DataNormalizer:
    def __init__(self):
        self.min = None
        self.range = None
        self.scale = None
        self.device = None

    def fit(self, tensor):
        # Assuming tensor shape is (N, 200) for each feature across all samples
        self.device = tensor.device
        self.min = torch.min(tensor, dim=1, keepdim=True)[0]
        self.range = torch.max(tensor, dim=1, keepdim=True)[0] - self.min
        self.scale = torch.where(
            self.range == 0, torch.ones_like(self.range), self.range
        )

    def transform(self, tensor):
        if self.min is None or self.range is None:
            raise RuntimeError(
                "DataNormalizer instance needs to be fitted before calling transform."
            )
        tensor = tensor.to(self.device)
        return (tensor - self.min) / self.scale

    def inverse_transform(self, tensor):
        if self.min is None or self.range is None:
            raise RuntimeError(
                "DataNormalizer instance needs to be fitted before calling inverse_transform."
            )
        tensor = tensor.to(self.device)
        return tensor * self.scale + self.min


class NormaliseTimeSeries:
    def __init__(self):
        self.scalers = []

    def fit(self, train_data):
        """
        Fit the DataNormalizer to the training data without transforming it.

        Parameters:
        - train_data (torch.Tensor): Training data tensor of shape Nx3x200.
        """
        self.scalers = []  # Reset scalers
        N, C, T = train_data.shape

        for i in range(C):  # Iterate over each channel/feature
            scaler = DataNormalizer()
            train_feature = train_data[:, i, :].reshape(-1, T)  # Reshape to (N, 200)
            scaler.fit(train_feature)
            self.scalers.append(scaler)

    def transform(self, data):
        """
        Apply the transformation to the data based on the fitted scalers.

        Parameters:
        - data (torch.Tensor): Data tensor of shape Nx3x200 to be normalized.

        Returns:
        - torch.Tensor: Normalized data.
        """
        if not self.scalers:
            raise RuntimeError("Scalers have not been fitted. Call fit first.")

        N, C, T = data.shape
        normalized_data = torch.zeros_like(data)

        for i, scaler in enumerate(self.scalers):
            feature = data[:, i, :].reshape(-1, T)
            normalized_data[:, i, :] = scaler.transform(feature)

        return normalized_data

    def inverse_transform(self, data):
        if not self.scalers:
            raise RuntimeError(
                "Scalers have not been fitted. Call fit_transform first."
            )

        N, C, T = data.shape
        denormalized_data = torch.zeros_like(data)

        for i, scaler in enumerate(self.scalers):
            feature = data[:, i, :].reshape(-1, T)
            denormalized_data[:, i, :] = scaler.inverse_transform(feature)

        return denormalized_data
