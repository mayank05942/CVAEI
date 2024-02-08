import torch

class DataNormalizer:
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
        self.scale = torch.where(self.range == 0, torch.ones_like(self.range), self.range)

    def transform(self, tensor):
        """
        Apply min-max scaling to the tensor based on the min and range from the fit.
        """
        if self.min is None or self.range is None:
            raise RuntimeError("DataNormalizer instance needs to be fitted before calling transform.")
        tensor = tensor.to(self.device)
        return (tensor - self.min) / self.scale

    def inverse_transform(self, tensor):
        """
        Scale back the data to the original representation.
        """
        if self.min is None or self.range is None:
            raise RuntimeError("DataNormalizer instance needs to be fitted before calling inverse_transform.")
        tensor = tensor.to(self.device)
        # Rescale only the features that were not constant
        rescaled_tensor = tensor * self.scale + self.min
        # Handle constant features (where range is zero)
        constant_features = self.range == 0
        # Ensure the mask has the same number of dimensions as the tensor being indexed
        constant_features_expanded = constant_features.expand_as(rescaled_tensor)
        rescaled_tensor[constant_features_expanded] = self.min[constant_features].expand_as(rescaled_tensor[constant_features_expanded])
        return rescaled_tensor
