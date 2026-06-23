"""
Preprocessing (Dimensionality Reduction) modules.

Reduce feature dimensions using adaptive average pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanMapper(nn.Module):
    """Reduce feature dimensions using adaptive average pooling."""

    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        """Transform (N, C, P*P) to (N, Target_Dim) via average pooling."""
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Preprocessing(nn.Module):
    """Apply dimensionality reduction to multiple feature layers."""

    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        self.preprocessing_modules = nn.ModuleList()
        for _ in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        """Process features and stack: (N, Layers, Output_Dim)."""
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)
