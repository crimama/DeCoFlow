"""
Patchify Helper Class.

Convert feature maps to patches using unfold operation.
"""

import torch.nn as nn


class PatchMaker:
    """Convert feature maps to patches using unfold operation."""

    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert (B, C, H, W) to (B*Grid, C, P, P) format."""
        padding = int((self.patchsize - 1) / 2)
        unfolder = nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)

        # Calculate spatial grid size
        number_of_total_patches = []
        for side in features.shape[-2:]:
            n_patches = (side + 2 * padding - 1 * (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))

        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features
