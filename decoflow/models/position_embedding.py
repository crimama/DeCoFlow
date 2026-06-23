"""
Positional Embedding Generator.

Generate and add 2D positional encodings to patch embeddings.
"""

import math
import torch


def positionalencoding2d(D, H, W, device=None):
    """
    Generate 2D positional encoding.

    Args:
        D: dimension of the model
        H: height of the positions
        W: width of the positions
        device: device to put the tensor on

    Returns:
        DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W, device=device)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2, device=device) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W, device=device).unsqueeze(1)
    pos_h = torch.arange(0.0, H, device=device).unsqueeze(1)
    P[0:D:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D+1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P


class PositionalEmbeddingGenerator:
    """Generate and add 2D positional encodings to patch embeddings."""

    def __init__(self, device):
        self.device = device

    def __call__(self, spatial_shape, patch_embeddings):
        """
        Add positional embeddings to patch embeddings.

        Args:
            spatial_shape: (H_patch, W_patch) spatial dimensions
            patch_embeddings: (B, H, W, D) patch embeddings

        Returns:
            (B, H, W, D) patch embeddings with positional encoding
        """
        H_patch, W_patch = spatial_shape
        pos_embed_dim = patch_embeddings.shape[-1]

        # Generate positional encoding
        pos_embed = positionalencoding2d(pos_embed_dim, H_patch, W_patch, device=self.device)

        # Reshape and expand for batch
        pos_embed_expanded = pos_embed.unsqueeze(0).permute(0, 2, 3, 1)
        batch_size = patch_embeddings.shape[0]
        pos_embed_batch = pos_embed_expanded.repeat(batch_size, 1, 1, 1)

        # Add positional embedding
        patch_embeddings_with_pos = patch_embeddings + pos_embed_batch

        return patch_embeddings_with_pos
