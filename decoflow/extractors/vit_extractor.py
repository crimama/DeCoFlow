"""
ViT Feature Extractor (DINOv2 ViT Support).

Extract features from specific Transformer blocks of ViT backbone.
"""

import numpy as np
import torch
import torch.nn as nn
import timm

from decoflow.extractors.hooks import ForwardHook, LastLayerToExtractReachedException


class ViTFeatureAggregator(nn.Module):
    """Extract features from specific Transformer blocks of ViT backbone."""

    def __init__(self, backbone, blocks_to_extract_from):
        super(ViTFeatureAggregator, self).__init__()
        self.blocks_to_extract_from = blocks_to_extract_from
        self.backbone = backbone
        self.outputs = {}

        # Register hooks
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.backbone.hook_handles = []

        for block_idx in blocks_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs,
                f"block_{block_idx}",
                f"block_{blocks_to_extract_from[-1]}"
            )
            if hasattr(backbone, 'blocks'):
                network_layer = backbone.blocks[block_idx]
            else:
                raise ValueError("Backbone does not have 'blocks' attribute")

            self.backbone.hook_handles.append(
                network_layer.register_forward_hook(forward_hook)
            )

    def forward(self, images):
        self.outputs.clear()
        with torch.no_grad():
            try:
                _ = self.backbone(images)
            except LastLayerToExtractReachedException:
                pass
        return self.outputs

    def feature_dimensions(self, input_shape, device):
        """Calculate embedding dimensions for each block."""
        _input = torch.ones([1] + list(input_shape)).to(device)
        _output = self(_input)
        return [_output[f"block_{idx}"].shape[-1] for idx in self.blocks_to_extract_from]


class ViTPatchCoreExtractor(nn.Module):
    """
    ViT-based PatchCore Feature Extractor.
    Uses transformer block outputs directly without additional patchification.
    """

    def __init__(self,
                 backbone_name="vit_base_patch14_dinov2.lvd142m",
                 blocks_to_extract=[8, 9, 10, 11],
                 input_shape=(3, 224, 224),
                 target_embed_dimension=1024,
                 device='cuda',
                 remove_cls_token=True):
        super(ViTPatchCoreExtractor, self).__init__()

        self.device = device
        self.blocks_to_extract_from = blocks_to_extract
        self.remove_cls_token = remove_cls_token

        # Load backbone (pass img_size for positional embedding interpolation)
        img_size = input_shape[1] if len(input_shape) >= 2 else 224
        self.backbone = timm.create_model(backbone_name, pretrained=True, img_size=img_size)
        self.backbone.to(device)
        self.backbone.eval()

        # Setup feature aggregator
        self.feature_aggregator = ViTFeatureAggregator(
            self.backbone, self.blocks_to_extract_from
        )

        # Calculate embedding dimensions
        feature_dims = self.feature_aggregator.feature_dimensions(input_shape, device)

        # Setup preprocessing (dimension alignment)
        self.preprocessing = nn.ModuleList()
        for dim in feature_dims:
            if dim != target_embed_dimension:
                self.preprocessing.append(nn.Linear(dim, target_embed_dimension))
            else:
                self.preprocessing.append(nn.Identity())
        self.preprocessing.to(device)

        self.target_embed_dimension = target_embed_dimension
        self.num_blocks = len(blocks_to_extract)

    def forward(self, images, return_spatial_shape=False):
        """
        Extract patch embeddings from images.

        Args:
            images: (B, C, H, W) input images
            return_spatial_shape: If True, return spatial shape as well

        Returns:
            features: (B, H_patch, W_patch, Target_Embed_Dimension)
            spatial_shape (optional): (H_patch, W_patch) tuple
        """
        images = images.to(self.device)
        batch_size = images.shape[0]

        # 1. Extract features from transformer blocks
        features_dict = self.feature_aggregator(images)

        # 2. Collect features from each block (B, N_tokens, D)
        features = []
        for block_idx in self.blocks_to_extract_from:
            feat = features_dict[f"block_{block_idx}"]

            # Remove CLS token
            if self.remove_cls_token:
                feat = feat[:, 1:, :]  # (B, N_patches, D)

            features.append(feat)

        # 3. Apply preprocessing (dimension adjustment)
        processed_features = []
        for i, feat in enumerate(features):
            feat = self.preprocessing[i](feat)
            processed_features.append(feat)

        # 4. Aggregate features from multiple blocks
        stacked_features = torch.stack(processed_features, dim=2)
        aggregated_features = stacked_features.mean(dim=2)

        # 5. Reshape to spatial structure
        num_patches = aggregated_features.shape[1]
        patch_grid_size = int(np.sqrt(num_patches))

        output = aggregated_features.reshape(
            batch_size, patch_grid_size, patch_grid_size, self.target_embed_dimension
        )

        if return_spatial_shape:
            return output, (patch_grid_size, patch_grid_size)

        return output

    def get_flatten_embeddings(self, images):
        """
        Return flattened patch embeddings (for backward compatibility).

        Returns:
            features: (Batch_Size * Num_Patches, Target_Embed_Dimension)
        """
        spatial_embeddings = self.forward(images, return_spatial_shape=False)
        flattened = spatial_embeddings.reshape(-1, self.target_embed_dimension)
        return flattened

    def get_image_level_features(self, images):
        """
        Return image-level features via global average pooling.
        Used for prototype-based routing.

        Returns:
            features: (Batch_Size, Target_Embed_Dimension)
        """
        spatial_embeddings = self.forward(images, return_spatial_shape=False)
        # Global average pooling: (B, H, W, D) -> (B, D)
        image_features = spatial_embeddings.mean(dim=(1, 2))
        return image_features
