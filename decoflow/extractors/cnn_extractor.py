"""
PatchCore Feature Extractor (CNN-based).

Extract patch embeddings from CNN backbones (e.g., ResNet, WideResNet, EfficientNet).
Provides the same interface as ViTPatchCoreExtractor for seamless backbone switching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from decoflow.extractors.hooks import NetworkFeatureAggregator
from decoflow.extractors.patch import PatchMaker
from decoflow.extractors.preprocessing import Preprocessing


# Common CNN backbone layer configurations
CNN_LAYER_CONFIGS = {
    # ResNet family
    'resnet18': ('layer2', 'layer3'),
    'resnet34': ('layer2', 'layer3'),
    'resnet50': ('layer2', 'layer3'),
    'resnet101': ('layer2', 'layer3'),
    'resnet152': ('layer2', 'layer3'),
    # Wide ResNet
    'wide_resnet50_2': ('layer2', 'layer3'),
    'wide_resnet101_2': ('layer2', 'layer3'),
    # EfficientNet family (uses numbered blocks)
    'efficientnet_b0': ('blocks.2', 'blocks.4'),
    'efficientnet_b1': ('blocks.2', 'blocks.4'),
    'efficientnet_b2': ('blocks.2', 'blocks.4'),
    'efficientnet_b3': ('blocks.2', 'blocks.4'),
    'efficientnet_b4': ('blocks.2', 'blocks.4'),
    'efficientnet_b5': ('blocks.2', 'blocks.4'),
    'efficientnet_b6': ('blocks.2', 'blocks.4'),
    'efficientnet_b7': ('blocks.2', 'blocks.4'),
    'tf_efficientnet_b0': ('blocks.2', 'blocks.4'),
    'tf_efficientnet_b1': ('blocks.2', 'blocks.4'),
    'tf_efficientnet_b2': ('blocks.2', 'blocks.4'),
    'tf_efficientnet_b3': ('blocks.2', 'blocks.4'),
    'tf_efficientnet_b4': ('blocks.2', 'blocks.4'),
    'tf_efficientnet_b5': ('blocks.2', 'blocks.4'),
    'tf_efficientnet_b6': ('blocks.2', 'blocks.4'),
    'tf_efficientnet_b7': ('blocks.2', 'blocks.4'),
    # ConvNeXt family
    'convnext_tiny': ('stages.1', 'stages.2'),
    'convnext_small': ('stages.1', 'stages.2'),
    'convnext_base': ('stages.1', 'stages.2'),
    'convnext_large': ('stages.1', 'stages.2'),
    # RegNet family
    'regnetx_002': ('s2', 's3'),
    'regnetx_004': ('s2', 's3'),
    'regnetx_006': ('s2', 's3'),
    'regnetx_008': ('s2', 's3'),
    'regnetx_016': ('s2', 's3'),
    'regnetx_032': ('s2', 's3'),
    'regnety_002': ('s2', 's3'),
    'regnety_004': ('s2', 's3'),
    'regnety_006': ('s2', 's3'),
    'regnety_008': ('s2', 's3'),
    'regnety_016': ('s2', 's3'),
    'regnety_032': ('s2', 's3'),
}


def get_cnn_layers(backbone_name: str, use_high_res: bool = False) -> tuple:
    """
    Get appropriate layer names for a CNN backbone.

    Args:
        backbone_name: Name of the backbone
        use_high_res: If True, include layer1 for 56×56 resolution

    Returns:
        Tuple of layer names to extract
    """
    # Check exact match first
    if backbone_name in CNN_LAYER_CONFIGS:
        base_layers = CNN_LAYER_CONFIGS[backbone_name]
    else:
        # Check prefix matches
        backbone_lower = backbone_name.lower()
        found = False
        for key in CNN_LAYER_CONFIGS:
            if backbone_lower.startswith(key):
                base_layers = CNN_LAYER_CONFIGS[key]
                found = True
                break

        if not found:
            # Default for ResNet-like architectures
            if 'resnet' in backbone_lower or 'resnext' in backbone_lower:
                base_layers = ('layer2', 'layer3')
            elif 'efficientnet' in backbone_lower:
                base_layers = ('blocks.2', 'blocks.4')
            elif 'convnext' in backbone_lower:
                base_layers = ('stages.1', 'stages.2')
            elif 'regnet' in backbone_lower:
                base_layers = ('s2', 's3')
            else:
                # Fallback
                base_layers = ('layer2', 'layer3')

    # Add high-resolution layer if requested
    if use_high_res:
        # For ResNet-like: add layer1 (56×56)
        if 'layer2' in base_layers[0]:
            return ('layer1', 'layer2', 'layer3')
        # For EfficientNet: add blocks.1
        elif 'blocks.2' in base_layers[0]:
            return ('blocks.1', 'blocks.2', 'blocks.4')
        # For ConvNeXt: add stages.0
        elif 'stages.1' in base_layers[0]:
            return ('stages.0', 'stages.1', 'stages.2')
        # For RegNet: add s1
        elif 's2' in base_layers[0]:
            return ('s1', 's2', 's3')
        else:
            # Can't determine, just return base
            return base_layers

    return base_layers


class CNNPatchCoreExtractor(nn.Module):
    """
    CNN-based PatchCore Feature Extractor.

    Provides the same interface as ViTPatchCoreExtractor:
    - forward(images, return_spatial_shape=False) -> (B, H, W, D) or (B, H, W, D), (H, W)
    - get_image_level_features(images) -> (B, D)
    - get_flatten_embeddings(images) -> (B*H*W, D)

    Supports various CNN backbones from timm:
    - ResNet family: resnet18, resnet50, resnet101, etc.
    - Wide ResNet: wide_resnet50_2, wide_resnet101_2
    - EfficientNet: efficientnet_b0 ~ b7, tf_efficientnet_b0 ~ b7
    - ConvNeXt: convnext_tiny, convnext_small, convnext_base, convnext_large
    - RegNet: regnetx_*, regnety_*
    """

    def __init__(self,
                 backbone_name="wide_resnet50_2",
                 layers=None,
                 input_shape=(3, 224, 224),
                 patch_size=3,
                 patch_stride=1,
                 target_embed_dimension=1024,
                 device='cuda',
                 use_high_res=False):
        super(CNNPatchCoreExtractor, self).__init__()

        self.device = device
        self.backbone_name = backbone_name
        self.use_high_res = use_high_res

        # Auto-detect layers if not provided
        if layers is None:
            layers = get_cnn_layers(backbone_name, use_high_res=use_high_res)
        self.layers_to_extract_from = layers

        # Load backbone
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=False)
        self.backbone.to(device)
        self.backbone.eval()

        # Setup feature aggregator
        self.feature_aggregator = NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from
        )

        # Calculate feature dimensions
        feature_dims = self.feature_aggregator.feature_dimensions(input_shape, device)

        # Setup patch maker
        self.patch_maker = PatchMaker(patch_size, stride=patch_stride)

        # Setup preprocessing (channel reduction)
        self.preprocessing = Preprocessing(feature_dims, target_embed_dimension)
        self.preprocessing.to(device)

        self.target_embed_dimension = target_embed_dimension

        # Calculate spatial shape for given input
        self._cached_spatial_shape = self._compute_spatial_shape(input_shape, device)

    def _compute_spatial_shape(self, input_shape, device):
        """Compute the output spatial shape for a given input shape."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape).to(device)
            features_dict = self.feature_aggregator(dummy_input)
            first_layer = self.layers_to_extract_from[0]
            feat = features_dict[first_layer]
            # After patchify
            _, (h, w) = self.patch_maker.patchify(feat, return_spatial_info=True)
            return (h, w)

    def _extract_features(self, images):
        """Internal method to extract and process features."""
        images = images.to(self.device)
        batch_size = images.shape[0]

        # 1. Backbone feature extraction
        features_dict = self.feature_aggregator(images)
        features = [features_dict[layer] for layer in self.layers_to_extract_from]

        # 2. Patchify (unfold operation)
        features_info = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features_info]
        features = [x[0] for x in features_info]

        # 3. Align feature maps via interpolation
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )

            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape

            _features = _features.reshape(-1, *_features.shape[-2:])

            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)

            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features

        # 4. Preprocessing and aggregation
        # Shape: (B * H * W, patch_size, patch_size, D)
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        features = self.preprocessing(features)  # (B*H*W, patch_size*patch_size, D)
        features = features.mean(dim=1)  # (B*H*W, D)

        return features, ref_num_patches, batch_size

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
        features, ref_num_patches, batch_size = self._extract_features(images)

        # Reshape to spatial format (B, H, W, D)
        H, W = ref_num_patches
        features = features.reshape(batch_size, H, W, self.target_embed_dimension)

        if return_spatial_shape:
            return features, (H, W)

        return features

    def get_flatten_embeddings(self, images):
        """
        Return flattened patch embeddings (for backward compatibility).

        Returns:
            features: (Batch_Size * Num_Patches, Target_Embed_Dimension)
        """
        features, _, _ = self._extract_features(images)
        return features

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


# Backward compatibility alias
PatchCoreExtractor = CNNPatchCoreExtractor
