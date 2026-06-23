"""Feature extraction modules for DeCoFlow."""

from decoflow.extractors.hooks import (
    LastLayerToExtractReachedException,
    ForwardHook,
    NetworkFeatureAggregator
)
from decoflow.extractors.patch import PatchMaker
from decoflow.extractors.preprocessing import MeanMapper, Preprocessing
from decoflow.extractors.cnn_extractor import PatchCoreExtractor, CNNPatchCoreExtractor, get_cnn_layers
from decoflow.extractors.vit_extractor import ViTFeatureAggregator, ViTPatchCoreExtractor


# ============================================================================
# Backbone type detection and unified factory
# ============================================================================

# Known ViT backbone prefixes
VIT_PREFIXES = (
    'vit_', 'deit_', 'dinov2', 'beit_', 'eva_', 'swin_',
    'maxvit_', 'coatnet_', 'flexivit_'
)

# Known CNN backbone prefixes
CNN_PREFIXES = (
    'resnet', 'resnext', 'wide_resnet', 'efficientnet', 'tf_efficientnet',
    'convnext', 'regnet', 'densenet', 'mobilenet', 'mnasnet',
    'shufflenet', 'inception', 'xception', 'dpn', 'darknet',
    'cspnet', 'hrnet', 'res2net', 'rexnet', 'ghostnet', 'hardcorenas',
    'dla', 'selecsls', 'vgg', 'squeezenet'
)


def is_vit_backbone(backbone_name: str) -> bool:
    """Check if the backbone is a Vision Transformer variant."""
    backbone_lower = backbone_name.lower()
    return any(backbone_lower.startswith(prefix) for prefix in VIT_PREFIXES)


def is_cnn_backbone(backbone_name: str) -> bool:
    """Check if the backbone is a CNN variant."""
    backbone_lower = backbone_name.lower()
    return any(backbone_lower.startswith(prefix) for prefix in CNN_PREFIXES)


def get_backbone_type(backbone_name: str) -> str:
    """
    Determine the backbone type ('vit' or 'cnn').

    Args:
        backbone_name: Name of the timm backbone model

    Returns:
        'vit' or 'cnn'
    """
    if is_vit_backbone(backbone_name):
        return 'vit'
    elif is_cnn_backbone(backbone_name):
        return 'cnn'
    else:
        # Try to infer from timm model
        try:
            import timm
            model = timm.create_model(backbone_name, pretrained=False)
            # Check for ViT-like attributes
            if hasattr(model, 'blocks') and hasattr(model, 'patch_embed'):
                del model
                return 'vit'
            # Check for CNN-like attributes
            if hasattr(model, 'layer1') or hasattr(model, 'features') or hasattr(model, 'stages'):
                del model
                return 'cnn'
            del model
        except Exception:
            pass
        # Default to CNN
        return 'cnn'


def create_feature_extractor(
    backbone_name: str,
    input_shape: tuple = (3, 224, 224),
    target_embed_dimension: int = 768,
    device: str = 'cuda',
    # ViT-specific options
    blocks_to_extract: list = None,
    remove_cls_token: bool = True,
    # CNN-specific options
    layers: tuple = None,
    patch_size: int = 3,
    patch_stride: int = 1,
    use_high_res: bool = False,
):
    """
    Create a feature extractor based on the backbone type.

    This factory function automatically detects whether the backbone is a
    ViT or CNN and creates the appropriate extractor with a unified interface.

    Args:
        backbone_name: Name of the timm backbone model
        input_shape: Input image shape (C, H, W)
        target_embed_dimension: Target embedding dimension
        device: Device to use ('cuda' or 'cpu')

        # ViT-specific options:
        blocks_to_extract: List of transformer block indices to extract (default: [1, 3, 5, 11])
        remove_cls_token: Whether to remove CLS token from ViT output

        # CNN-specific options:
        layers: Tuple of layer names to extract from (auto-detected if None)
        patch_size: Patch size for CNN patchification
        patch_stride: Stride for CNN patchification
        use_high_res: Use high-resolution features (56×56 from layer1 for CNNs)

    Returns:
        A feature extractor with the following interface:
        - forward(images, return_spatial_shape=False) -> (B, H, W, D) or ((B, H, W, D), (H, W))
        - get_image_level_features(images) -> (B, D)
        - get_flatten_embeddings(images) -> (B*H*W, D)
    """
    backbone_type = get_backbone_type(backbone_name)

    if backbone_type == 'vit':
        if blocks_to_extract is None:
            blocks_to_extract = [1, 3, 5, 11]

        return ViTPatchCoreExtractor(
            backbone_name=backbone_name,
            blocks_to_extract=blocks_to_extract,
            input_shape=input_shape,
            target_embed_dimension=target_embed_dimension,
            device=device,
            remove_cls_token=remove_cls_token
        )
    else:  # CNN
        return CNNPatchCoreExtractor(
            backbone_name=backbone_name,
            layers=layers,
            input_shape=input_shape,
            patch_size=patch_size,
            patch_stride=patch_stride,
            target_embed_dimension=target_embed_dimension,
            device=device,
            use_high_res=use_high_res
        )


__all__ = [
    # Hooks
    "LastLayerToExtractReachedException",
    "ForwardHook",
    "NetworkFeatureAggregator",
    # Patch processing
    "PatchMaker",
    "MeanMapper",
    "Preprocessing",
    # CNN Extractor
    "PatchCoreExtractor",
    "CNNPatchCoreExtractor",
    "get_cnn_layers",
    # ViT Extractor
    "ViTFeatureAggregator",
    "ViTPatchCoreExtractor",
    # Unified factory
    "create_feature_extractor",
    "get_backbone_type",
    "is_vit_backbone",
    "is_cnn_backbone",
]
