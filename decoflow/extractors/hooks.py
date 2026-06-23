"""
Feature Hooking Helper Classes.

Provides utilities to extract intermediate layer outputs from neural networks.
"""

import torch
import torch.nn as nn


class LastLayerToExtractReachedException(Exception):
    """Exception to break forward pass after extracting last layer."""
    pass


class ForwardHook:
    """Hook to extract intermediate layer outputs."""

    def __init__(self, hook_dict, layer_name, last_layer_to_extract):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = (layer_name == last_layer_to_extract)

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()


class NetworkFeatureAggregator(nn.Module):
    """Extract features from specific layers of a backbone network."""

    def __init__(self, backbone, layers_to_extract_from):
        super(NetworkFeatureAggregator, self).__init__()
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.outputs = {}

        # Register hooks
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()

        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )
            # Navigate to layer based on timm/torchvision structure
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    network_layer = network_layer[int(extract_idx)]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                network_layer = backbone.__dict__["_modules"][extract_layer]

            if isinstance(network_layer, nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
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
        """Calculate channel dimensions for each extracted layer."""
        _input = torch.ones([1] + list(input_shape)).to(device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]
