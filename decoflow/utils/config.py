"""
Configuration utilities for DeCoFlow.

Provides a simple configuration system independent of NFCAD.
"""

import os
import yaml
from typing import Any, Dict, Optional
from easydict import EasyDict


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def get_default_config() -> Dict[str, Any]:
    """Load default configuration."""
    config_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(config_dir, 'configs', 'defaults.yaml')

    if os.path.exists(config_path):
        return load_yaml(config_path)

    # Fallback defaults if config file not found
    return {
        'seed': 0,
        'gpu': '0',
        'dataset': 'mvtec',
        'data_path': '/Data/MVTecAD',
        'img_size': 224,
        'msk_size': 256,
        'lr': 0.0001,
        'lr_decay_epochs': [48, 57, 88],
        'lr_decay_rate': 0.1,
        'lr_warm': True,
        'lr_warm_epochs': 2,
        'lr_cosine': True,
        'meta_epochs': 4,
        'sub_epochs': 3,
        'batch_size': 16,
        'backbone_name': 'vit_base_patch16_224.augreg2_in21k_ft_in1k',
        'coupling_layers': 8,
        'clamp_alpha': 1.9,
        'lora_rank': 32,
        'lora_alpha': 1.0,
        'ewc_lambda': 500.0,
        'slow_lr_ratio': 0.2,
        'slow_blocks_k': 2,
    }


def get_config(
    img_size: Optional[int] = None,
    data_path: Optional[str] = None,
    **overrides
) -> EasyDict:
    """Get configuration with optional overrides.

    Args:
        img_size: Override for image size
        data_path: Override for data path
        **overrides: Additional configuration overrides

    Returns:
        EasyDict configuration object
    """
    config = get_default_config()

    # Apply overrides
    if img_size is not None:
        config['img_size'] = img_size
    if data_path is not None:
        config['data_path'] = data_path

    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    return EasyDict(config)
