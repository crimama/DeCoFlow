"""
DeCoFlow: Structural Decomposition for Continual Anomaly Detection

This package provides modular components for continual learning based
anomaly detection using Normalizing Flows with LoRA adapters.

Ablation Support:
- AblationConfig: Configuration class for ablation studies
- ABLATION_PRESETS: Predefined ablation configurations
- get_ablation_config: Factory function for ablation configs
"""

from decoflow.models.decoflow_nf import DeCoFlowNF
from decoflow.trainer.continual_trainer import DeCoFlowContinualTrainer
from decoflow.extractors.vit_extractor import ViTPatchCoreExtractor
from decoflow.extractors.cnn_extractor import CNNPatchCoreExtractor, PatchCoreExtractor
from decoflow.extractors import (
    create_feature_extractor,
    get_backbone_type,
    is_vit_backbone,
    is_cnn_backbone,
)
from decoflow.models.position_embedding import PositionalEmbeddingGenerator
from decoflow.utils.logger import TrainingLogger, setup_training_logger
from decoflow.utils.helpers import init_seeds, setting_lr_parameters
from decoflow.utils.config import get_config, get_default_config
from decoflow.evaluation.evaluator import (
    evaluate_class,
    evaluate_all_tasks,
    evaluate_routing_performance
)
from decoflow.config.ablation import (
    AblationConfig,
    ABLATION_PRESETS,
    get_ablation_config,
    add_ablation_args,
    parse_ablation_args,
)
from decoflow.data import (
    MVTEC, MVTEC_CLASS_NAMES,
    VISA, VISA_CLASS_NAMES,
    MPDD, MPDD_CLASS_NAMES,
    ADNet, ADNET_CLASS_NAMES, discover_adnet_classes,
    RealIAD, REALIAD_CLASS_NAMES,
    create_task_dataset, TaskDataset,
    get_dataset_class, get_class_names, DATASET_REGISTRY,
)
from decoflow.utils.diagnostics import FlowDiagnostics, run_diagnostics_on_model

__version__ = "0.1.0"
__all__ = [
    # Core components
    "DeCoFlowNF",
    "DeCoFlowContinualTrainer",
    # Feature extractors
    "ViTPatchCoreExtractor",
    "CNNPatchCoreExtractor",
    "PatchCoreExtractor",
    "create_feature_extractor",
    "get_backbone_type",
    "is_vit_backbone",
    "is_cnn_backbone",
    # Position embedding
    "PositionalEmbeddingGenerator",
    # Logging
    "TrainingLogger",
    "setup_training_logger",
    # Utilities
    "init_seeds",
    "setting_lr_parameters",
    "get_config",
    "get_default_config",
    # Evaluation
    "evaluate_class",
    "evaluate_all_tasks",
    "evaluate_routing_performance",
    # Diagnostics
    "FlowDiagnostics",
    "run_diagnostics_on_model",
    # Ablation
    "AblationConfig",
    "ABLATION_PRESETS",
    "get_ablation_config",
    "add_ablation_args",
    "parse_ablation_args",
    # Data
    "MVTEC",
    "MVTEC_CLASS_NAMES",
    "VISA",
    "VISA_CLASS_NAMES",
    "MPDD",
    "MPDD_CLASS_NAMES",
    "ADNet",
    "ADNET_CLASS_NAMES",
    "discover_adnet_classes",
    "RealIAD",
    "REALIAD_CLASS_NAMES",
    "create_task_dataset",
    "TaskDataset",
    "get_dataset_class",
    "get_class_names",
    "DATASET_REGISTRY",
]
