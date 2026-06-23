"""
DeCoFlow Data Module

Contains dataset classes and utilities for loading data.
"""

from decoflow.data.mvtec import MVTEC, MVTEC_CLASS_NAMES
from decoflow.data.visa import VISA, VISA_CLASS_NAMES
from decoflow.data.mpdd import MPDD, MPDD_CLASS_NAMES
from decoflow.data.adnet import ADNet, ADNET_CLASS_NAMES, discover_adnet_classes
from decoflow.data.realiad import RealIAD, REALIAD_CLASS_NAMES
from decoflow.data.datasets import (
    create_task_dataset,
    TaskDataset,
    get_dataset_class,
    get_class_names,
    DATASET_REGISTRY,
)

__all__ = [
    # MVTec
    "MVTEC",
    "MVTEC_CLASS_NAMES",
    # VisA
    "VISA",
    "VISA_CLASS_NAMES",
    # MPDD
    "MPDD",
    "MPDD_CLASS_NAMES",
    # ADNet
    "ADNet",
    "ADNET_CLASS_NAMES",
    "discover_adnet_classes",
    # Real-IAD
    "RealIAD",
    "REALIAD_CLASS_NAMES",
    # Dataset utilities
    "create_task_dataset",
    "TaskDataset",
    "get_dataset_class",
    "get_class_names",
    "DATASET_REGISTRY",
]
