"""
Dataset utilities for DeCoFlow continual learning.
"""

from typing import List, Dict, Any, Tuple
from torch.utils.data import Dataset

from decoflow.data.mvtec import MVTEC
from decoflow.data.visa import VISA
from decoflow.data.mpdd import MPDD
from decoflow.data.adnet import ADNet
from decoflow.data.realiad import RealIAD


# Dataset registry
DATASET_REGISTRY = {
    'mvtec': MVTEC,
    'visa': VISA,
    'mpdd': MPDD,
    'adnet': ADNet,
    'realiad': RealIAD,
}


def get_dataset_class(dataset_name: str):
    """Get dataset class by name.

    Args:
        dataset_name: Name of the dataset ('mvtec', 'visa', 'mpdd', 'adnet', 'realiad')

    Returns:
        Dataset class
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[dataset_name]


def get_class_names(dataset_name: str) -> List[str]:
    """Get available class names for a dataset.

    Args:
        dataset_name: Name of the dataset ('mvtec', 'visa', 'mpdd', 'adnet', 'realiad')

    Returns:
        List of class names
    """
    dataset_cls = get_dataset_class(dataset_name)
    return dataset_cls.CLASS_NAMES


class TaskDataset(Dataset):
    """Dataset wrapper for continual learning tasks."""

    def __init__(self, data: List[Tuple], class_names: List[str], class_to_idx: Dict[str, int]):
        """
        Args:
            data: List of (image, class_id, mask, name, path) tuples
            class_names: List of class names in this task
            class_to_idx: Mapping from class name to global class index
        """
        self.data = data
        self.class_names = class_names
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        return self.data[idx]


def create_task_dataset(
    args,
    task_classes: List[str],
    global_class_to_idx: Dict[str, int],
    train: bool = True,
    use_rotation_aug: bool = False,
    rotation_degrees: float = 180.0
) -> TaskDataset:
    """Create dataset for specific task classes.

    Args:
        args: Configuration object with data_path, img_size, msk_size, dataset attributes
        task_classes: List of class names for this task
        global_class_to_idx: Global mapping from class name to index
        train: If True, load training data, otherwise test data
        use_rotation_aug: If True, apply random rotation augmentation (training only)
        rotation_degrees: Range of rotation degrees

    Returns:
        TaskDataset containing data for the specified classes
    """
    filtered_data = []

    # Get dataset class based on args.dataset (default to mvtec for backward compatibility)
    dataset_name = getattr(args, 'dataset', 'mvtec')
    DatasetClass = get_dataset_class(dataset_name)

    for class_name in task_classes:
        # Create dataset for specific class
        class_dataset = DatasetClass(
            root=args.data_path,
            class_name=class_name,
            train=train,
            img_size=args.img_size,
            crp_size=args.img_size,
            msk_size=args.msk_size,
            use_rotation_aug=use_rotation_aug if train else False,
            rotation_degrees=rotation_degrees
        )

        # Add all data from this class
        for i in range(len(class_dataset)):
            image, label, mask, name, path = class_dataset[i]
            # Use global class ID
            final_class_id = int(global_class_to_idx[class_name])
            filtered_data.append((image, final_class_id, mask, name, path))

    # Create task-specific class mapping
    task_class_to_idx = {cls: global_class_to_idx[cls] for cls in task_classes}

    return TaskDataset(filtered_data, task_classes, task_class_to_idx)
