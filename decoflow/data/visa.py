"""
VisA (Visual Anomaly) Dataset

Adapted for DeCoFlow from the original VisA dataset.
Dataset source: https://github.com/amazon-science/spot-diff
"""

import os
import csv
from typing import Any, Callable, Optional, Tuple
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


VISA_CLASS_NAMES = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(VISA_CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}


class VISA(Dataset):
    """VisA (Visual Anomaly) Dataset.

    Args:
        root: Root directory of dataset (e.g., '/Data/VISA').
        class_name: Name of the class to load (e.g., 'candle').
        train: If True, load training data, otherwise test data.
        transform: Transform to apply to images.
        target_transform: Transform to apply to masks.
        img_size: Size to resize images to.
        crp_size: Size to center crop images to.
        msk_size: Size to resize masks to.
        use_rotation_aug: If True, apply random rotation augmentation (training only).
        rotation_degrees: Range of rotation degrees (default: 180 for full rotation).
    """

    CLASS_NAMES = VISA_CLASS_NAMES

    def __init__(
            self,
            root: str,
            class_name: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            img_size: int = 518,
            crp_size: int = 518,
            msk_size: int = 256,
            use_rotation_aug: bool = False,
            rotation_degrees: float = 180.0,
            **kwargs):
        self.root = root
        self.class_name = class_name
        self.train = train
        self.img_size = img_size
        self.cropsize = [crp_size, crp_size]
        self.masksize = msk_size
        self.use_rotation_aug = use_rotation_aug and train
        self.rotation_degrees = rotation_degrees

        # Load dataset using CSV
        if self.class_name is None:
            self.image_paths, self.labels, self.mask_paths, self.img_types = self._load_all_data()
        else:
            self.image_paths, self.labels, self.mask_paths, self.img_types = self._load_data()

        # Set transforms
        self.transform = transform
        if transform is None:
            if self.use_rotation_aug:
                self.transform = T.Compose([
                    T.Resize(img_size, Image.LANCZOS),
                    T.RandomRotation(degrees=rotation_degrees, fill=0),
                    T.CenterCrop(crp_size),
                    T.ToTensor(),
                    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
                ])
                print(f"   [Augmentation] Random rotation enabled: +/-{rotation_degrees} deg")
            else:
                self.transform = T.Compose([
                    T.Resize(img_size, Image.LANCZOS),
                    T.CenterCrop(crp_size),
                    T.ToTensor(),
                    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
                ])

        self.target_transform = target_transform
        if target_transform is None:
            self.target_transform = T.Compose([
                T.Resize(self.masksize, Image.NEAREST),
                T.CenterCrop(self.masksize),
                T.ToTensor()
            ])

        self.class_to_idx = CLASS_TO_IDX.copy()
        self.idx_to_class = IDX_TO_CLASS.copy()

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, str, str]:
        """Get item by index.

        Returns:
            tuple: (image, label, mask, filename, img_type)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        mask_path = self.mask_paths[idx]
        img_type = self.img_types[idx]

        if self.class_name is None:
            # Extract class name from path: {class_name}/Data/Images/...
            class_name = image_path.split('/')[0]
        else:
            class_name = self.class_name

        # Load and process image
        full_image_path = os.path.join(self.root, image_path)
        image = Image.open(full_image_path).convert('RGB')
        image = self.transform(image)

        # Load mask
        if label == 0 or mask_path is None or mask_path == '':
            mask = torch.zeros([1, self.masksize, self.masksize])
        else:
            full_mask_path = os.path.join(self.root, mask_path)
            mask = Image.open(full_mask_path).convert('L')
            mask = self.target_transform(mask)

        # Use class index as label for training
        if self.train:
            label = CLASS_TO_IDX[class_name]

        filename = os.path.basename(image_path).rsplit('.', 1)[0]
        return image, label, mask, filename, img_type

    def __len__(self):
        return len(self.image_paths)

    def _load_data(self):
        """Load data for a single class using CSV file."""
        split = 'train' if self.train else 'test'
        image_paths, labels, mask_paths, types = [], [], [], []

        csv_path = os.path.join(self.root, 'split_csv', '1cls.csv')

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['object'] == self.class_name and row['split'] == split:
                    image_paths.append(row['image'])

                    if row['label'] == 'normal':
                        labels.append(0)
                        mask_paths.append(None)
                        types.append('good')
                    else:
                        labels.append(1)
                        mask_paths.append(row['mask'] if row['mask'] else None)
                        types.append('anomaly')

        return image_paths, labels, mask_paths, types

    def _load_all_data(self):
        """Load data for all classes."""
        all_image_paths = []
        all_labels = []
        all_mask_paths = []
        all_types = []

        original_class_name = self.class_name
        for class_name in self.CLASS_NAMES:
            self.class_name = class_name
            image_paths, labels, mask_paths, types = self._load_data()
            all_image_paths.extend(image_paths)
            all_labels.extend(labels)
            all_mask_paths.extend(mask_paths)
            all_types.extend(types)

        self.class_name = original_class_name
        return all_image_paths, all_labels, all_mask_paths, all_types

    def update_class_to_idx(self, class_to_idx):
        """Update class to index mapping."""
        for class_name in self.class_to_idx.keys():
            if class_name in class_to_idx:
                self.class_to_idx[class_name] = class_to_idx[class_name]
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
