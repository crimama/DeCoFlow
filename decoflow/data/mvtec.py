"""
MVTec AD Dataset

Adapted for DeCoFlow from NFCAD implementation.
"""

import os
from typing import Any, Callable, Optional, Tuple
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


MVTEC_CLASS_NAMES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_TO_IDX = {
    'bottle': 0, 'cable': 1, 'capsule': 2, 'carpet': 3,
    'grid': 4, 'hazelnut': 5, 'leather': 6, 'metal_nut': 7,
    'pill': 8, 'screw': 9, 'tile': 10, 'toothbrush': 11,
    'transistor': 12, 'wood': 13, 'zipper': 14
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


class MVTEC(Dataset):
    """MVTec Anomaly Detection Dataset.

    Args:
        root: Root directory of dataset (e.g., '/Data/MVTecAD').
        class_name: Name of the class to load (e.g., 'bottle').
        train: If True, load training data, otherwise test data.
        transform: Transform to apply to images.
        target_transform: Transform to apply to masks.
        img_size: Size to resize images to.
        crp_size: Size to center crop images to.
        msk_size: Size to resize masks to.
        use_rotation_aug: If True, apply random rotation augmentation (training only).
        rotation_degrees: Range of rotation degrees (default: 180 for full rotation).
    """

    CLASS_NAMES = MVTEC_CLASS_NAMES

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
        self.use_rotation_aug = use_rotation_aug and train  # Only for training
        self.rotation_degrees = rotation_degrees

        # Load dataset
        if self.class_name is None:
            self.image_paths, self.labels, self.mask_paths, self.img_types = self._load_all_data()
        else:
            self.image_paths, self.labels, self.mask_paths, self.img_types = self._load_data()

        # Set transforms
        self.transform = transform
        if transform is None:
            if self.use_rotation_aug:
                # With rotation augmentation (training only)
                self.transform = T.Compose([
                    T.Resize(img_size, Image.LANCZOS),
                    T.RandomRotation(degrees=(-rotation_degrees, rotation_degrees), fill=0),
                    T.CenterCrop(crp_size),
                    T.ToTensor(),
                    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
                ])
                print(f"   🔄 [Augmentation] Random rotation enabled: ±{rotation_degrees}°")
            else:
                # Standard transform (no augmentation)
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
            class_name = image_path.split('/')[-4]
        else:
            class_name = self.class_name

        # Load and process image
        image = Image.open(image_path)
        if class_name in ['zipper', 'screw', 'grid']:
            # Handle greyscale classes
            image = np.expand_dims(np.array(image), axis=2)
            image = np.concatenate([image, image, image], axis=2)
            image = Image.fromarray(image.astype('uint8')).convert('RGB')
        image = self.transform(image)

        # Load mask
        if label == 0:
            mask = torch.zeros([1, self.masksize, self.masksize])
        else:
            mask = Image.open(mask_path)
            mask = self.target_transform(mask)

        # Use class index as label for training
        if self.train:
            label = CLASS_TO_IDX[class_name]

        return image, label, mask, os.path.basename(image_path[:-4]), img_type

    def __len__(self):
        return len(self.image_paths)

    def _load_data(self):
        """Load data for a single class."""
        phase = 'train' if self.train else 'test'
        image_paths, labels, mask_paths, types = [], [], [], []

        image_dir = os.path.join(self.root, self.class_name, phase)
        mask_dir = os.path.join(self.root, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(image_dir))
        for img_type in img_types:
            img_type_dir = os.path.join(image_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue

            img_fpath_list = sorted([
                os.path.join(img_type_dir, f)
                for f in os.listdir(img_type_dir)
                if f.endswith('.png')
            ])
            image_paths.extend(img_fpath_list)

            if img_type == 'good':
                labels.extend([0] * len(img_fpath_list))
                mask_paths.extend([None] * len(img_fpath_list))
                types.extend(['good'] * len(img_fpath_list))
            else:
                labels.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(mask_dir, img_type)
                img_fname_list = [
                    os.path.splitext(os.path.basename(f))[0]
                    for f in img_fpath_list
                ]
                gt_fpath_list = [
                    os.path.join(gt_type_dir, img_fname + '_mask.png')
                    for img_fname in img_fname_list
                ]
                mask_paths.extend(gt_fpath_list)
                types.extend([img_type] * len(img_fpath_list))

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
