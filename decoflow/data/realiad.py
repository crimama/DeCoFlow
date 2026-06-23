"""
Real-IAD dataset support.

Real-IAD archives are organized by object class with OK and NG folders:
``class/OK/Sxxxx/*.jpg`` and ``class/NG/<defect>/Sxxxx/*.jpg`` plus same-stem
``.png`` masks. The release does not provide a train/test split, so this loader
uses a deterministic OK sample split: train uses the first 80% of OK sample
folders, and test uses the held-out OK folders plus all NG samples.
"""

import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


REALIAD_CLASS_NAMES = [
    "audiojack", "bottle_cap", "button_battery", "end_cap", "eraser",
    "fire_hood", "mint", "mounts", "pcb", "phone_battery",
    "plastic_nut", "plastic_plug", "porcelain_doll", "regulator",
    "rolled_strip_base", "sim_card_set", "switch", "tape",
    "terminalblock", "toothbrush", "toy", "toy_brick", "transistor1",
    "u_block", "usb", "usb_adaptor", "vcpill", "wooden_beads",
    "woodstick", "zipper",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
CLASS_TO_IDX = {name: idx for idx, name in enumerate(REALIAD_CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}


class RealIAD(Dataset):
    """Real-IAD object-level anomaly detection dataset."""

    CLASS_NAMES = REALIAD_CLASS_NAMES

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
            ok_train_ratio: float = 0.8,
            **kwargs):
        self.root = Path(root).expanduser().resolve()
        self.class_name = class_name
        self.train = train
        self.img_size = img_size
        self.cropsize = [crp_size, crp_size]
        self.masksize = msk_size
        self.use_rotation_aug = use_rotation_aug and train
        self.rotation_degrees = rotation_degrees
        self.ok_train_ratio = ok_train_ratio

        if self.class_name is None:
            self.class_names = list(REALIAD_CLASS_NAMES)
        else:
            self.class_names = [self.class_name]

        self.image_paths, self.labels, self.mask_paths, self.img_types, self.sample_classes = self._load_data()

        self.transform = transform
        if transform is None:
            if self.use_rotation_aug:
                self.transform = T.Compose([
                    T.Resize(img_size, Image.LANCZOS),
                    T.RandomRotation(degrees=(-rotation_degrees, rotation_degrees), fill=0),
                    T.CenterCrop(crp_size),
                    T.ToTensor(),
                    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ])
            else:
                self.transform = T.Compose([
                    T.Resize(img_size, Image.LANCZOS),
                    T.CenterCrop(crp_size),
                    T.ToTensor(),
                    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ])

        self.target_transform = target_transform
        if target_transform is None:
            self.target_transform = T.Compose([
                T.Resize(self.masksize, Image.NEAREST),
                T.CenterCrop(self.masksize),
                T.ToTensor(),
            ])

        self.class_to_idx = CLASS_TO_IDX.copy()
        self.idx_to_class = IDX_TO_CLASS.copy()

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, str, str]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        mask_path = self.mask_paths[idx]
        img_type = self.img_types[idx]
        class_name = self.sample_classes[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        if label == 0 or mask_path is None or not mask_path.exists():
            mask = torch.zeros([1, self.masksize, self.masksize])
        else:
            mask = Image.open(mask_path).convert("L")
            mask = self.target_transform(mask)

        if self.train:
            label = CLASS_TO_IDX[class_name]

        return image, label, mask, image_path.stem, img_type

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_data(self):
        image_paths: List[Path] = []
        labels: List[int] = []
        mask_paths: List[Optional[Path]] = []
        img_types: List[str] = []
        sample_classes: List[str] = []

        for class_name in self.class_names:
            class_dir = self._resolve_class_dir(class_name)
            ok_samples = self._list_sample_dirs(class_dir / "OK")
            train_samples, test_samples = self._split_ok_samples(ok_samples)
            selected_ok = train_samples if self.train else test_samples
            for sample_dir in selected_ok:
                for image_path in self._list_images(sample_dir):
                    image_paths.append(image_path)
                    labels.append(0)
                    mask_paths.append(None)
                    img_types.append("OK")
                    sample_classes.append(class_name)

            if not self.train:
                ng_root = class_dir / "NG"
                for defect_dir in sorted(path for path in ng_root.iterdir() if path.is_dir()) if ng_root.exists() else []:
                    for sample_dir in self._list_sample_dirs(defect_dir):
                        for image_path in self._list_images(sample_dir):
                            mask_path = image_path.with_suffix(".png")
                            label = 1 if mask_path.exists() else 0
                            image_paths.append(image_path)
                            labels.append(label)
                            mask_paths.append(mask_path if label == 1 else None)
                            img_types.append(defect_dir.name)
                            sample_classes.append(class_name)

        return image_paths, labels, mask_paths, img_types, sample_classes

    def _resolve_class_dir(self, class_name: str) -> Path:
        candidates = [
            self.root / class_name,
            self.root / "realiad_256" / class_name,
            self.root / "realiad_512" / class_name,
            self.root / "realiad_1024" / class_name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Real-IAD class '{class_name}' was not found under {self.root}")

    def _split_ok_samples(self, samples: List[Path]) -> Tuple[List[Path], List[Path]]:
        if len(samples) <= 1:
            return samples, samples
        split_idx = int(len(samples) * self.ok_train_ratio)
        split_idx = min(max(split_idx, 1), len(samples) - 1)
        return samples[:split_idx], samples[split_idx:]

    @staticmethod
    def _list_sample_dirs(root: Path) -> List[Path]:
        if not root.exists():
            return []
        sample_dirs = [path for path in root.iterdir() if path.is_dir()]
        if sample_dirs:
            return sorted(sample_dirs)
        return [root]

    @staticmethod
    def _list_images(root: Path) -> List[Path]:
        return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)

    def update_class_to_idx(self, class_to_idx):
        for class_name in self.class_to_idx.keys():
            if class_name in class_to_idx:
                self.class_to_idx[class_name] = class_to_idx[class_name]
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
