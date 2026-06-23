"""
ADNet dataset support.

The ADNet release is distributed as domain-level archives and is described as
using MVTec-style pixel annotations. This loader keeps the assumptions narrow:
it discovers class directories that contain train/test splits and reads normal
training data from train/good-like folders.
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


ADNET_DOMAIN_NAMES = [
    "Agrifood",
    "Electronics",
    "Industry",
    "Infrastructure",
    "Medical",
]

ADNET_CLASS_NAMES: List[str] = []

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
NORMAL_TYPE_NAMES = {"good", "normal", "ok", "healthy", "negative", "0"}
GT_DIR_NAMES = ("ground_truth", "groundtruth", "gt", "mask", "masks")


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _normalized_class_name(root: Path, class_dir: Path) -> str:
    return "__".join(class_dir.relative_to(root).parts)


def _find_child_dir(parent: Path, candidates: Tuple[str, ...]) -> Optional[Path]:
    if not parent.exists():
        return None
    lower_to_child = {child.name.lower(): child for child in parent.iterdir() if child.is_dir()}
    for name in candidates:
        child = lower_to_child.get(name.lower())
        if child is not None:
            return child
    return None


def discover_adnet_classes(
    root: str,
    domains: Optional[List[str]] = None,
    max_classes: Optional[int] = None,
) -> List[str]:
    """Discover ADNet class names from an extracted ADNet root.

    A class directory is any directory with both train and test subdirectories.
    Returned names are normalized relative paths joined with ``__`` so that
    domain/category class ids are shell- and CSV-friendly.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        return []

    domain_filter = {d.lower() for d in domains} if domains else None
    class_dirs: List[Path] = []
    for dirpath, dirnames, _ in os.walk(root_path):
        current = Path(dirpath)
        child_names = {name.lower() for name in dirnames}
        if "train" in child_names and "test" in child_names:
            if domain_filter:
                rel_parts = {part.lower() for part in current.relative_to(root_path).parts}
                if rel_parts.isdisjoint(domain_filter):
                    continue
            class_dirs.append(current)
            dirnames[:] = []

    class_names = sorted(_normalized_class_name(root_path, path) for path in class_dirs)
    if max_classes is not None:
        class_names = class_names[:max_classes]
    return class_names


class ADNet(Dataset):
    """ADNet dataset with MVTec-style class directories.

    Args:
        root: Extracted ADNet root.
        class_name: Normalized class name, e.g. ``Medical__brain``.
        train: If True, load normal training images; otherwise load test images.
    """

    CLASS_NAMES = ADNET_CLASS_NAMES

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
        self.root = Path(root).expanduser().resolve()
        self.class_name = class_name
        self.train = train
        self.img_size = img_size
        self.cropsize = [crp_size, crp_size]
        self.masksize = msk_size
        self.use_rotation_aug = use_rotation_aug and train
        self.rotation_degrees = rotation_degrees
        self._mask_index_cache: Dict[Path, Dict[str, Path]] = {}

        if self.class_name is None:
            self.class_names = discover_adnet_classes(str(self.root))
        else:
            self.class_names = [self.class_name]

        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

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

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, str, str]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        mask_path = self.mask_paths[idx]
        img_type = self.img_types[idx]
        class_name = self.sample_classes[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        if label == 0 or mask_path is None or not Path(mask_path).exists():
            mask = torch.zeros([1, self.masksize, self.masksize])
        else:
            mask = Image.open(mask_path).convert("L")
            mask = self.target_transform(mask)

        if self.train:
            label = self.class_to_idx[class_name]

        return image, label, mask, image_path.stem, img_type

    def __len__(self) -> int:
        return len(self.image_paths)

    def _resolve_class_dir(self, class_name: str) -> Path:
        candidates = [
            self.root / Path(*class_name.split("__")),
            self.root / class_name,
            self.root / class_name.replace("__", os.sep),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        matches = [
            path for path in self.root.rglob(class_name.split("__")[-1])
            if path.is_dir() and (path / "train").exists() and (path / "test").exists()
        ]
        if len(matches) == 1:
            return matches[0]

        raise FileNotFoundError(
            f"ADNet class '{class_name}' was not found under {self.root}. "
            "Run scripts/rebuttal/list_adnet_classes.py to inspect discovered names."
        )

    def _load_data(self):
        image_paths: List[Path] = []
        labels: List[int] = []
        mask_paths: List[Optional[Path]] = []
        img_types: List[str] = []
        sample_classes: List[str] = []

        for class_name in self.class_names:
            class_dir = self._resolve_class_dir(class_name)
            if self.train:
                self._load_train_class(class_name, class_dir, image_paths, labels, mask_paths, img_types, sample_classes)
            else:
                self._load_test_class(class_name, class_dir, image_paths, labels, mask_paths, img_types, sample_classes)

        return image_paths, labels, mask_paths, img_types, sample_classes

    def _load_train_class(self, class_name, class_dir, image_paths, labels, mask_paths, img_types, sample_classes):
        train_dir = _find_child_dir(class_dir, ("train",))
        if train_dir is None:
            return

        normal_dirs = [
            child for child in train_dir.iterdir()
            if child.is_dir() and child.name.lower() in NORMAL_TYPE_NAMES
        ]
        source_dirs = normal_dirs if normal_dirs else [train_dir]

        for source_dir in source_dirs:
            for image_path in sorted(path for path in source_dir.rglob("*") if _is_image(path)):
                image_paths.append(image_path)
                labels.append(0)
                mask_paths.append(None)
                img_types.append(source_dir.name if source_dir != train_dir else "good")
                sample_classes.append(class_name)

    def _load_test_class(self, class_name, class_dir, image_paths, labels, mask_paths, img_types, sample_classes):
        test_dir = _find_child_dir(class_dir, ("test",))
        if test_dir is None:
            return

        gt_root = _find_child_dir(class_dir, GT_DIR_NAMES)
        type_dirs = [child for child in sorted(test_dir.iterdir()) if child.is_dir()]
        if not type_dirs:
            type_dirs = [test_dir]

        for type_dir in type_dirs:
            img_type = type_dir.name if type_dir != test_dir else "unknown"
            is_normal = img_type.lower() in NORMAL_TYPE_NAMES
            for image_path in sorted(path for path in type_dir.rglob("*") if _is_image(path)):
                label = 0 if is_normal else 1
                image_paths.append(image_path)
                labels.append(label)
                mask_paths.append(None if label == 0 else self._find_mask(gt_root, img_type, image_path))
                img_types.append(img_type)
                sample_classes.append(class_name)

    def _find_mask(self, gt_root: Optional[Path], img_type: str, image_path: Path) -> Optional[Path]:
        if gt_root is None or not gt_root.exists():
            return None

        candidate_dirs = [
            gt_root / img_type,
            gt_root / img_type.lower(),
            gt_root,
        ]
        stems = [image_path.stem, f"{image_path.stem}_mask"]
        for candidate_dir in candidate_dirs:
            if not candidate_dir.exists():
                continue
            for stem in stems:
                for ext in IMAGE_EXTENSIONS:
                    candidate = candidate_dir / f"{stem}{ext}"
                    if candidate.exists():
                        return candidate

        index = self._mask_index_cache.get(gt_root)
        if index is None:
            index = {}
            for path in gt_root.rglob("*"):
                if not _is_image(path):
                    continue
                index[path.stem] = path
                if path.stem.endswith("_mask"):
                    index[path.stem[:-5]] = path
            self._mask_index_cache[gt_root] = index
        return index.get(image_path.stem)

    def update_class_to_idx(self, class_to_idx):
        self.class_to_idx.update({name: class_to_idx[name] for name in self.class_to_idx if name in class_to_idx})
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
