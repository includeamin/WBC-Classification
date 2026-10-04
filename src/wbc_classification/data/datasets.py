"""Dataset listing, splitting, loading and DataLoader construction."""

import logging
import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from wbc_classification.config import Config
from wbc_classification.data.labels import discover_classes, is_image_file, require_split_dir
from wbc_classification.data.transforms import build_transforms
from wbc_classification.errors import DataError, SegmentationError
from wbc_classification.preprocessing.segmentation import crop_cell

logger = logging.getLogger(__name__)

Sample = tuple[Path, int]


def list_samples(split_dir: Path, class_names: Sequence[str]) -> list[Sample]:
    samples: list[Sample] = []
    for index, name in enumerate(class_names):
        class_dir = split_dir / name
        if not class_dir.is_dir():
            raise DataError(f"Missing class folder {class_dir}")
        files = sorted(p for p in class_dir.iterdir() if is_image_file(p))
        if not files:
            raise DataError(f"Class folder {class_dir} contains no images")
        samples.extend((p, index) for p in files)
    return samples


def stratified_split(
    samples: list[Sample], val_fraction: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    labels = [label for _, label in samples]
    counts = Counter(labels)
    n_classes = len(counts)
    n_val = math.ceil(val_fraction * len(samples))
    n_train = len(samples) - n_val
    if min(counts.values()) < 2 or n_val < n_classes or n_train < n_classes:
        raise DataError(
            f"Not enough images to make a stratified validation split "
            f"({len(samples)} images, {n_classes} classes, val_fraction={val_fraction}); "
            "add more images or raise data.val_fraction."
        )
    train, val = train_test_split(
        samples, test_size=val_fraction, stratify=labels, random_state=seed
    )
    return train, val


def load_image(path: Path, crop: bool = False) -> Image.Image:
    """Load an image as RGB, optionally cropped to the cell (full image if none found)."""
    try:
        with Image.open(path) as handle:
            image = handle.convert("RGB")
    except OSError as exc:
        raise DataError(f"Cannot read image {path}: {exc}") from exc
    if crop:
        try:
            image = Image.fromarray(crop_cell(np.asarray(image)))
        except SegmentationError as exc:
            logger.warning("No cell found in %s (%s); using the full image", path, exc)
    return image


class WBCDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        samples: list[Sample],
        transform: Callable[[Image.Image], torch.Tensor],
        crop: bool = False,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.crop = crop

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        return self.transform(load_image(path, self.crop)), label


@dataclass(frozen=True)
class Loaders:
    train: DataLoader
    val: DataLoader
    class_names: list[str]


def build_train_val_loaders(cfg: Config) -> Loaders:
    train_dir = cfg.data.root / "TRAIN"
    class_names = discover_classes(train_dir)
    train_samples, val_samples = stratified_split(
        list_samples(train_dir, class_names), cfg.data.val_fraction, cfg.train.seed
    )
    crop = cfg.preprocessing.crop
    train_ds = WBCDataset(train_samples, build_transforms(cfg.data.image_size, True), crop)
    val_ds = WBCDataset(val_samples, build_transforms(cfg.data.image_size, False), crop)
    batch_size = cfg.data.batch_size
    num_workers = cfg.data.num_workers
    pin_memory = torch.cuda.is_available()
    generator = torch.Generator().manual_seed(cfg.train.seed)
    return Loaders(
        train=DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        val=DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        class_names=class_names,
    )


def build_test_loader(cfg: Config, class_names: Sequence[str]) -> DataLoader:
    test_dir = cfg.data.root / "TEST"
    require_split_dir(test_dir)
    samples = list_samples(test_dir, class_names)
    dataset = WBCDataset(
        samples, build_transforms(cfg.data.image_size, False), cfg.preprocessing.crop
    )
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
