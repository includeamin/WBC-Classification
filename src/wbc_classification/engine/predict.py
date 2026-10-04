"""Predict classes for images or folders of images."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import ImageDraw

from wbc_classification.data.datasets import load_image
from wbc_classification.data.labels import is_image_file
from wbc_classification.data.transforms import build_transforms
from wbc_classification.engine.checkpoint import load_checkpoint
from wbc_classification.engine.runtime import resolve_device
from wbc_classification.errors import DataError


@dataclass(frozen=True)
class Prediction:
    path: Path
    label: str
    probabilities: dict[str, float]


def collect_images(inputs: Sequence[Path]) -> list[Path]:
    images: list[Path] = []
    for item in inputs:
        if item.is_dir():
            images.extend(sorted(p for p in item.rglob("*") if is_image_file(p)))
        elif item.is_file():
            images.append(item)
        else:
            raise DataError(f"Path not found: {item}")
    if not images:
        raise DataError("No images found in the given inputs")
    return images


def predict_images(
    checkpoint: Path, inputs: Sequence[Path], device: str = "auto"
) -> list[Prediction]:
    model, class_names, cfg = load_checkpoint(checkpoint)
    torch_device = resolve_device(device)
    model.to(torch_device).eval()
    transform = build_transforms(cfg.data.image_size, train=False)

    predictions: list[Prediction] = []
    with torch.no_grad():
        for path in collect_images(inputs):
            image = load_image(path, crop=cfg.preprocessing.crop)
            batch = transform(image).unsqueeze(0).to(torch_device)
            probs = torch.softmax(model(batch), dim=1)[0].cpu().tolist()
            best = max(range(len(class_names)), key=probs.__getitem__)
            predictions.append(
                Prediction(path, class_names[best], dict(zip(class_names, probs, strict=True)))
            )
    return predictions


def annotate_image(prediction: Prediction, output_dir: Path, name: str | None = None) -> Path:
    """Write a copy of the image with the predicted label drawn on it.

    The file is called ``name`` if given, otherwise ``<stem>_pred.png``.
    """
    image = load_image(prediction.path)
    confidence = prediction.probabilities[prediction.label]
    ImageDraw.Draw(image).text((8, 8), f"{prediction.label} {confidence:.0%}", fill=(0, 255, 0))
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / (name or f"{prediction.path.stem}_pred.png")
    image.save(out)
    return out


def annotate_predictions(predictions: Sequence[Prediction], output_dir: Path) -> list[Path]:
    """Annotate a batch, giving predictions that share a stem unique file names."""
    seen: dict[str, int] = {}
    written: list[Path] = []
    for prediction in predictions:
        stem = prediction.path.stem
        seen[stem] = seen.get(stem, 0) + 1
        count = seen[stem]
        name = f"{stem}_pred.png" if count == 1 else f"{stem}_{count}_pred.png"
        written.append(annotate_image(prediction, output_dir, name=name))
    return written
