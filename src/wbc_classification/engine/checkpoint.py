"""Single-file checkpoints: weights + class names + config."""

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from wbc_classification.config import Config
from wbc_classification.errors import CheckpointError, ConfigError
from wbc_classification.models.registry import build_model

_REQUIRED_KEYS = {"state_dict", "class_names", "config"}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_names: Sequence[str],
    config: Config,
    epoch: int,
    val_accuracy: float,
) -> None:
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "class_names": list(class_names),
            "config": config.model_dump(mode="json"),
            "epoch": epoch,
            "val_accuracy": val_accuracy,
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[nn.Module, list[str], Config]:
    if not path.is_file():
        raise CheckpointError(f"Checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # torch raises varying types for corrupt files across versions
        raise CheckpointError(f"Cannot read checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or not payload.keys() >= _REQUIRED_KEYS:
        raise CheckpointError(f"{path} is not a wbc-classification checkpoint")
    try:
        config = Config.model_validate(payload["config"])
        class_names = list(payload["class_names"])
        model = build_model(config.model.name, num_classes=len(class_names), pretrained=False)
        model.load_state_dict(payload["state_dict"])
    except ConfigError as exc:
        raise CheckpointError(f"{path} stores an unusable model config: {exc}") from exc
    except ValueError as exc:  # pydantic.ValidationError subclasses ValueError
        raise CheckpointError(f"{path} stores an invalid config: {exc}") from exc
    except RuntimeError as exc:
        raise CheckpointError(
            f"Weights in {path} do not match its stored class names / model ({exc})"
        ) from exc
    return model, class_names, config
