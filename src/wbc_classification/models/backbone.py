"""Pretrained torchvision backbones with a replaced classification head."""

from typing import Any, cast

from torch import nn
from torchvision import models as tv_models

from wbc_classification.errors import ConfigError

SUPPORTED_BACKBONES = ("resnet18", "resnet50", "efficientnet_b0")


def build_backbone(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    try:
        model = cast(Any, tv_models.get_model(name, weights="DEFAULT" if pretrained else None))
    except Exception as exc:
        if not pretrained:
            raise
        raise ConfigError(
            f"Could not download pretrained weights for {name!r}: {exc}. "
            "Check your network connection or set model.pretrained: false."
        ) from exc
    if name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return cast(nn.Module, model)


def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    """Freeze everything except the classification head (or unfreeze everything)."""
    m = cast(Any, model)
    head = m.fc if hasattr(m, "fc") else m.classifier
    head_ids = {id(p) for p in head.parameters()}
    for param in model.parameters():
        param.requires_grad = (not frozen) or id(param) in head_ids
