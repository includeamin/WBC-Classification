"""Name -> model factory."""

from torch import nn

from wbc_classification.errors import ConfigError
from wbc_classification.models.backbone import SUPPORTED_BACKBONES, build_backbone
from wbc_classification.models.baseline import IncludeNet

MODEL_NAMES = ("baseline", *SUPPORTED_BACKBONES)


def build_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    if name == "baseline":
        return IncludeNet(num_classes)  # no pretrained weights exist for the baseline
    if name in SUPPORTED_BACKBONES:
        return build_backbone(name, num_classes, pretrained)
    raise ConfigError(f"Unknown model {name!r}. Available: {', '.join(MODEL_NAMES)}")


def supports_freezing(name: str) -> bool:
    return name in SUPPORTED_BACKBONES
