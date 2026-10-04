"""IncludeNet: the project's original small CNN, rewritten in PyTorch."""

import torch
from torch import nn


def _block(in_channels: int, out_channels: int, pool: bool) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class IncludeNet(nn.Module):
    """Four conv blocks, global average pooling and a linear head; input-size independent."""

    def __init__(self, num_classes: int, width: int = 32, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _block(3, width, pool=True),
            _block(width, 2 * width, pool=True),
            _block(2 * width, 4 * width, pool=True),
            _block(4 * width, 4 * width, pool=False),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(4 * width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x)).flatten(1)
        return self.classifier(self.dropout(x))
