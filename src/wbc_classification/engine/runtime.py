"""Seeding and device selection."""

import random

import numpy as np
import torch

from wbc_classification.errors import ConfigError


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        return torch.device(name)
    except RuntimeError as exc:
        raise ConfigError(f"Unknown device {name!r} (use auto, cpu, cuda, cuda:0, mps)") from exc
