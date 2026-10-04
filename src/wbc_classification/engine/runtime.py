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
        device = torch.device(name)
    except RuntimeError as exc:
        raise ConfigError(f"Unknown device {name!r} (use auto, cpu, cuda, cuda:0, mps)") from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ConfigError(f"CUDA was requested ({name!r}) but no CUDA device is available")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ConfigError(
                f"CUDA device {device.index} was requested but only "
                f"{torch.cuda.device_count()} CUDA device(s) are available"
            )
    elif device.type == "mps" and not torch.backends.mps.is_available():
        raise ConfigError(f"MPS was requested ({name!r}) but no MPS device is available")
    return device
