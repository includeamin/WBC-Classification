"""Experiment configuration: pydantic models, YAML loading and CLI overrides."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wbc_classification.errors import ConfigError


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PreprocessingConfig(_Section):
    crop: bool = False


class DataConfig(_Section):
    root: Path = Path("data")
    image_size: int = Field(default=224, ge=16)
    batch_size: int = Field(default=32, ge=1)
    val_fraction: float = Field(default=0.2, gt=0, lt=0.5)
    num_workers: int = Field(default=0, ge=0)


class ModelConfig(_Section):
    name: str = "resnet18"
    pretrained: bool = True
    freeze_backbone_epochs: int = Field(default=0, ge=0)


class TrainConfig(_Section):
    name: str = "run"
    epochs: int = Field(default=30, ge=1)
    lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    patience: int = Field(default=7, ge=1)
    amp: bool = True
    seed: int = 42
    device: str = "auto"
    output_dir: Path = Path("runs")


class Config(_Section):
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()


def load_config(path: Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config (optional) and apply ``{"section.field": value}`` overrides."""
    raw: dict[str, Any] = {}
    if path is not None:
        if not path.is_file():
            raise ConfigError(f"Config file not found: {path}")
        try:
            loaded = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"Config {path} must be a mapping of sections")
        raw = loaded

    for key, value in (overrides or {}).items():
        if value is None:
            continue
        section, _, field = key.partition(".")
        if not field:
            raise ConfigError(f"Override key must look like 'section.field', got {key!r}")
        target = raw.setdefault(section, {})
        if not isinstance(target, dict):
            raise ConfigError(f"Config section {section!r} must be a mapping")
        target[field] = value

    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(f"Invalid configuration: {details}") from exc


def dump_config(config: Config) -> str:
    """Serialise a config to YAML (paths become strings)."""
    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
