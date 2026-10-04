from pathlib import Path

import pytest

from wbc_classification.config import Config, dump_config, load_config
from wbc_classification.errors import ConfigError


def test_defaults():
    cfg = Config()
    assert cfg.model.name == "resnet18"
    assert cfg.preprocessing.crop is False
    assert cfg.data.root == Path("data")


def test_load_yaml_with_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("model:\n  name: baseline\ntrain:\n  epochs: 3\n")
    cfg = load_config(path, {"train.epochs": 5, "data.image_size": None, "data.batch_size": 8})
    assert cfg.model.name == "baseline"
    assert cfg.train.epochs == 5
    assert cfg.data.image_size == 224  # None override ignored
    assert cfg.data.batch_size == 8


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_typo_in_section_is_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("trian:\n  epochs: 3\n")
    with pytest.raises(ConfigError, match="trian"):
        load_config(path)


def test_typo_in_field_names_the_key(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("train:\n  epohcs: 3\n")
    with pytest.raises(ConfigError, match="epohcs"):
        load_config(path)


def test_out_of_range_value():
    with pytest.raises(ConfigError, match="val_fraction"):
        load_config(None, {"data.val_fraction": 0.9})


def test_bad_override_key():
    with pytest.raises(ConfigError, match="section.field"):
        load_config(None, {"epochs": 3})


def test_invalid_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("train: [unclosed")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(path)


def test_dump_roundtrip(tmp_path):
    cfg = load_config(None, {"train.epochs": 7, "data.root": "somewhere"})
    path = tmp_path / "out.yaml"
    path.write_text(dump_config(cfg))
    assert load_config(path) == cfg
