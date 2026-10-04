import csv

import pytest

from wbc_classification.engine.checkpoint import load_checkpoint
from wbc_classification.engine.train import train
from wbc_classification.errors import ConfigError


def test_train_writes_all_artifacts(trained):
    run = trained.run_dir
    for name in ("config.yaml", "metrics.csv", "curves.png", "best.pt", "last.pt"):
        assert (run / name).is_file(), name
    assert len(trained.history) == 1
    assert 0.0 <= trained.best_val_accuracy <= 1.0
    with (run / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["epoch"] == "1" and "val_acc" in rows[0]


def test_trained_checkpoint_loads(trained):
    model, class_names, cfg = load_checkpoint(trained.best_checkpoint)
    assert len(class_names) == 4 and cfg.model.name == "baseline"


def test_freeze_on_baseline_is_rejected(tiny_config):
    tiny_config.model.freeze_backbone_epochs = 1
    with pytest.raises(ConfigError, match="baseline"):
        train(tiny_config)


def test_early_stopping(tiny_config):
    tiny_config.train.epochs = 6
    tiny_config.train.patience = 1
    result = train(tiny_config)
    assert 1 <= len(result.history) <= 6


def test_same_seed_same_first_epoch(tiny_config, tmp_path):
    first = train(tiny_config).history[0]["train_loss"]
    tiny_config.train.output_dir = tmp_path / "again"
    assert train(tiny_config).history[0]["train_loss"] == pytest.approx(first, rel=1e-4)
