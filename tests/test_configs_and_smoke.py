from pathlib import Path

import pytest
from typer.testing import CliRunner

from wbc_classification.cli import app
from wbc_classification.config import load_config
from wbc_classification.models.registry import MODEL_NAMES

CONFIGS = sorted((Path(__file__).parent.parent / "configs").glob("*.yaml"))


def test_configs_exist():
    assert len(CONFIGS) >= 3


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_shipped_configs_are_valid(path):
    cfg = load_config(path)
    assert cfg.model.name in MODEL_NAMES


def test_end_to_end_smoke(data_root, tmp_path):
    """train -> evaluate -> predict through the CLI, reloading the saved checkpoint."""
    runner = CliRunner()
    train = runner.invoke(
        app,
        [
            "train",
            "-c",
            str(CONFIGS[0].parent / "baseline.yaml"),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(tmp_path),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--image-size",
            "32",
            "--device",
            "cpu",
        ],
    )
    assert train.exit_code == 0, train.output
    (checkpoint,) = tmp_path.glob("baseline-*/best.pt")
    evaluate = runner.invoke(
        app, ["evaluate", str(checkpoint), "--data-root", str(data_root), "--device", "cpu"]
    )
    assert evaluate.exit_code == 0, evaluate.output
    predict = runner.invoke(
        app, ["predict", str(data_root / "TEST"), "-c", str(checkpoint), "--device", "cpu"]
    )
    assert predict.exit_code == 0, predict.output
    assert predict.output.count("(") == 8  # one confidence per TEST image
