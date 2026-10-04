import json

import pytest
import torch

from wbc_classification.config import Config, ModelConfig
from wbc_classification.engine.checkpoint import load_checkpoint, save_checkpoint
from wbc_classification.engine.metrics import (
    compute_metrics,
    format_report,
    save_confusion_matrix,
    save_curves,
)
from wbc_classification.engine.runtime import resolve_device, set_seed
from wbc_classification.errors import CheckpointError, ConfigError
from wbc_classification.models.registry import build_model

NAMES = ["A", "B", "C", "D"]


def test_resolve_device():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}
    with pytest.raises(ConfigError, match="bogus"):
        resolve_device("bogus")


def test_set_seed_makes_torch_deterministic():
    set_seed(3)
    first = torch.rand(3)
    set_seed(3)
    assert torch.equal(first, torch.rand(3))


def test_compute_metrics_perfect():
    m = compute_metrics([0, 1, 2, 3], [0, 1, 2, 3], NAMES)
    assert m["accuracy"] == 1.0
    assert m["confusion_matrix"] == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    json.dumps(m)  # must be JSON-serialisable
    assert "A" in format_report(m)


def test_compute_metrics_handles_missing_class_in_predictions():
    m = compute_metrics([0, 0, 1, 1], [0, 0, 0, 0], NAMES)
    assert m["accuracy"] == 0.5 and len(m["confusion_matrix"]) == 4
    json.dumps(m)


def test_plots_are_written(tmp_path):
    save_confusion_matrix([[1, 0], [0, 1]], ["A", "B"], tmp_path / "cm.png")
    history = [
        {
            "epoch": 1,
            "train_loss": 1.0,
            "train_acc": 0.5,
            "val_loss": 1.1,
            "val_acc": 0.4,
            "lr": 0.1,
        },
        {
            "epoch": 2,
            "train_loss": 0.8,
            "train_acc": 0.6,
            "val_loss": 0.9,
            "val_acc": 0.5,
            "lr": 0.05,
        },
    ]
    save_curves(history, tmp_path / "curves.png")
    assert (tmp_path / "cm.png").stat().st_size > 0 and (tmp_path / "curves.png").stat().st_size > 0


def _baseline():
    return build_model("baseline", num_classes=4)


def test_checkpoint_roundtrip(tmp_path):
    model = _baseline().eval()
    cfg = Config(model=ModelConfig(name="baseline", pretrained=False))
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, NAMES, cfg, epoch=3, val_accuracy=0.9)
    loaded, names, loaded_cfg = load_checkpoint(path)
    assert names == NAMES and loaded_cfg == cfg
    x = torch.rand(1, 3, 32, 32)
    assert torch.allclose(model(x), loaded.eval()(x))


def test_checkpoint_missing(tmp_path):
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "nope.pt")


def test_checkpoint_corrupt_file(tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"junk")
    with pytest.raises(CheckpointError, match="bad.pt"):
        load_checkpoint(bad)


def test_checkpoint_not_ours(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"hello": 1}, path)
    with pytest.raises(CheckpointError, match="not a wbc-classification checkpoint"):
        load_checkpoint(path)


def test_checkpoint_class_count_mismatch(tmp_path):
    cfg = Config(model=ModelConfig(name="baseline", pretrained=False))
    path = tmp_path / "m.pt"
    save_checkpoint(
        path, _baseline(), ["A", "B", "C"], cfg, epoch=1, val_accuracy=0.1
    )  # 4 outputs vs 3 names
    with pytest.raises(CheckpointError, match="do not match"):
        load_checkpoint(path)


def _real_checkpoint(tmp_path):
    cfg = Config(model=ModelConfig(name="baseline", pretrained=False))
    path = tmp_path / "real.pt"
    save_checkpoint(path, _baseline(), NAMES, cfg, epoch=1, val_accuracy=0.1)
    return path.read_bytes()


def test_checkpoint_plain_text_file(tmp_path):
    bad = tmp_path / "text.pt"
    bad.write_text("hello world")
    with pytest.raises(CheckpointError, match="text.pt"):
        load_checkpoint(bad)


def test_checkpoint_truncated_file(tmp_path):
    data = _real_checkpoint(tmp_path)
    bad = tmp_path / "truncated.pt"
    bad.write_bytes(data[: len(data) // 2])
    with pytest.raises(CheckpointError, match="truncated.pt"):
        load_checkpoint(bad)


def test_checkpoint_bit_flipped_file(tmp_path):
    data = bytearray(_real_checkpoint(tmp_path))
    start = 100  # inside the zip's pickle record (tensor payload bytes would load silently)
    for i in range(start, start + 64):
        data[i] ^= 0xFF
    bad = tmp_path / "flipped.pt"
    bad.write_bytes(bytes(data))
    with pytest.raises(CheckpointError, match="flipped.pt"):
        load_checkpoint(bad)
