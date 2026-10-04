import json

import pytest

from wbc_classification.engine.evaluate import evaluate_checkpoint
from wbc_classification.engine.predict import (
    annotate_image,
    annotate_predictions,
    collect_images,
    predict_images,
)
from wbc_classification.errors import CheckpointError, DataError


def test_evaluate_reports_per_class_and_writes_outputs(trained, data_root, tmp_path):
    out = tmp_path / "metrics.json"
    metrics = evaluate_checkpoint(
        trained.best_checkpoint, data_root=data_root, device="cpu", output=out
    )
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert set(metrics["class_names"]) == {"EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"}
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == 8  # the 8 TEST fixtures
    assert json.loads(out.read_text())["accuracy"] == metrics["accuracy"]
    assert (tmp_path / "metrics_confusion.png").is_file()


def test_evaluate_missing_checkpoint(tmp_path):
    with pytest.raises(CheckpointError):
        evaluate_checkpoint(tmp_path / "nope.pt")


def test_evaluate_missing_data_dir(trained, tmp_path):
    with pytest.raises(DataError, match="wbc download-data"):
        evaluate_checkpoint(trained.best_checkpoint, data_root=tmp_path, device="cpu")


def test_predict_directory(trained, data_root):
    preds = predict_images(trained.best_checkpoint, [data_root / "TEST"], device="cpu")
    assert len(preds) == 8
    for pred in preds:
        assert pred.label in pred.probabilities
        assert sum(pred.probabilities.values()) == pytest.approx(1.0, abs=1e-4)


def test_predict_single_file(trained, cell_image):
    (pred,) = predict_images(trained.best_checkpoint, [cell_image], device="cpu")
    assert pred.path == cell_image


def test_predict_empty_dir_raises(trained, tmp_path):
    with pytest.raises(DataError, match="No images"):
        predict_images(trained.best_checkpoint, [tmp_path], device="cpu")


def test_predict_missing_path_raises(trained, tmp_path):
    with pytest.raises(DataError, match="not found"):
        predict_images(trained.best_checkpoint, [tmp_path / "ghost.jpg"], device="cpu")


def test_predict_corrupt_image_raises(trained, tmp_path):
    bad = tmp_path / "bad.jpeg"
    bad.write_bytes(b"nope")
    with pytest.raises(DataError, match="bad.jpeg"):
        predict_images(trained.best_checkpoint, [bad], device="cpu")


def test_collect_images_ignores_non_images(tmp_path, write_image):
    write_image(tmp_path / "a.png")
    (tmp_path / "notes.txt").write_text("x")
    assert collect_images([tmp_path]) == [tmp_path / "a.png"]


def test_annotate_writes_file(trained, cell_image, tmp_path):
    (pred,) = predict_images(trained.best_checkpoint, [cell_image], device="cpu")
    assert annotate_image(pred, tmp_path).is_file()


def test_annotate_predictions_same_stem_in_subdirs(trained, write_image, tmp_path):
    write_image(tmp_path / "a" / "x.png")
    write_image(tmp_path / "b" / "x.png")
    preds = predict_images(trained.best_checkpoint, [tmp_path / "a", tmp_path / "b"], device="cpu")
    out = tmp_path / "out"
    paths = annotate_predictions(preds, out)
    assert [p.name for p in paths] == ["x_pred.png", "x_2_pred.png"]
    assert all(p.is_file() for p in paths)


def test_annotate_predictions_same_stem_different_suffix(trained, write_image, tmp_path):
    write_image(tmp_path / "x.png")
    write_image(tmp_path / "x.jpg")
    preds = predict_images(trained.best_checkpoint, [tmp_path], device="cpu")
    paths = annotate_predictions(preds, tmp_path / "out")
    assert len({p.name for p in paths}) == 2
    assert all(p.is_file() for p in paths)


def test_annotate_predictions_unique_stem(trained, cell_image, tmp_path):
    preds = predict_images(trained.best_checkpoint, [cell_image], device="cpu")
    (path,) = annotate_predictions(preds, tmp_path)
    assert path.name == f"{cell_image.stem}_pred.png"
