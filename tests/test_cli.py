import pytest
from typer.testing import CliRunner

from wbc_classification.cli import app

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("train", "evaluate", "predict", "segment", "download-data"):
        assert command in result.output


def test_train_command(data_root, tmp_path):
    result = runner.invoke(
        app,
        [
            "train",
            "--data-root",
            str(data_root),
            "--output-dir",
            str(tmp_path),
            "--model",
            "baseline",
            "--no-pretrained",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--image-size",
            "32",
            "--device",
            "cpu",
            "--name",
            "cli",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "best validation accuracy" in result.output.lower()
    assert list(tmp_path.glob("cli-*/best.pt"))


def test_evaluate_command(trained, data_root, tmp_path):
    out = tmp_path / "m.json"
    result = runner.invoke(
        app,
        [
            "evaluate",
            str(trained.best_checkpoint),
            "--data-root",
            str(data_root),
            "--device",
            "cpu",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "accuracy" in result.output and out.is_file()


def test_predict_command(trained, cell_image, tmp_path):
    result = runner.invoke(
        app,
        [
            "predict",
            str(cell_image),
            "-c",
            str(trained.best_checkpoint),
            "--device",
            "cpu",
            "--annotate-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "cell.jpeg" in result.output
    assert (tmp_path / "cell_pred.png").is_file()


def test_segment_command(cell_image, tmp_path):
    result = runner.invoke(app, ["segment", str(cell_image), "--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    for suffix in ("crop", "overlay", "mask"):
        assert (tmp_path / f"cell_{suffix}.png").is_file()


def test_segment_no_cell_is_a_clean_error(tmp_path, write_image):
    black = write_image(tmp_path / "black.png")
    result = runner.invoke(app, ["segment", str(black), "--output-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "Error:" in result.output and "Traceback" not in result.output


def test_missing_checkpoint_is_a_clean_error(tmp_path):
    result = runner.invoke(app, ["evaluate", str(tmp_path / "nope.pt")])
    assert result.exit_code == 1 and "Checkpoint not found" in result.output


def test_bad_config_is_a_clean_error(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("train:\n  epohcs: 2\n")
    result = runner.invoke(app, ["train", "-c", str(cfg)])
    assert result.exit_code == 1 and "epohcs" in result.output


def test_predict_same_stem_inputs_get_distinct_annotations(trained, tmp_path, write_image):
    first = write_image(tmp_path / "a" / "cell.png")
    second = write_image(tmp_path / "b" / "cell.png")
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "predict",
            str(first),
            str(second),
            "-c",
            str(trained.best_checkpoint),
            "--device",
            "cpu",
            "--annotate-dir",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(list(out.glob("*.png"))) == 2


_TINY = [
    "--model", "baseline", "--no-pretrained", "--epochs", "1",
    "--batch-size", "4", "--image-size", "32", "--device", "cpu", "--name", "cli",
]  # fmt: skip


def test_set_overrides_any_field(data_root, tmp_path):
    result = runner.invoke(
        app,
        ["train", "--data-root", str(data_root), "--output-dir", str(tmp_path), *_TINY,
         "--set", "train.seed=7", "--set", "data.num_workers=0", "--set", "train.patience=3"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    saved = next(tmp_path.glob("cli-*/config.yaml")).read_text()
    assert "seed: 7" in saved and "patience: 3" in saved


def test_set_wins_over_named_flag(data_root, tmp_path):
    result = runner.invoke(
        app,
        ["train", "--data-root", str(data_root), "--output-dir", str(tmp_path), *_TINY,
         "--set", "train.name=winner"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert list(tmp_path.glob("winner-*/best.pt"))


@pytest.mark.parametrize("item", ["train.seed", "seed=3", "train.seed=", "train.seed=null"])
def test_set_malformed_is_a_clean_error(item):
    result = runner.invoke(app, ["train", "--set", item])
    assert result.exit_code == 1
    assert "Error:" in result.output and "Traceback" not in result.output


def test_set_unknown_field_names_it():
    result = runner.invoke(app, ["train", "--set", "train.epohcs=3"])
    assert result.exit_code == 1 and "epohcs" in result.output
