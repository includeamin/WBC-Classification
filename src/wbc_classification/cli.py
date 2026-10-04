"""`wbc` command-line interface."""

import functools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
import yaml
from PIL import Image

from wbc_classification import __version__
from wbc_classification.config import load_config
from wbc_classification.data.datasets import load_image
from wbc_classification.data.download import download_dataset
from wbc_classification.engine.evaluate import evaluate_checkpoint
from wbc_classification.engine.metrics import format_report
from wbc_classification.engine.predict import annotate_predictions, predict_images
from wbc_classification.engine.train import train as run_training
from wbc_classification.errors import ConfigError, WBCError
from wbc_classification.preprocessing.segmentation import crop_cell, draw_overlay, segment_cell

app = typer.Typer(
    help="Classify white blood cell images (eosinophil, lymphocyte, monocyte, neutrophil).",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Log progress per epoch.")
    ] = False,
) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")


def _handle_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except WBCError as exc:
            typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

    return wrapper


def _parse_set_options(items: list[str]) -> dict[str, Any]:
    """Turn ``section.field=value`` strings into typed config overrides."""
    parsed: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or "." not in key or key.startswith(".") or key.endswith("."):
            raise ConfigError(f"--set expects section.field=value, got {item!r}")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ConfigError(f"--set {item!r}: cannot parse value: {exc}") from exc
        if value is None:
            raise ConfigError(f"--set {item!r} has an empty value; expected section.field=value")
        parsed[key] = value
    return parsed


@app.command()
@_handle_errors
def train(
    config: Annotated[Path | None, typer.Option("--config", "-c", help="YAML config file.")] = None,
    model: Annotated[
        str | None, typer.Option(help="baseline, resnet18, resnet50, efficientnet_b0.")
    ] = None,
    epochs: Annotated[int | None, typer.Option(help="Number of epochs.")] = None,
    batch_size: Annotated[int | None, typer.Option(help="Batch size.")] = None,
    image_size: Annotated[int | None, typer.Option(help="Square input size in pixels.")] = None,
    lr: Annotated[float | None, typer.Option(help="Learning rate.")] = None,
    data_root: Annotated[
        Path | None, typer.Option(help="Folder containing TRAIN/ and TEST/.")
    ] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Where run folders are written.")] = None,
    device: Annotated[str | None, typer.Option(help="auto, cpu, cuda, mps.")] = None,
    name: Annotated[str | None, typer.Option(help="Run name prefix.")] = None,
    pretrained: Annotated[
        bool | None,
        typer.Option(
            "--pretrained/--no-pretrained",
            help="Use ImageNet-pretrained weights (backbones only).",
        ),
    ] = None,
    crop: Annotated[
        bool | None, typer.Option("--crop/--no-crop", help="Crop to the cell first.")
    ] = None,
    set_options: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            "-s",
            help="Override any config field, e.g. --set train.seed=1 (repeatable).",
        ),
    ] = None,
) -> None:
    """Train a model and write a run folder with checkpoints, metrics and plots."""
    cfg = load_config(
        config,
        {
            "model.name": model,
            "train.epochs": epochs,
            "data.batch_size": batch_size,
            "data.image_size": image_size,
            "train.lr": lr,
            "data.root": data_root,
            "train.output_dir": output_dir,
            "train.device": device,
            "train.name": name,
            "model.pretrained": pretrained,
            "preprocessing.crop": crop,
            **_parse_set_options(set_options or []),
        },
    )
    result = run_training(cfg)
    typer.echo(f"Best validation accuracy: {result.best_val_accuracy:.4f}")
    typer.echo(f"Run folder: {result.run_dir}")
    typer.echo(f"Best checkpoint: {result.best_checkpoint}")


@app.command()
@_handle_errors
def evaluate(
    checkpoint: Annotated[Path, typer.Argument(help="Checkpoint (.pt) written by `wbc train`.")],
    data_root: Annotated[Path | None, typer.Option(help="Folder containing TEST/.")] = None,
    device: Annotated[str, typer.Option(help="auto, cpu, cuda, mps.")] = "auto",
    output: Annotated[
        Path | None, typer.Option(help="Write metrics JSON (+ confusion PNG) here.")
    ] = None,
) -> None:
    """Evaluate a checkpoint on the TEST split."""
    metrics = evaluate_checkpoint(checkpoint, data_root, device, output)
    typer.echo(format_report(metrics))


@app.command()
@_handle_errors
def predict(
    inputs: Annotated[list[Path], typer.Argument(help="Image files and/or folders of images.")],
    checkpoint: Annotated[Path, typer.Option("--checkpoint", "-c", help="Checkpoint (.pt).")],
    device: Annotated[str, typer.Option(help="auto, cpu, cuda, mps.")] = "auto",
    annotate_dir: Annotated[
        Path | None, typer.Option(help="Save labelled copies of the images here.")
    ] = None,
) -> None:
    """Predict the cell type of one or more images."""
    predictions = predict_images(checkpoint, inputs, device)
    for prediction in predictions:
        confidence = prediction.probabilities[prediction.label]
        typer.echo(f"{prediction.path}: {prediction.label} ({confidence:.1%})")
    if annotate_dir is not None:
        annotate_predictions(predictions, annotate_dir)


@app.command()
@_handle_errors
def segment(
    image: Annotated[Path, typer.Argument(help="Image to segment.")],
    output_dir: Annotated[Path, typer.Option(help="Where to write crop/overlay/mask PNGs.")] = Path(
        "segmentation"
    ),
) -> None:
    """Locate the cell with HSV segmentation and save the crop, overlay and mask."""
    rgb = np.asarray(load_image(image))
    segmentation = segment_cell(rgb)
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(crop_cell(rgb)).save(output_dir / f"{image.stem}_crop.png")
    Image.fromarray(draw_overlay(rgb, segmentation)).save(output_dir / f"{image.stem}_overlay.png")
    Image.fromarray(segmentation.mask).save(output_dir / f"{image.stem}_mask.png")
    typer.echo(f"Saved crop, overlay and mask to {output_dir}")


@app.command("download-data")
@_handle_errors
def download_data(
    dest: Annotated[Path, typer.Option(help="Destination folder.")] = Path("data"),
    force: Annotated[bool, typer.Option(help="Overwrite existing TRAIN/TEST folders.")] = False,
) -> None:
    """Download the Kaggle blood-cells dataset into DEST (needs the `kaggle` extra)."""
    path = download_dataset(dest, force)
    typer.echo(f"Dataset installed in {path}")
