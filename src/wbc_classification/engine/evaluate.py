"""Evaluate a checkpoint on the TEST split."""

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from wbc_classification.data.datasets import build_test_loader
from wbc_classification.engine.checkpoint import load_checkpoint
from wbc_classification.engine.metrics import compute_metrics, save_confusion_matrix
from wbc_classification.engine.runtime import resolve_device


def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[int], list[int]]:
    """Run ``model`` over ``loader`` in eval mode; return ``(y_true, y_pred)`` label lists."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.tolist())
    return y_true, y_pred


def evaluate_checkpoint(
    checkpoint: Path,
    data_root: Path | None = None,
    device: str = "auto",
    output: Path | None = None,
) -> dict[str, Any]:
    model, class_names, cfg = load_checkpoint(checkpoint)
    if data_root is not None:
        cfg.data.root = data_root
    torch_device = resolve_device(device)
    model.to(torch_device).eval()
    loader = build_test_loader(cfg, class_names)

    y_true, y_pred = collect_predictions(model, loader, torch_device)
    metrics = compute_metrics(y_true, y_pred, class_names)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2))
        save_confusion_matrix(
            metrics["confusion_matrix"],
            class_names,
            output.with_name(f"{output.stem}_confusion.png"),
        )
    return metrics
