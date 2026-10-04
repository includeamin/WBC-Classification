"""Classification metrics and plots."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def compute_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]
) -> dict[str, Any]:
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "report": report,
        "confusion_matrix": matrix.tolist(),
        "class_names": list(class_names),
    }
    return json.loads(json.dumps(result, default=float))  # plain, JSON-safe types


def format_report(metrics: dict[str, Any]) -> str:
    lines = [f"{'class':<14}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"]
    for name in metrics["class_names"]:
        row = metrics["report"][name]
        lines.append(
            f"{name:<14}{row['precision']:>10.3f}{row['recall']:>10.3f}"
            f"{row['f1-score']:>10.3f}{int(row['support']):>10}"
        )
    lines.append(f"{'accuracy':<14}{metrics['accuracy']:>40.3f}")
    return "\n".join(lines)


def save_confusion_matrix(
    matrix: Sequence[Sequence[int]], class_names: Sequence[str], path: Path
) -> None:
    fig = Figure(figsize=(6, 5))
    ax = fig.subplots()
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def save_curves(history: Sequence[dict[str, float]], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig = Figure(figsize=(10, 4))
    loss_ax, acc_ax = fig.subplots(1, 2)
    loss_ax.plot(epochs, [r["train_loss"] for r in history], label="train")
    loss_ax.plot(epochs, [r["val_loss"] for r in history], label="val")
    loss_ax.set(title="Loss", xlabel="Epoch")
    acc_ax.plot(epochs, [r["train_acc"] for r in history], label="train")
    acc_ax.plot(epochs, [r["val_acc"] for r in history], label="val")
    acc_ax.set(title="Accuracy", xlabel="Epoch")
    loss_ax.legend()
    acc_ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
