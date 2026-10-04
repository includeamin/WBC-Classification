"""Training loop."""

import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from wbc_classification.config import Config, dump_config
from wbc_classification.data.datasets import build_train_val_loaders
from wbc_classification.engine.checkpoint import load_checkpoint, save_checkpoint
from wbc_classification.engine.evaluate import collect_predictions
from wbc_classification.engine.metrics import compute_metrics, save_confusion_matrix, save_curves
from wbc_classification.engine.runtime import resolve_device, set_seed
from wbc_classification.errors import ConfigError
from wbc_classification.models.backbone import set_backbone_frozen
from wbc_classification.models.registry import build_model, supports_freezing

logger = logging.getLogger(__name__)

HISTORY_FIELDS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]


@dataclass(frozen=True)
class TrainResult:
    run_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    best_val_accuracy: float
    history: list[dict[str, float]]


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp: bool = False,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if optimizer is not None and scaler is not None:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)
    return total_loss / count, correct / count


def train(cfg: Config) -> TrainResult:
    if cfg.model.freeze_backbone_epochs and not supports_freezing(cfg.model.name):
        raise ConfigError(
            f"model.freeze_backbone_epochs is not supported for {cfg.model.name!r} "
            "(it has no pretrained backbone)"
        )
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    loaders = build_train_val_loaders(cfg)
    model = build_model(cfg.model.name, len(loaders.class_names), cfg.model.pretrained).to(device)

    run_dir = cfg.train.output_dir / f"{cfg.train.name}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(dump_config(cfg))

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    best_acc, bad_epochs = -1.0, 0
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.train.epochs + 1):
        if cfg.model.freeze_backbone_epochs:
            set_backbone_frozen(model, epoch <= cfg.model.freeze_backbone_epochs)
        train_loss, train_acc = _run_epoch(
            model, loaders.train, device, criterion, optimizer, scaler, amp
        )
        val_loss, val_acc = _run_epoch(model, loaders.val, device, criterion, amp=amp)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        scheduler.step()
        logger.info(
            "epoch %d/%d train_loss=%.4f train_acc=%.3f val_loss=%.4f val_acc=%.3f",
            epoch,
            cfg.train.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )
        save_checkpoint(last_path, model, loaders.class_names, cfg, epoch, val_acc)
        if val_acc > best_acc:
            best_acc, bad_epochs = val_acc, 0
            save_checkpoint(best_path, model, loaders.class_names, cfg, epoch, val_acc)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.train.patience:
                logger.info("early stopping at epoch %d", epoch)
                break

    with (run_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)
    save_curves(history, run_dir / "curves.png")

    best_model, _, _ = load_checkpoint(best_path)
    y_true, y_pred = collect_predictions(best_model.to(device), loaders.val, device)
    metrics = compute_metrics(y_true, y_pred, loaders.class_names)
    save_confusion_matrix(
        metrics["confusion_matrix"], loaders.class_names, run_dir / "confusion_matrix.png"
    )
    return TrainResult(run_dir, best_path, last_path, best_acc, history)
