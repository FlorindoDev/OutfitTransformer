"""Training loop for Compatibility Prediction."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from model import FocalLoss
from training.common import (
    BinaryEpochAccumulator,
    EpochMetrics,
    copy_checkpoint,
    save_checkpoint,
    write_json,
)

from .config import (
    CPTrainingConfig,
    MAX_GRAD_NORM,
    ONE_CYCLE_DIV_FACTOR,
    ONE_CYCLE_FINAL_DIV_FACTOR,
    ONE_CYCLE_PCT_START,
)
from .data import CompatibilityLoaders
from .model import CPTrainingModel

LOGGER = logging.getLogger("training.CP")
CHECKPOINT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class TrainingResult:
    """Summary of a completed or early-stopped CP run."""

    best_checkpoint: Path
    best_metric: str
    best_value: float
    epochs_completed: int


def train(
    model: CPTrainingModel,
    loaders: CompatibilityLoaders,
    config: CPTrainingConfig,
    device: torch.device,
) -> TrainingResult:
    """Train CP, checkpoint every epoch and update best model."""
    from .plots import save_cumulative_plots

    config.validate()
    _prepare_output(config.checkpoint_dir)
    run_config = config.as_dict(resolved_device=str(device))

    model.to(device)
    criterion = FocalLoss(
        alpha=config.focal_alpha,
        gamma=config.focal_gamma,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(loaders.train) / config.gradient_accumulation_steps
    )
    if updates_per_epoch == 0:
        raise ValueError("training loader cannot be empty")
    total_optimizer_steps = updates_per_epoch * config.epochs
    run_config["training"]["optimizer_steps_per_epoch"] = updates_per_epoch
    run_config["training"]["total_optimizer_steps"] = total_optimizer_steps
    write_json(run_config, config.checkpoint_dir / "config.json")
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_optimizer_steps,
        pct_start=ONE_CYCLE_PCT_START,
        anneal_strategy="cos",
        div_factor=ONE_CYCLE_DIV_FACTOR,
        final_div_factor=ONE_CYCLE_FINAL_DIV_FACTOR,
    )

    history: dict[str, Any] = {
        "epochs": [],
        "train": [],
        "validation": [],
    }
    best_value: float | None = None
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        train_metrics = _train_epoch(
            model,
            loaders.train,
            criterion,
            optimizer,
            scheduler,
            config,
            device,
            epoch,
        )
        validation_metrics = _evaluate(
            model,
            loaders.validation,
            criterion,
            device,
        )
        history["epochs"].append(epoch)
        history["train"].append(train_metrics.as_dict())
        history["validation"].append(validation_metrics.as_dict())

        monitored_value = _monitored_value(
            config.best_metric,
            validation_metrics,
        )
        improved = _is_improvement(
            monitored_value,
            best_value,
            metric=config.best_metric,
            min_delta=config.early_stopping_min_delta,
        )
        if improved:
            best_value = monitored_value
            stale_epochs = 0
        else:
            stale_epochs += 1

        epoch_path = config.checkpoint_dir / "epochs" / (
            f"cp_epoch_{epoch:03d}.pt"
        )
        payload = _checkpoint_payload(
            epoch=epoch,
            model=model,
            run_config=run_config,
            history=history,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            monitored_value=monitored_value,
            best_value=best_value,
            metric=config.best_metric,
            is_best=improved,
        )
        save_checkpoint(payload, epoch_path)
        if improved:
            copy_checkpoint(epoch_path, config.checkpoint_dir / "best.pt")
        save_cumulative_plots(
            history,
            config.checkpoint_dir / "plots",
            epoch,
        )

        LOGGER.info(
            "epoch=%d/%d train_loss=%.6f train_accuracy=%.4f "
            "train_auc=%.4f val_loss=%.6f val_accuracy=%.4f "
            "val_auc=%.4f best_%s=%.6f",
            epoch,
            config.epochs,
            train_metrics.loss,
            train_metrics.accuracy,
            train_metrics.auc,
            validation_metrics.loss,
            validation_metrics.accuracy,
            validation_metrics.auc,
            config.best_metric,
            best_value,
        )

        if (
            config.early_stopping_patience is not None
            and stale_epochs >= config.early_stopping_patience
        ):
            LOGGER.info(
                "early_stopping epoch=%d stale_epochs=%d",
                epoch,
                stale_epochs,
            )
            break

    if best_value is None:
        raise RuntimeError("training completed without a best checkpoint")
    return TrainingResult(
        best_checkpoint=config.checkpoint_dir / "best.pt",
        best_metric=config.best_metric,
        best_value=best_value,
        epochs_completed=len(history["epochs"]),
    )


def _train_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: AdamW,
    scheduler: OneCycleLR,
    config: CPTrainingConfig,
    device: torch.device,
    epoch: int,
) -> EpochMetrics:
    model.train()
    accumulator = BinaryEpochAccumulator()
    optimizer.zero_grad(set_to_none=True)
    batch_count = len(loader)

    for batch_index, batch in enumerate(loader, start=1):
        labels = batch.labels.to(device=device, non_blocking=config.pin_memory)
        probabilities = model(batch.outfits)
        loss = criterion(probabilities, labels)
        group_size = _accumulation_group_size(
            batch_index,
            batch_count,
            config.gradient_accumulation_steps,
        )
        (loss / group_size).backward()
        accumulator.update(loss, probabilities, labels)

        if _is_optimizer_step(
            batch_index,
            batch_count,
            config.gradient_accumulation_steps,
        ):
            clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if batch_index % config.log_every == 0 or batch_index == batch_count:
            LOGGER.info(
                "epoch=%d microbatch=%d/%d loss=%.6f lr=%.8g",
                epoch,
                batch_index,
                batch_count,
                float(loss.detach().item()),
                scheduler.get_last_lr()[0],
            )

    return accumulator.compute()


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    model.eval()
    accumulator = BinaryEpochAccumulator()
    for batch in loader:
        labels = batch.labels.to(device=device, non_blocking=True)
        probabilities = model(batch.outfits)
        loss = criterion(probabilities, labels)
        accumulator.update(loss, probabilities, labels)
    return accumulator.compute()


def _accumulation_group_size(
    batch_index: int,
    batch_count: int,
    accumulation_steps: int,
) -> int:
    group_start = ((batch_index - 1) // accumulation_steps) * accumulation_steps
    return min(accumulation_steps, batch_count - group_start)


def _is_optimizer_step(
    batch_index: int,
    batch_count: int,
    accumulation_steps: int,
) -> bool:
    return batch_index % accumulation_steps == 0 or batch_index == batch_count


def _monitored_value(metric: str, validation: EpochMetrics) -> float:
    values = {
        "val_auc": validation.auc,
        "val_accuracy": validation.accuracy,
        "val_loss": validation.loss,
    }
    try:
        return values[metric]
    except KeyError as error:
        raise ValueError(f"unsupported best metric: {metric}") from error


def _is_improvement(
    value: float,
    best_value: float | None,
    *,
    metric: str,
    min_delta: float,
) -> bool:
    if best_value is None:
        return True
    if metric == "val_loss":
        return value < best_value - min_delta
    return value > best_value + min_delta


def _checkpoint_payload(
    *,
    epoch: int,
    model: nn.Module,
    run_config: dict[str, Any],
    history: dict[str, Any],
    train_metrics: EpochMetrics,
    validation_metrics: EpochMetrics,
    monitored_value: float,
    best_value: float | None,
    metric: str,
    is_best: bool,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "run_config": run_config,
        "selection": {
            "metric": metric,
            "direction": "min" if metric == "val_loss" else "max",
            "value": monitored_value,
            "best_value": best_value,
            "is_best": is_best,
        },
        "train_metrics": train_metrics.as_dict(),
        "validation_metrics": validation_metrics.as_dict(),
        "training_history": history,
    }


def _prepare_output(checkpoint_dir: Path) -> None:
    existing_epochs = tuple((checkpoint_dir / "epochs").glob("cp_epoch_*.pt"))
    if existing_epochs or (checkpoint_dir / "best.pt").exists():
        raise FileExistsError(
            f"checkpoint directory already contains a run: {checkpoint_dir}; "
            "choose another --checkpoint-dir"
        )
    (checkpoint_dir / "epochs").mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "plots").mkdir(parents=True, exist_ok=True)
