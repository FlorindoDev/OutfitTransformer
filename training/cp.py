from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from data import CompatibilityBatch


@dataclass(frozen=True)
class CPEpochMetrics:
    loss: float
    accuracy: float
    examples: int


@dataclass(frozen=True)
class CPTrainingHistory:
    train: tuple[CPEpochMetrics, ...]
    validation: tuple[CPEpochMetrics, ...]


@dataclass(frozen=True)
class CPBatchProgress:
    epoch: int
    phase: str
    batch: int
    batches: int | None
    loss: float
    running_loss: float
    running_accuracy: float
    examples: int


@dataclass(frozen=True)
class CPCheckpointInfo:
    epoch: int
    kind: str
    path: Path
    monitored_loss: float


EpochCallback = Callable[
    [int, CPEpochMetrics, CPEpochMetrics | None],
    None,
]
BatchProgressCallback = Callable[[CPBatchProgress], None]
CheckpointCallback = Callable[[CPCheckpointInfo], None]


def run_cp_epoch(
    model: nn.Module,
    batches: Iterable[CompatibilityBatch],
    criterion: nn.Module,
    device: torch.device | str,
    *,
    optimizer: Optimizer | None = None,
    max_grad_norm: float | None = None,
    epoch: int = 0,
    phase: str = "train",
    progress_interval: int | None = None,
    on_batch_end: BatchProgressCallback | None = None,
) -> CPEpochMetrics:
    """Run one CP epoch; an optimizer switches evaluation to training."""
    if max_grad_norm is not None and max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive or None")
    if progress_interval is not None and progress_interval <= 0:
        raise ValueError("progress_interval must be positive or None")

    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_batches = _safe_len(batches)

    with torch.set_grad_enabled(is_training):
        for batch_index, batch in enumerate(batches, start=1):
            if not isinstance(batch, CompatibilityBatch):
                raise TypeError(
                    "CP data loader must return CompatibilityBatch instances"
                )
            batch = batch.to(device)
            if is_training:
                optimizer.zero_grad(set_to_none=True)

            output = model(
                batch.images,
                batch.descriptions,
                batch.padding_mask,
            )
            logits = output.logits
            loss = criterion(logits, batch.labels)
            if loss.ndim != 0:
                raise ValueError("CP criterion must return a scalar loss")

            if is_training:
                loss.backward()
                if max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            example_count = batch.labels.numel()
            total_loss += loss.detach().item() * example_count
            predictions = logits.detach() >= 0.0
            targets = batch.labels >= 0.5
            total_correct += int((predictions == targets).sum().item())
            total_examples += example_count
            if _should_report_progress(
                batch_index,
                total_batches,
                progress_interval,
            ) and on_batch_end is not None:
                on_batch_end(
                    CPBatchProgress(
                        epoch=epoch,
                        phase=phase,
                        batch=batch_index,
                        batches=total_batches,
                        loss=loss.detach().item(),
                        running_loss=total_loss / total_examples,
                        running_accuracy=total_correct / total_examples,
                        examples=total_examples,
                    )
                )

    if total_examples == 0:
        raise ValueError("CP data loader produced no examples")
    return CPEpochMetrics(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
        examples=total_examples,
    )


def train_cp(
    model: nn.Module,
    train_batches: Iterable[CompatibilityBatch],
    optimizer: Optimizer,
    criterion: nn.Module,
    *,
    epochs: int,
    device: torch.device | str,
    validation_batches: Iterable[CompatibilityBatch] | None = None,
    scheduler: Any | None = None,
    max_grad_norm: float | None = None,
    checkpoint_path: str | Path | None = None,
    epoch_checkpoint_dir: str | Path | None = None,
    progress_interval: int | None = None,
    on_epoch_end: EpochCallback | None = None,
    on_batch_end: BatchProgressCallback | None = None,
    on_checkpoint_saved: CheckpointCallback | None = None,
) -> CPTrainingHistory:
    """Train OutfitTransformer only on compatibility prediction."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if progress_interval is not None and progress_interval <= 0:
        raise ValueError("progress_interval must be positive or None")

    model.to(device)
    train_history: list[CPEpochMetrics] = []
    validation_history: list[CPEpochMetrics] = []
    best_loss = float("inf")
    best_checkpoint_path = (
        Path(checkpoint_path) if checkpoint_path is not None else None
    )
    epoch_checkpoint_directory = (
        Path(epoch_checkpoint_dir) if epoch_checkpoint_dir is not None else None
    )

    for epoch in range(1, epochs + 1):
        train_metrics = run_cp_epoch(
            model,
            train_batches,
            criterion,
            device,
            optimizer=optimizer,
            max_grad_norm=max_grad_norm,
            epoch=epoch,
            phase="train",
            progress_interval=progress_interval,
            on_batch_end=on_batch_end,
        )
        train_history.append(train_metrics)

        validation_metrics = None
        if validation_batches is not None:
            validation_metrics = run_cp_epoch(
                model,
                validation_batches,
                criterion,
                device,
                epoch=epoch,
                phase="validation",
                progress_interval=progress_interval,
                on_batch_end=on_batch_end,
            )
            validation_history.append(validation_metrics)

        if scheduler is not None:
            scheduler.step()

        monitored_loss = (
            validation_metrics.loss
            if validation_metrics is not None
            else train_metrics.loss
        )
        if epoch_checkpoint_directory is not None:
            epoch_checkpoint_path = _epoch_checkpoint_path(
                epoch_checkpoint_directory,
                epoch,
                epochs,
            )
            _save_checkpoint(
                path=epoch_checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                monitored_loss=monitored_loss,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
            )
            _notify_checkpoint_saved(
                on_checkpoint_saved,
                epoch=epoch,
                kind="epoch",
                path=epoch_checkpoint_path,
                monitored_loss=monitored_loss,
            )

        if best_checkpoint_path is not None and monitored_loss < best_loss:
            best_loss = monitored_loss
            _save_checkpoint(
                path=best_checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                monitored_loss=monitored_loss,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
            )
            _notify_checkpoint_saved(
                on_checkpoint_saved,
                epoch=epoch,
                kind="best",
                path=best_checkpoint_path,
                monitored_loss=monitored_loss,
            )

        if on_epoch_end is not None:
            on_epoch_end(epoch, train_metrics, validation_metrics)

    return CPTrainingHistory(
        train=tuple(train_history),
        validation=tuple(validation_history),
    )


def _save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any | None,
    monitored_loss: float,
    train_metrics: CPEpochMetrics | None = None,
    validation_metrics: CPEpochMetrics | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "monitored_loss": monitored_loss,
        "train_metrics": _metrics_payload(train_metrics),
        "validation_metrics": _metrics_payload(validation_metrics),
    }
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def _safe_len(batches: Iterable[CompatibilityBatch]) -> int | None:
    try:
        return len(batches)  # type: ignore[arg-type]
    except TypeError:
        return None


def _should_report_progress(
    batch: int,
    batches: int | None,
    progress_interval: int | None,
) -> bool:
    if progress_interval is None:
        return False
    if batch == 1:
        return True
    if batches is not None and batch == batches:
        return True
    return batch % progress_interval == 0


def _epoch_checkpoint_path(directory: Path, epoch: int, epochs: int) -> Path:
    width = max(3, len(str(epochs)))
    return directory / f"cp_epoch_{epoch:0{width}d}.pt"


def _metrics_payload(metrics: CPEpochMetrics | None) -> dict[str, float | int] | None:
    if metrics is None:
        return None
    return {
        "loss": metrics.loss,
        "accuracy": metrics.accuracy,
        "examples": metrics.examples,
    }


def _notify_checkpoint_saved(
    callback: CheckpointCallback | None,
    *,
    epoch: int,
    kind: str,
    path: Path,
    monitored_loss: float,
) -> None:
    if callback is None:
        return
    callback(
        CPCheckpointInfo(
            epoch=epoch,
            kind=kind,
            path=path,
            monitored_loss=monitored_loss,
        )
    )
