from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from data import CompatibilityBatch
from .checkpointing import CPCheckpointManager
from .epoch import BatchProgressCallback, run_cp_epoch
from .selection import CPBestMetric, CPSelectionCriterion
from .types import (
    CPBatchProgress,
    CPCheckpointInfo,
    CPEpochMetrics,
    CPTrainingHistory,
)


EpochCallback = Callable[
    [int, CPEpochMetrics, CPEpochMetrics | None],
    None,
]
CheckpointCallback = Callable[[CPCheckpointInfo], None]
HistoryCallback = Callable[[CPTrainingHistory], Any]
EpochRunner = Callable[..., CPEpochMetrics]


@dataclass(frozen=True)
class CPTrainerConfig:
    epochs: int
    device: torch.device | str
    start_epoch: int = 1
    max_grad_norm: float | None = None
    progress_interval: int | None = None

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.start_epoch <= 0:
            raise ValueError("start_epoch must be positive")
        if self.start_epoch > self.epochs:
            raise ValueError("start_epoch must be less than or equal to epochs")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive or None")
        if self.progress_interval is not None and self.progress_interval <= 0:
            raise ValueError("progress_interval must be positive or None")


@dataclass(frozen=True)
class CPTrainingCallbacks:
    on_epoch_end: EpochCallback | None = None
    on_batch_end: BatchProgressCallback | None = None
    on_checkpoint_saved: CheckpointCallback | None = None
    on_history_updated: HistoryCallback | None = None


class CPTrainer:
    """Orchestrate CP phases using replaceable training components."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        scheduler: Any | None = None,
        epoch_runner: EpochRunner = run_cp_epoch,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._scheduler = scheduler
        self._epoch_runner = epoch_runner

    def fit(
        self,
        train_batches: Iterable[CompatibilityBatch],
        *,
        config: CPTrainerConfig,
        validation_batches: Iterable[CompatibilityBatch] | None = None,
        checkpoint_manager: CPCheckpointManager | None = None,
        initial_history: CPTrainingHistory | None = None,
        callbacks: CPTrainingCallbacks | None = None,
    ) -> CPTrainingHistory:
        config.validate()
        history = initial_history or CPTrainingHistory()
        _validate_initial_history(history, config.start_epoch)
        callbacks = callbacks or CPTrainingCallbacks()
        self._model.to(config.device)

        for epoch in range(config.start_epoch, config.epochs + 1):
            train_metrics = self._run_training_epoch(
                train_batches,
                epoch,
                config,
                callbacks,
            )
            validation_metrics = self._run_validation_epoch(
                validation_batches,
                epoch,
                config,
                callbacks,
            )

            if self._scheduler is not None:
                self._scheduler.step()

            history = history.append(
                epoch,
                train_metrics,
                validation_metrics,
            )
            if checkpoint_manager is not None:
                saved_checkpoints = checkpoint_manager.save(
                    epoch=epoch,
                    model=self._model,
                    optimizer=self._optimizer,
                    scheduler=self._scheduler,
                    train_metrics=train_metrics,
                    validation_metrics=validation_metrics,
                    history=history,
                )
                if callbacks.on_checkpoint_saved is not None:
                    for checkpoint in saved_checkpoints:
                        callbacks.on_checkpoint_saved(checkpoint)

            if callbacks.on_history_updated is not None:
                callbacks.on_history_updated(history)
            if callbacks.on_epoch_end is not None:
                callbacks.on_epoch_end(
                    epoch,
                    train_metrics,
                    validation_metrics,
                )

        return history

    def _run_training_epoch(
        self,
        batches: Iterable[CompatibilityBatch],
        epoch: int,
        config: CPTrainerConfig,
        callbacks: CPTrainingCallbacks,
    ) -> CPEpochMetrics:
        return self._epoch_runner(
            self._model,
            batches,
            self._criterion,
            config.device,
            optimizer=self._optimizer,
            max_grad_norm=config.max_grad_norm,
            epoch=epoch,
            phase="train",
            progress_interval=config.progress_interval,
            on_batch_end=callbacks.on_batch_end,
            calculate_auc=True,
        )

    def _run_validation_epoch(
        self,
        batches: Iterable[CompatibilityBatch] | None,
        epoch: int,
        config: CPTrainerConfig,
        callbacks: CPTrainingCallbacks,
    ) -> CPEpochMetrics | None:
        if batches is None:
            return None
        return self._epoch_runner(
            self._model,
            batches,
            self._criterion,
            config.device,
            epoch=epoch,
            phase="validation",
            progress_interval=config.progress_interval,
            on_batch_end=callbacks.on_batch_end,
            calculate_auc=True,
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
    start_epoch: int = 1,
    best_metric: CPBestMetric = "val_loss",
    initial_best_value: float | None = None,
    initial_best_loss: float | None = None,
    initial_history: CPTrainingHistory | None = None,
    checkpoint_metadata: Mapping[str, Any] | None = None,
    progress_interval: int | None = None,
    on_epoch_end: EpochCallback | None = None,
    on_batch_end: BatchProgressCallback | None = None,
    on_checkpoint_saved: CheckpointCallback | None = None,
    on_history_updated: HistoryCallback | None = None,
) -> CPTrainingHistory:
    """Convenience API composing the modular CP training components."""
    if initial_best_value is not None and initial_best_loss is not None:
        raise ValueError(
            "provide either initial_best_value or initial_best_loss, not both"
        )
    resolved_best_value = (
        initial_best_value
        if initial_best_value is not None
        else initial_best_loss
    )
    checkpoint_manager = None
    if checkpoint_path is not None or epoch_checkpoint_dir is not None:
        checkpoint_manager = CPCheckpointManager(
            best_path=checkpoint_path,
            epoch_directory=epoch_checkpoint_dir,
            total_epochs=epochs,
            selection_criterion=CPSelectionCriterion(best_metric),
            initial_best_value=resolved_best_value,
            run_config=checkpoint_metadata,
        )

    trainer = CPTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
    )
    return trainer.fit(
        train_batches,
        validation_batches=validation_batches,
        config=CPTrainerConfig(
            epochs=epochs,
            device=device,
            start_epoch=start_epoch,
            max_grad_norm=max_grad_norm,
            progress_interval=progress_interval,
        ),
        checkpoint_manager=checkpoint_manager,
        initial_history=initial_history,
        callbacks=CPTrainingCallbacks(
            on_epoch_end=on_epoch_end,
            on_batch_end=on_batch_end,
            on_checkpoint_saved=on_checkpoint_saved,
            on_history_updated=on_history_updated,
        ),
    )


def _validate_initial_history(
    history: CPTrainingHistory,
    start_epoch: int,
) -> None:
    if history.last_epoch is None:
        return
    expected_last_epoch = start_epoch - 1
    if history.last_epoch != expected_last_epoch:
        raise ValueError(
            "initial history must end immediately before start_epoch"
        )


__all__ = [
    "CPBatchProgress",
    "CPCheckpointInfo",
    "CPEpochMetrics",
    "CPTrainer",
    "CPTrainerConfig",
    "CPTrainingCallbacks",
    "CPTrainingHistory",
    "run_cp_epoch",
    "train_cp",
]
