from __future__ import annotations

from dataclasses import dataclass

from .selection import CPBestMetric, CPSelectionCriterion
from .types import CPEpochMetrics, CPTrainingHistory


@dataclass(frozen=True)
class CPEarlyStoppingConfig:
    """Stop CP training after validation metric stops improving."""

    metric: CPBestMetric
    patience: int
    min_delta: float = 0.0

    def validate(self) -> None:
        CPSelectionCriterion(self.metric)
        if self.patience <= 0:
            raise ValueError("early stopping patience must be positive")
        if self.min_delta < 0.0:
            raise ValueError("early stopping min_delta must be non-negative")


@dataclass(frozen=True)
class CPEarlyStoppingStatus:
    epoch: int
    metric: CPBestMetric
    value: float
    best_value: float
    epochs_without_improvement: int
    patience: int

    @property
    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience


class CPEarlyStopper:
    """Track validation progress without depending on training I/O."""

    def __init__(
        self,
        config: CPEarlyStoppingConfig,
        history: CPTrainingHistory | None = None,
    ) -> None:
        config.validate()
        self._config = config
        self._criterion = CPSelectionCriterion(config.metric)
        self._best_value = self._criterion.initial_best_value
        self._epochs_without_improvement = 0
        if history is not None:
            self._restore_history(history)

    def observe(
        self,
        epoch: int,
        validation_metrics: CPEpochMetrics,
    ) -> CPEarlyStoppingStatus:
        value = self._criterion.value(validation_metrics)
        self._record(value)
        return CPEarlyStoppingStatus(
            epoch=epoch,
            metric=self._config.metric,
            value=value,
            best_value=self._best_value,
            epochs_without_improvement=self._epochs_without_improvement,
            patience=self._config.patience,
        )

    def _restore_history(self, history: CPTrainingHistory) -> None:
        for metrics in history.validation:
            self._record(self._criterion.value(metrics))

    def _record(self, value: float) -> None:
        if self._criterion.is_better(
            value,
            self._best_value,
            min_delta=self._config.min_delta,
        ):
            self._best_value = value
            self._epochs_without_improvement = 0
            return
        self._epochs_without_improvement += 1


def create_early_stopping_config(
    *,
    metric: CPBestMetric,
    patience: int | None,
    min_delta: float,
) -> CPEarlyStoppingConfig | None:
    """Build optional early stopping and reject ambiguous CLI combinations."""
    if patience is None:
        if min_delta != 0.0:
            raise ValueError(
                "early stopping min_delta requires early stopping patience"
            )
        return None

    config = CPEarlyStoppingConfig(
        metric=metric,
        patience=patience,
        min_delta=min_delta,
    )
    config.validate()
    return config


__all__ = [
    "CPEarlyStopper",
    "CPEarlyStoppingConfig",
    "CPEarlyStoppingStatus",
    "create_early_stopping_config",
]
