from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .types import CPEpochMetrics, CPTrainingHistory


CPBestMetric = Literal["val_loss", "val_accuracy", "val_auc"]
CP_BEST_METRICS: tuple[CPBestMetric, ...] = (
    "val_loss",
    "val_accuracy",
    "val_auc",
)


@dataclass(frozen=True)
class CPSelectionCriterion:
    """Select the best validation epoch using one configured metric."""

    metric: CPBestMetric = "val_loss"

    def __post_init__(self) -> None:
        if self.metric not in CP_BEST_METRICS:
            raise ValueError(f"metric must be one of {CP_BEST_METRICS}")

    @property
    def direction(self) -> Literal["min", "max"]:
        return "min" if self.metric == "val_loss" else "max"

    @property
    def initial_best_value(self) -> float:
        return float("inf") if self.direction == "min" else float("-inf")

    def value(self, metrics: CPEpochMetrics) -> float:
        if self.metric == "val_loss":
            return metrics.loss
        if self.metric == "val_accuracy":
            return metrics.accuracy
        if metrics.auc is None:
            raise ValueError("val_auc selection requires validation AUC")
        return metrics.auc

    def is_better(
        self,
        current: float,
        best: float,
        *,
        min_delta: float = 0.0,
    ) -> bool:
        if min_delta < 0.0:
            raise ValueError("min_delta must be non-negative")
        if self.direction == "min":
            return current < best - min_delta
        return current > best + min_delta

    def best_value(self, history: CPTrainingHistory) -> float:
        if not history.validation:
            return self.initial_best_value
        values = [self.value(metrics) for metrics in history.validation]
        return min(values) if self.direction == "min" else max(values)
