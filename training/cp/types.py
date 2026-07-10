from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class CPEpochMetrics:
    loss: float
    accuracy: float
    examples: int
    auc: float | None = None

    def to_payload(self) -> dict[str, float | int]:
        payload: dict[str, float | int] = {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "examples": self.examples,
        }
        if self.auc is not None:
            payload["auc"] = self.auc
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CPEpochMetrics":
        loss = _numeric_value(payload, "loss")
        accuracy = _numeric_value(payload, "accuracy")
        examples = payload.get("examples")
        if not isinstance(examples, int):
            raise ValueError("metrics payload missing integer examples")
        auc_value = payload.get("auc")
        if auc_value is not None and not isinstance(auc_value, int | float):
            raise ValueError("metrics payload auc must be numeric or None")
        return cls(
            loss=float(loss),
            accuracy=float(accuracy),
            examples=examples,
            auc=float(auc_value) if auc_value is not None else None,
        )


@dataclass(frozen=True)
class CPTrainingHistory:
    train: tuple[CPEpochMetrics, ...] = ()
    validation: tuple[CPEpochMetrics, ...] = ()
    epochs: tuple[int, ...] = ()
    validation_epochs: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        epochs = self.epochs
        if not epochs and self.train:
            epochs = tuple(range(1, len(self.train) + 1))
            object.__setattr__(self, "epochs", epochs)

        validation_epochs = self.validation_epochs
        if not validation_epochs and self.validation:
            if len(self.validation) != len(self.train):
                raise ValueError(
                    "validation epochs are required when history lengths differ"
                )
            validation_epochs = epochs
            object.__setattr__(self, "validation_epochs", validation_epochs)

        if len(epochs) != len(self.train):
            raise ValueError("epochs and train metrics must have the same length")
        if len(validation_epochs) != len(self.validation):
            raise ValueError(
                "validation epochs and metrics must have the same length"
            )
        if any(epoch <= 0 for epoch in (*epochs, *validation_epochs)):
            raise ValueError("history epochs must be positive")
        if tuple(sorted(epochs)) != epochs or len(set(epochs)) != len(epochs):
            raise ValueError("training epochs must be strictly increasing")
        if (
            tuple(sorted(validation_epochs)) != validation_epochs
            or len(set(validation_epochs)) != len(validation_epochs)
        ):
            raise ValueError("validation epochs must be strictly increasing")
        if not set(validation_epochs).issubset(epochs):
            raise ValueError("validation epochs must belong to training history")

    @property
    def last_epoch(self) -> int | None:
        return self.epochs[-1] if self.epochs else None

    def append(
        self,
        epoch: int,
        train_metrics: CPEpochMetrics,
        validation_metrics: CPEpochMetrics | None,
    ) -> "CPTrainingHistory":
        if epoch <= 0:
            raise ValueError("epoch must be positive")
        if self.last_epoch is not None and epoch <= self.last_epoch:
            raise ValueError("history epochs must be appended in increasing order")

        validation = self.validation
        validation_epochs = self.validation_epochs
        if validation_metrics is not None:
            validation = (*validation, validation_metrics)
            validation_epochs = (*validation_epochs, epoch)

        return CPTrainingHistory(
            train=(*self.train, train_metrics),
            validation=validation,
            epochs=(*self.epochs, epoch),
            validation_epochs=validation_epochs,
        )

    def to_payload(self) -> dict[str, list[Any]]:
        return {
            "epochs": list(self.epochs),
            "train": [metrics.to_payload() for metrics in self.train],
            "validation_epochs": list(self.validation_epochs),
            "validation": [
                metrics.to_payload() for metrics in self.validation
            ],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CPTrainingHistory":
        epochs = _integer_sequence(payload, "epochs")
        validation_epochs = _integer_sequence(payload, "validation_epochs")
        train = _metrics_sequence(payload, "train")
        validation = _metrics_sequence(payload, "validation")
        return cls(
            train=train,
            validation=validation,
            epochs=epochs,
            validation_epochs=validation_epochs,
        )


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
    selection_metric: str
    selection_source: str
    selection_value: float
    best_selection_value: float

    @property
    def monitored_loss(self) -> float:
        """Legacy alias; accurate when selection_metric is val_loss."""
        return self.selection_value


def _numeric_value(payload: Mapping[str, Any], key: str) -> int | float:
    value = payload.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"metrics payload missing numeric {key}")
    return value


def _integer_sequence(payload: Mapping[str, Any], key: str) -> tuple[int, ...]:
    values = payload.get(key, ())
    if not isinstance(values, list | tuple) or not all(
        isinstance(value, int) for value in values
    ):
        raise ValueError(f"history payload {key} must contain integers")
    return tuple(values)


def _metrics_sequence(
    payload: Mapping[str, Any],
    key: str,
) -> tuple[CPEpochMetrics, ...]:
    values = payload.get(key, ())
    if not isinstance(values, list | tuple):
        raise ValueError(f"history payload {key} must be a sequence")
    metrics: list[CPEpochMetrics] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError(f"history payload {key} contains invalid metrics")
        metrics.append(CPEpochMetrics.from_payload(value))
    return tuple(metrics)
