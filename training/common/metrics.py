"""Epoch-level binary metrics shared by classification training jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

from metrics import binary_roc_auc


@dataclass(frozen=True)
class EpochMetrics:
    """Loss and observable binary classification metrics for one epoch."""

    loss: float
    accuracy: float
    auc: float
    examples: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class BinaryEpochAccumulator:
    """Accumulate probability-based binary metrics without keeping graphs."""

    def __init__(self) -> None:
        self._loss_sum = 0.0
        self._examples = 0
        self._correct = 0
        self._probabilities: list[Tensor] = []
        self._targets: list[Tensor] = []

    def update(
        self,
        loss: Tensor,
        probabilities: Tensor,
        targets: Tensor,
    ) -> None:
        if probabilities.shape != targets.shape:
            raise ValueError("probabilities and targets must have equal shape")
        if probabilities.numel() == 0:
            raise ValueError("probabilities and targets cannot be empty")
        if loss.numel() != 1 or not bool(torch.isfinite(loss.detach())):
            raise ValueError("loss must be one finite scalar")

        detached_probabilities = probabilities.detach().flatten().to("cpu")
        detached_targets = targets.detach().flatten().to("cpu")
        if not bool(torch.isfinite(detached_probabilities).all()):
            raise ValueError("probabilities must contain only finite values")
        if bool(
            ((detached_probabilities < 0.0) | (detached_probabilities > 1.0)).any()
        ):
            raise ValueError("probabilities must be in [0, 1]")
        if not bool(((detached_targets == 0) | (detached_targets == 1)).all()):
            raise ValueError("targets must contain only 0 and 1")

        examples = detached_targets.numel()
        predictions = detached_probabilities >= 0.5
        target_classes = detached_targets >= 0.5
        self._loss_sum += float(loss.detach().item()) * examples
        self._examples += examples
        self._correct += int((predictions == target_classes).sum().item())
        self._probabilities.append(detached_probabilities)
        self._targets.append(detached_targets)

    def compute(self) -> EpochMetrics:
        if self._examples == 0:
            raise ValueError("epoch metrics require at least one example")
        probabilities = torch.cat(self._probabilities)
        targets = torch.cat(self._targets)
        return EpochMetrics(
            loss=self._loss_sum / self._examples,
            accuracy=self._correct / self._examples,
            auc=binary_roc_auc(probabilities, targets),
            examples=self._examples,
        )

