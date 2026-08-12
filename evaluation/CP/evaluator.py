"""Inference loop and report model for CP evaluation."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from data import DataSplit
from metrics import BinaryClassificationMetrics, binary_classification_metrics
from training.CP import FeatureMode

LOGGER = logging.getLogger("evaluation.CP")


class BatchLoader(Protocol):
    """Minimal loader contract needed by the evaluation loop."""

    def __iter__(self) -> Iterator[Any]: ...

    def __len__(self) -> int: ...


@dataclass(frozen=True)
class CPEvaluationResult:
    """Serializable summary of one checkpoint evaluation."""

    checkpoint: Path
    checkpoint_epoch: int
    output_path: Path
    split: DataSplit
    dataset_name: str
    dataset_id: str
    subset: str
    feature_mode: FeatureMode
    threshold: float
    metrics: BinaryClassificationMetrics

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint),
            "checkpoint_epoch": self.checkpoint_epoch,
            "dataset": self.dataset_name,
            "dataset_id": self.dataset_id,
            "split": self.split.value,
            "subset": self.subset,
            "feature_mode": self.feature_mode.value,
            "threshold": self.threshold,
            "metrics": self.metrics.as_dict(),
        }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: BatchLoader,
    device: torch.device,
    *,
    threshold: float = 0.5,
    log_every: int = 10,
) -> BinaryClassificationMetrics:
    """Run CP inference and compute metrics over the complete split."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if log_every <= 0:
        raise ValueError("log_every must be positive")
    if len(loader) == 0:
        raise ValueError("evaluation loader cannot be empty")

    model.to(device)
    model.eval()
    probabilities: list[Tensor] = []
    targets: list[Tensor] = []
    batch_count = len(loader)
    for batch_index, batch in enumerate(loader, start=1):
        batch_probabilities = model(batch.outfits)
        batch_targets = batch.labels
        if batch_probabilities.shape != batch_targets.shape:
            raise ValueError(
                "model probabilities and batch labels must have equal shape"
            )
        probabilities.append(batch_probabilities.detach().flatten().to("cpu"))
        targets.append(batch_targets.detach().flatten().to("cpu"))
        if batch_index % log_every == 0 or batch_index == batch_count:
            LOGGER.info("batch=%d/%d", batch_index, batch_count)

    return binary_classification_metrics(
        torch.cat(probabilities),
        torch.cat(targets),
        threshold=threshold,
    )
