from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from data import CompatibilityBatch
from metrics import BinaryAccuracy, binary_roc_auc
from .types import CPBatchProgress, CPEpochMetrics


BatchProgressCallback = Callable[[CPBatchProgress], None]


@dataclass
class CPEpochAccumulator:
    calculate_auc: bool = False
    total_loss: float = 0.0
    examples: int = 0
    _accuracy: BinaryAccuracy = field(default_factory=BinaryAccuracy)
    _auc_scores: list[Tensor] = field(default_factory=list)
    _auc_targets: list[Tensor] = field(default_factory=list)

    @property
    def running_loss(self) -> float:
        if self.examples == 0:
            raise ValueError("running loss requires at least one example")
        return self.total_loss / self.examples

    @property
    def running_accuracy(self) -> float:
        return self._accuracy.compute()

    def update(self, logits: Tensor, targets: Tensor, loss: Tensor) -> None:
        example_count = targets.numel()
        self.total_loss += loss.detach().item() * example_count
        self._accuracy.update(logits, targets)
        self.examples += example_count
        if self.calculate_auc:
            self._auc_scores.append(logits.detach().reshape(-1).cpu())
            self._auc_targets.append(
                (targets.detach() >= 0.5).reshape(-1).cpu()
            )

    def compute(self) -> CPEpochMetrics:
        if self.examples == 0:
            raise ValueError("CP data loader produced no examples")
        auc = None
        if self.calculate_auc:
            auc = binary_roc_auc(
                torch.cat(self._auc_scores),
                torch.cat(self._auc_targets),
            )
        return CPEpochMetrics(
            loss=self.running_loss,
            accuracy=self.running_accuracy,
            examples=self.examples,
            auc=auc,
        )


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
    calculate_auc: bool = False,
) -> CPEpochMetrics:
    """Run one CP phase; passing an optimizer enables parameter updates."""
    _validate_epoch_options(max_grad_norm, progress_interval)
    is_training = optimizer is not None
    model.train(is_training)
    total_batches = _safe_len(batches)
    accumulator = CPEpochAccumulator(calculate_auc=calculate_auc)

    with torch.set_grad_enabled(is_training):
        for batch_index, batch in enumerate(batches, start=1):
            if not isinstance(batch, CompatibilityBatch):
                raise TypeError(
                    "CP data loader must return CompatibilityBatch instances"
                )
            batch = batch.to(device)
            if optimizer is not None:
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

            if optimizer is not None:
                loss.backward()
                if max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            accumulator.update(logits, batch.labels, loss)
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
                        running_loss=accumulator.running_loss,
                        running_accuracy=accumulator.running_accuracy,
                        examples=accumulator.examples,
                    )
                )

    return accumulator.compute()


def _validate_epoch_options(
    max_grad_norm: float | None,
    progress_interval: int | None,
) -> None:
    if max_grad_norm is not None and max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive or None")
    if progress_interval is not None and progress_interval <= 0:
        raise ValueError("progress_interval must be positive or None")


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
    if batch == 1 or (batches is not None and batch == batches):
        return True
    return batch % progress_interval == 0
