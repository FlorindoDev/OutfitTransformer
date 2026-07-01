"""Classification metrics shared by training and evaluation."""

import torch
from torch import Tensor


class BinaryAccuracy:
    """Accumulate threshold-based accuracy for binary logits."""

    def __init__(self) -> None:
        self._correct = 0
        self._examples = 0

    def update(self, logits: Tensor, targets: Tensor) -> None:
        if logits.shape != targets.shape:
            raise ValueError("logits and targets must have the same shape")

        predictions = logits.detach() >= 0.0
        target_classes = targets.detach() >= 0.5
        self._correct += int((predictions == target_classes).sum().item())
        self._examples += targets.numel()

    def compute(self) -> float:
        if self._examples == 0:
            raise ValueError("binary accuracy requires at least one example")
        return self._correct / self._examples


def binary_roc_auc(scores: Tensor, targets: Tensor) -> float:
    """Return the tie-aware ROC AUC for binary targets."""
    if scores.ndim != 1 or targets.ndim != 1:
        raise ValueError("scores and targets must be one-dimensional")
    if scores.shape != targets.shape:
        raise ValueError("scores and targets must have the same shape")
    if scores.numel() == 0:
        raise ValueError("scores and targets cannot be empty")

    scores = scores.detach().to(device="cpu", dtype=torch.float64)
    targets = targets.detach().to(device="cpu")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must contain only finite values")
    if not bool(((targets == 0) | (targets == 1)).all()):
        raise ValueError("targets must contain only 0 and 1")

    positive_mask = targets == 1
    positive_count = int(positive_mask.sum().item())
    negative_count = targets.numel() - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC AUC requires both positive and negative examples")

    sorted_scores, order = torch.sort(scores)
    sorted_positive_mask = positive_mask[order]
    _, tie_counts = torch.unique_consecutive(
        sorted_scores,
        return_counts=True,
    )

    tie_counts_float = tie_counts.to(dtype=torch.float64)
    end_ranks = torch.cumsum(tie_counts_float, dim=0)
    start_ranks = end_ranks - tie_counts_float + 1.0
    average_ranks = (start_ranks + end_ranks) / 2.0
    ranks = torch.repeat_interleave(average_ranks, tie_counts)

    positive_rank_sum = ranks[sorted_positive_mask].sum()
    minimum_positive_rank_sum = positive_count * (positive_count + 1) / 2
    auc = (
        positive_rank_sum.item() - minimum_positive_rank_sum
    ) / (positive_count * negative_count)
    return float(auc)
