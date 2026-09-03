"""Candidate-ranking metrics for Complementary Item Retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class RetrievalMetrics:
    """FITB accuracy, mean reciprocal rank and recall at two."""

    fitb_accuracy: float
    mrr: float
    recall_at_2: float
    examples: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "fitb_accuracy": self.fitb_accuracy,
            "mrr": self.mrr,
            "recall_at_2": self.recall_at_2,
            "examples": self.examples,
        }


def retrieval_rank(
    candidate_distances: Tensor,
    *,
    positive_index: int = 0,
) -> int:
    """Return the one-based rank of the positive candidate.

    Smaller distances rank first. Exact ties are resolved conservatively in
    favour of competing candidates, avoiding optimistic FITB scores when a
    model collapses to identical vectors.
    """
    _validate_candidate_distances(candidate_distances, positive_index)
    positive_distance = candidate_distances[positive_index]
    competitor_mask = torch.ones(
        candidate_distances.numel(),
        device=candidate_distances.device,
        dtype=torch.bool,
    )
    competitor_mask[positive_index] = False
    competitors = candidate_distances[competitor_mask]
    return 1 + int((competitors <= positive_distance).sum().item())


def fitb_accuracy(ranks: Tensor) -> float:
    """Return fraction of examples whose positive candidate ranks first."""
    validated = _validate_ranks(ranks)
    return float((validated == 1).to(torch.float64).mean().item())


def mean_reciprocal_rank(ranks: Tensor) -> float:
    """Return mean inverse rank of positive candidates."""
    validated = _validate_ranks(ranks)
    return float((1.0 / validated.to(torch.float64)).mean().item())


def recall_at_k(ranks: Tensor, k: int) -> float:
    """Return fraction of positive candidates ranked within the first ``k``."""
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    validated = _validate_ranks(ranks)
    return float((validated <= k).to(torch.float64).mean().item())


def retrieval_metrics(ranks: Tensor) -> RetrievalMetrics:
    """Compute the three CIR metrics requested for FITB evaluation."""
    validated = _validate_ranks(ranks)
    return RetrievalMetrics(
        fitb_accuracy=fitb_accuracy(validated),
        mrr=mean_reciprocal_rank(validated),
        recall_at_2=recall_at_k(validated, 2),
        examples=validated.numel(),
    )


def _validate_candidate_distances(
    distances: Tensor,
    positive_index: int,
) -> None:
    if distances.ndim != 1:
        raise ValueError("candidate_distances must be one-dimensional")
    if distances.numel() < 2:
        raise ValueError("at least two candidate distances are required")
    if not torch.is_floating_point(distances):
        raise TypeError("candidate_distances must be floating point")
    if not bool(torch.isfinite(distances).all()):
        raise ValueError("candidate_distances must contain only finite values")
    if (
        not isinstance(positive_index, int)
        or isinstance(positive_index, bool)
        or not 0 <= positive_index < distances.numel()
    ):
        raise ValueError("positive_index is outside candidate_distances")


def _validate_ranks(ranks: Tensor) -> Tensor:
    if ranks.ndim != 1 or ranks.numel() == 0:
        raise ValueError("ranks must be a non-empty one-dimensional tensor")
    if ranks.dtype == torch.bool or torch.is_floating_point(ranks):
        raise TypeError("ranks must use an integer dtype")
    if bool((ranks <= 0).any()):
        raise ValueError("ranks must contain positive integers")
    return ranks
