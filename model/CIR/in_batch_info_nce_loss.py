"""Contrastive training objectives for Complementary Item Retrieval."""

import math

from torch import Tensor, nn
from torch.nn import functional as F

from ..common.config import DEFAULT_CIR_CONFIG, Reduction
from .progressive_in_batch_triplet_margin_loss import (
    _reduce_view_losses,
    _triplet_view_losses,
    _view_distances,
)

DEFAULT_INFO_NCE_TEMPERATURE = 1.0
DEFAULT_JOINT_ALPHA = 0.5


class InBatchInfoNCELoss(nn.Module):
    """Classify each query's target among all in-batch targets."""

    def __init__(
        self,
        temperature: float = DEFAULT_INFO_NCE_TEMPERATURE,
        reduction: Reduction = DEFAULT_CIR_CONFIG.loss_reduction,
    ) -> None:
        super().__init__()
        _validate_info_nce_settings(temperature, reduction)
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        query_embeddings: Tensor,
        positive_embeddings: Tensor,
        query_positive_indices: Tensor,
    ) -> Tensor:
        distances, view_counts = _view_distances(
            query_embeddings, positive_embeddings, query_positive_indices,
        )
        view_losses = _info_nce_view_losses(
            distances, query_positive_indices, self.temperature,
        )
        return _reduce_view_losses(
            view_losses, query_positive_indices, view_counts, self.reduction,
        )


class JointInBatchRetrievalLoss(nn.Module):
    """Mix InfoNCE and Progressive Triplet on the same views and targets."""

    def __init__(
        self,
        margin: float = DEFAULT_CIR_CONFIG.triplet_margin,
        temperature: float = DEFAULT_INFO_NCE_TEMPERATURE,
        alpha: float = DEFAULT_JOINT_ALPHA,
        reduction: Reduction = DEFAULT_CIR_CONFIG.loss_reduction,
    ) -> None:
        super().__init__()
        _validate_info_nce_settings(temperature, reduction)
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError("margin must be finite and non-negative")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.margin = margin
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        query_embeddings: Tensor,
        positive_embeddings: Tensor,
        query_positive_indices: Tensor,
    ) -> Tensor:
        distances, view_counts = _view_distances(
            query_embeddings, positive_embeddings, query_positive_indices,
        )
        info_nce = _info_nce_view_losses(
            distances, query_positive_indices, self.temperature,
        )
        triplet = _triplet_view_losses(
            distances, query_positive_indices, self.margin,
        )
        view_losses = self.alpha * info_nce + (1.0 - self.alpha) * triplet
        return _reduce_view_losses(
            view_losses, query_positive_indices, view_counts, self.reduction,
        )


def _info_nce_view_losses(
    distances: Tensor,
    query_positive_indices: Tensor,
    temperature: float,
) -> Tensor:
    return F.cross_entropy(
        -distances / temperature, query_positive_indices, reduction="none",
    )


def _validate_info_nce_settings(temperature: float, reduction: Reduction) -> None:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
