"""Training loss for Complementary Item Retrieval."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..common.config import DEFAULT_CIR_CONFIG, Reduction


class InBatchTripletMarginLoss(nn.Module):
    """Use matching batch positions as positives and mine hardest negatives."""

    def __init__(
        self,
        margin: float = DEFAULT_CIR_CONFIG.triplet_margin,
        reduction: Reduction = DEFAULT_CIR_CONFIG.loss_reduction,
    ) -> None:
        super().__init__()
        if margin < 0.0:
            raise ValueError("margin cannot be negative")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")

        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        query_embeddings: Tensor,
        positive_embeddings: Tensor,
    ) -> Tensor:
        """Return triplet loss over aligned query-positive batch pairs."""
        _validate_embedding_pairs(query_embeddings, positive_embeddings)

        pairwise_distances = torch.cdist(
            query_embeddings,
            positive_embeddings,
            p=2,
        )
        positive_distances = pairwise_distances.diagonal()
        negative_distances = pairwise_distances.masked_fill(
            torch.eye(
                pairwise_distances.size(0),
                device=pairwise_distances.device,
                dtype=torch.bool,
            ),
            torch.inf,
        )
        hardest_negative_distances = negative_distances.min(dim=1).values
        losses = F.relu(
            positive_distances
            - hardest_negative_distances
            + self.margin
        )

        if self.reduction == "mean":
            return losses.mean()
        if self.reduction == "sum":
            return losses.sum()
        return losses


def _validate_embedding_pairs(
    query_embeddings: Tensor,
    positive_embeddings: Tensor,
) -> None:
    if query_embeddings.ndim != 2 or positive_embeddings.ndim != 2:
        raise ValueError(
            "query_embeddings and positive_embeddings must have shape "
            "[batch, features]"
        )
    if query_embeddings.shape != positive_embeddings.shape:
        raise ValueError(
            "query_embeddings and positive_embeddings must have the same shape"
        )
    if query_embeddings.size(0) < 2:
        raise ValueError("in-batch negative mining requires at least two pairs")
    if not torch.is_floating_point(query_embeddings):
        raise TypeError("query_embeddings must be a floating-point tensor")
    if not torch.is_floating_point(positive_embeddings):
        raise TypeError("positive_embeddings must be a floating-point tensor")
    if query_embeddings.device != positive_embeddings.device:
        raise ValueError("query and positive embeddings must share a device")
    if query_embeddings.dtype != positive_embeddings.dtype:
        raise TypeError("query and positive embeddings must share a dtype")
    if not bool(torch.isfinite(query_embeddings).all()):
        raise ValueError("query_embeddings must contain only finite values")
    if not bool(torch.isfinite(positive_embeddings).all()):
        raise ValueError("positive_embeddings must contain only finite values")
