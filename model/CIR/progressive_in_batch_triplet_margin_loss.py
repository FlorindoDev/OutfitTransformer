"""Triplet loss averaged over progressively reduced contexts per outfit."""

import torch
from torch import Tensor
from torch.nn import functional as F

from .in_batch_triplet_margin_loss import (
    InBatchTripletMarginLoss,
    _validate_embedding_pairs,
)


class ProgressiveInBatchTripletMarginLoss(InBatchTripletMarginLoss):
    """Average view losses per outfit before applying the batch reduction.

    Queries have shape [views, features], positives [batch, features], and
    query_positive_indices [views] maps each query to its unchanged target.
    Each outfit must have at least one view. Other outfits' targets remain
    the negative pool, including when only one outfit has a longer context.
    """

    def forward(
        self,
        query_embeddings: Tensor,
        positive_embeddings: Tensor,
        query_positive_indices: Tensor,
    ) -> Tensor:
        if query_embeddings.ndim != 2 or positive_embeddings.ndim != 2:
            raise ValueError("query and positive embeddings must be 2D")
        batch_size = positive_embeddings.size(0)
        if batch_size < 2:
            raise ValueError("in-batch negative mining requires at least two targets")
        if (
            query_positive_indices.ndim != 1
            or query_positive_indices.size(0) != query_embeddings.size(0)
        ):
            raise ValueError("query_positive_indices must have one index per query")
        if query_positive_indices.dtype != torch.long:
            raise TypeError("query_positive_indices must have dtype torch.long")
        if query_positive_indices.device != positive_embeddings.device:
            raise ValueError("query indices and positive embeddings must share a device")
        invalid_indices = (
            (query_positive_indices < 0) | (query_positive_indices >= batch_size)
        )
        if bool(invalid_indices.any()):
            raise ValueError("query_positive_indices contains an invalid target index")
        view_counts = torch.bincount(query_positive_indices, minlength=batch_size)
        if bool((view_counts == 0).any()):
            raise ValueError("each target must have at least one query view")
        _validate_embedding_pairs(
            query_embeddings, positive_embeddings[query_positive_indices],
        )

        distances = torch.cdist(query_embeddings, positive_embeddings, p=2)
        positive_mask = F.one_hot(
            query_positive_indices, num_classes=batch_size,
        ).bool()
        positive_distances = distances.gather(
            1, query_positive_indices.unsqueeze(1),
        ).squeeze(1)
        negative_distances = distances.masked_fill(positive_mask, torch.inf)
        view_losses = F.relu(
            positive_distances - negative_distances.min(dim=1).values + self.margin
        )
        outfit_losses = view_losses.new_zeros(batch_size).scatter_add(
            0, query_positive_indices, view_losses,
        ) / view_counts

        if self.reduction == "mean":
            return outfit_losses.mean()
        if self.reduction == "sum":
            return outfit_losses.sum()
        return outfit_losses
