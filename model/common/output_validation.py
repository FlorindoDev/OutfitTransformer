"""Embedding-batch validation shared by task-specific Transformers."""

import torch
from torch import Tensor

from .embeddings import OutfitEmbeddingBatch


def validate_outfit_batch(
    outfit_batch: OutfitEmbeddingBatch,
    *,
    expected_dim: int,
) -> tuple[Tensor, Tensor]:
    """Return validated item embeddings and their padding mask."""
    if not isinstance(outfit_batch, OutfitEmbeddingBatch):
        raise TypeError("outfit_batch must be an OutfitEmbeddingBatch")

    item_embeddings = outfit_batch.item_embeddings
    padding_mask = outfit_batch.padding_mask
    if item_embeddings.ndim != 3 or item_embeddings.size(2) != expected_dim:
        raise ValueError(
            "item_embeddings must have shape "
            f"[batch, items, {expected_dim}]"
        )
    if item_embeddings.size(0) == 0 or item_embeddings.size(1) == 0:
        raise ValueError("item_embeddings cannot be empty")
    if padding_mask.shape != item_embeddings.shape[:2]:
        raise ValueError(
            "padding_mask must match item_embeddings batch and items"
        )
    if padding_mask.dtype != torch.bool:
        raise TypeError("padding_mask must be boolean")
    if padding_mask.device != item_embeddings.device:
        raise ValueError(
            "padding_mask and item_embeddings must share a device"
        )
    if bool(padding_mask.all(dim=1).any()):
        raise ValueError("each outfit must contain at least one real item")
    if not bool(torch.isfinite(item_embeddings).all()):
        raise ValueError("item_embeddings must contain finite values")
    return item_embeddings, padding_mask
