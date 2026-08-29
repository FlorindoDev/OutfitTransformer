"""Validation shared by task-specific outfit Transformers."""

import torch
from torch import Tensor

from .transformer import OutfitTransformerOutput


def validate_common_output(
    common_output: OutfitTransformerOutput,
    *,
    expected_dim: int,
) -> tuple[Tensor, Tensor]:
    """Return validated contextual embeddings and their padding mask."""
    if not isinstance(common_output, OutfitTransformerOutput):
        raise TypeError("common_output must be an OutfitTransformerOutput")

    item_embeddings = common_output.contextual_embeddings
    padding_mask = common_output.padding_mask
    if item_embeddings.ndim != 3 or item_embeddings.size(2) != expected_dim:
        raise ValueError(
            "common contextual_embeddings must have shape "
            f"[batch, items, {expected_dim}]"
        )
    if item_embeddings.size(0) == 0 or item_embeddings.size(1) == 0:
        raise ValueError("common contextual_embeddings cannot be empty")
    if padding_mask.shape != item_embeddings.shape[:2]:
        raise ValueError(
            "common padding_mask must match contextual_embeddings batch and items"
        )
    if padding_mask.dtype != torch.bool:
        raise TypeError("common padding_mask must be boolean")
    if padding_mask.device != item_embeddings.device:
        raise ValueError(
            "common padding_mask and contextual_embeddings must share a device"
        )
    if bool(padding_mask.all(dim=1).any()):
        raise ValueError("each outfit must contain at least one real item")
    if not bool(torch.isfinite(item_embeddings).all()):
        raise ValueError("common contextual_embeddings must contain finite values")
    return item_embeddings, padding_mask
