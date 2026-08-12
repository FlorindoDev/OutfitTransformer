"""Shared samples and model-ready batches for fashion data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from model import OutfitItem


@dataclass(frozen=True)
class FashionItem:
    """One catalog item before conversion to the model input API."""

    item_id: str
    image: Tensor
    description: str
    category: str

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("item_id cannot be empty")
        if self.image.ndim != 3 or self.image.size(0) != 3:
            raise ValueError("image must have shape [3, height, width]")
        if not torch.is_floating_point(self.image):
            raise TypeError("image must be a floating-point tensor")
        if not bool(torch.isfinite(self.image).all()):
            raise ValueError("image must contain only finite values")
        if not self.description.strip():
            raise ValueError("description cannot be empty")
        if not self.category.strip():
            raise ValueError("category cannot be empty")


@dataclass(frozen=True)
class CompatibilityExample:
    """One variable-length outfit and its binary compatibility label."""

    example_id: str
    outfit: tuple[FashionItem, ...]
    label: int

    def __post_init__(self) -> None:
        if not self.example_id.strip():
            raise ValueError("example_id cannot be empty")
        if not self.outfit:
            raise ValueError("outfit cannot be empty")
        if self.label not in {0, 1}:
            raise ValueError("label must be 0 or 1")


@dataclass(frozen=True)
class CompatibilityIndexExample:
    """Compatibility example containing item IDs but no loaded features."""

    example_id: str
    item_ids: tuple[str, ...]
    label: int

    def __post_init__(self) -> None:
        if not self.example_id.strip():
            raise ValueError("example_id cannot be empty")
        if not self.item_ids:
            raise ValueError("item_ids cannot be empty")
        if any(not item_id.strip() for item_id in self.item_ids):
            raise ValueError("item_ids cannot contain empty values")
        if self.label not in {0, 1}:
            raise ValueError("label must be 0 or 1")


@dataclass(frozen=True)
class RetrievalExample:
    """Partial outfit, positive completion and explicit negative candidates."""

    example_id: str
    partial_outfit: tuple[FashionItem, ...]
    positive_item: FashionItem
    negative_items: tuple[FashionItem, ...]

    def __post_init__(self) -> None:
        if not self.example_id.strip():
            raise ValueError("example_id cannot be empty")
        if not self.partial_outfit:
            raise ValueError("partial_outfit cannot be empty")
        if not self.negative_items:
            raise ValueError("negative_items cannot be empty")


@dataclass(frozen=True)
class ItemBatch:
    """Catalog items ready for multimodal embedding precomputation."""

    item_ids: tuple[str, ...]
    categories: tuple[str, ...]
    model_items: tuple[OutfitItem, ...]

    def as_single_item_outfits(self) -> tuple[tuple[OutfitItem, ...], ...]:
        """Return the shape expected by ``OutfitTransformer.forward``."""
        return tuple((item,) for item in self.model_items)

    def pin_memory(self) -> ItemBatch:
        """Let PyTorch pin tensors stored inside custom batch objects."""
        return ItemBatch(
            item_ids=self.item_ids,
            categories=self.categories,
            model_items=tuple(_pin_model_item(item) for item in self.model_items),
        )


@dataclass(frozen=True)
class CompatibilityBatch:
    """Model-ready CP inputs; labels have shape ``[batch, 1]``."""

    example_ids: tuple[str, ...]
    outfit_item_ids: tuple[tuple[str, ...], ...]
    outfits: tuple[tuple[OutfitItem, ...], ...]
    labels: Tensor

    def pin_memory(self) -> CompatibilityBatch:
        return CompatibilityBatch(
            example_ids=self.example_ids,
            outfit_item_ids=self.outfit_item_ids,
            outfits=tuple(
                tuple(_pin_model_item(item) for item in outfit)
                for outfit in self.outfits
            ),
            labels=self.labels.pin_memory(),
        )


@dataclass(frozen=True)
class RetrievalBatch:
    """Model-ready CIR inputs with a variable number of negatives."""

    example_ids: tuple[str, ...]
    partial_item_ids: tuple[tuple[str, ...], ...]
    positive_item_ids: tuple[str, ...]
    negative_item_ids: tuple[tuple[str, ...], ...]
    partial_outfits: tuple[tuple[OutfitItem, ...], ...]
    positive_items: tuple[OutfitItem, ...]
    negative_items: tuple[tuple[OutfitItem, ...], ...]
    target_categories: tuple[str, ...]

    def pin_memory(self) -> RetrievalBatch:
        return RetrievalBatch(
            example_ids=self.example_ids,
            partial_item_ids=self.partial_item_ids,
            positive_item_ids=self.positive_item_ids,
            negative_item_ids=self.negative_item_ids,
            partial_outfits=tuple(
                tuple(_pin_model_item(item) for item in outfit)
                for outfit in self.partial_outfits
            ),
            positive_items=tuple(
                _pin_model_item(item) for item in self.positive_items
            ),
            negative_items=tuple(
                tuple(_pin_model_item(item) for item in negatives)
                for negatives in self.negative_items
            ),
            target_categories=self.target_categories,
        )


def _pin_model_item(item: OutfitItem) -> OutfitItem:
    from model import OutfitItem

    return OutfitItem(
        image=item.image.pin_memory() if item.image is not None else None,
        text=item.text,
        embedding=(
            item.embedding.pin_memory() if item.embedding is not None else None
        ),
    )
