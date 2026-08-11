"""Collate functions for variable-length fashion examples."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from model import OutfitItem

from data.types import (
    CompatibilityBatch,
    CompatibilityExample,
    FashionItem,
    ItemBatch,
    RetrievalBatch,
    RetrievalExample,
)


def collate_items(examples: Sequence[FashionItem]) -> ItemBatch:
    """Create a batch for item embedding precomputation."""
    _require_non_empty(examples)
    return ItemBatch(
        item_ids=tuple(example.item_id for example in examples),
        categories=tuple(example.category for example in examples),
        model_items=tuple(_to_model_item(example) for example in examples),
    )


def collate_compatibility(
    examples: Sequence[CompatibilityExample],
) -> CompatibilityBatch:
    """Keep outfit lengths variable and stack only binary labels."""
    _require_non_empty(examples)
    return CompatibilityBatch(
        example_ids=tuple(example.example_id for example in examples),
        outfit_item_ids=tuple(
            tuple(item.item_id for item in example.outfit) for example in examples
        ),
        outfits=tuple(
            tuple(_to_model_item(item) for item in example.outfit)
            for example in examples
        ),
        labels=torch.tensor(
            [[example.label] for example in examples],
            dtype=torch.float32,
        ),
    )


def collate_retrieval(examples: Sequence[RetrievalExample]) -> RetrievalBatch:
    """Keep query and negative counts variable for future CIR objectives."""
    _require_non_empty(examples)
    return RetrievalBatch(
        example_ids=tuple(example.example_id for example in examples),
        partial_item_ids=tuple(
            tuple(item.item_id for item in example.partial_outfit)
            for example in examples
        ),
        positive_item_ids=tuple(
            example.positive_item.item_id for example in examples
        ),
        negative_item_ids=tuple(
            tuple(item.item_id for item in example.negative_items)
            for example in examples
        ),
        partial_outfits=tuple(
            tuple(_to_model_item(item) for item in example.partial_outfit)
            for example in examples
        ),
        positive_items=tuple(
            _to_model_item(example.positive_item) for example in examples
        ),
        negative_items=tuple(
            tuple(_to_model_item(item) for item in example.negative_items)
            for example in examples
        ),
        target_categories=tuple(
            example.positive_item.category for example in examples
        ),
    )


def _require_non_empty(examples: Sequence[object]) -> None:
    if not examples:
        raise ValueError("cannot collate an empty batch")


def _to_model_item(item: FashionItem) -> OutfitItem:
    return OutfitItem(image=item.image, text=item.description)
