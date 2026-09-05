"""Random outfit completions for CIR training, sampled on every access."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset

from data.types import RetrievalExample, RetrievalIndexExample

from .catalog import PolyvoreCatalog, load_outfit_token_index


@dataclass(frozen=True)
class _TrainingOutfit:
    example_id: str
    item_ids: tuple[str, ...]


class PolyvoreRetrievalTrainingIndexDataset(Dataset[RetrievalIndexExample]):
    """Sample a target from a complete outfit without loading images."""

    def __init__(
        self,
        outfits_path: str | Path,
        category_by_item_id: Mapping[str, str] | None = None,
    ) -> None:
        self._outfits = _load_outfits(outfits_path)
        self._categories = dict(category_by_item_id or {})
        self._item_ids = tuple(dict.fromkeys(
            item_id for outfit in self._outfits for item_id in outfit.item_ids
        ))

    def __len__(self) -> int:
        return len(self._outfits)

    @property
    def item_ids(self) -> tuple[str, ...]:
        """Expose all possible targets and query items without consuming RNG."""
        return self._item_ids

    def __getitem__(self, index: int) -> RetrievalIndexExample:
        outfit = self._outfits[index]
        positive_id = random.choice(outfit.item_ids)
        return RetrievalIndexExample(
            example_id=outfit.example_id,
            partial_item_ids=tuple(
                item_id for item_id in outfit.item_ids if item_id != positive_id
            ),
            positive_item_id=positive_id,
            negative_item_ids=(),
            target_category=self._categories.get(positive_id, "unknown"),
        )


class PolyvoreRetrievalTrainingDataset(Dataset[RetrievalExample]):
    """Resolve each freshly sampled query and target through the raw catalog."""

    def __init__(
        self,
        catalog: PolyvoreCatalog,
        outfits_path: str | Path,
    ) -> None:
        self.catalog = catalog
        self._examples = PolyvoreRetrievalTrainingIndexDataset(outfits_path)
        for item_id in self._examples.item_ids:
            if item_id not in catalog:
                raise ValueError(
                    f"training item {item_id!r} is absent from the image split"
                )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> RetrievalExample:
        example = self._examples[index]
        return RetrievalExample(
            example_id=example.example_id,
            partial_outfit=self.catalog.get_many(example.partial_item_ids),
            positive_item=self.catalog.get(example.positive_item_id),
            negative_items=(),
        )


def _load_outfits(path: str | Path) -> tuple[_TrainingOutfit, ...]:
    token_index = load_outfit_token_index(path)
    grouped: dict[str, list[str]] = {}
    for token, item_id in token_index.items():
        set_id = token.rsplit("_", maxsplit=1)[0]
        grouped.setdefault(set_id, []).append(item_id)
    if not grouped:
        raise ValueError("CIR training requires a non-empty outfit list")

    outfits: list[_TrainingOutfit] = []
    for set_id, item_ids in grouped.items():
        if len(set(item_ids)) < 2:
            raise ValueError(
                f"training outfit {set_id!r} needs at least two distinct items"
            )
        outfits.append(_TrainingOutfit(
            example_id=f"retrieval_train:{set_id}",
            item_ids=tuple(item_ids),
        ))
    return tuple(outfits)
