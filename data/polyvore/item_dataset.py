"""Dataset of individual Polyvore items for embedding precomputation."""

from __future__ import annotations

from collections.abc import Sequence

from torch.utils.data import Dataset

from data.types import FashionItem

from .catalog import PolyvoreCatalog


class PolyvoreItemDataset(Dataset[FashionItem]):
    """Return one raw multimodal catalog item at a time."""

    def __init__(
        self,
        catalog: PolyvoreCatalog,
        item_ids: Sequence[str] | None = None,
    ) -> None:
        self.catalog = catalog
        self._item_ids = (
            catalog.item_ids
            if item_ids is None
            else tuple(str(item_id) for item_id in item_ids)
        )
        if not self._item_ids:
            raise ValueError("item_ids cannot be empty")
        unknown = [item_id for item_id in self._item_ids if item_id not in catalog]
        if unknown:
            raise KeyError(f"unknown Polyvore item_id: {unknown[0]}")

    def __len__(self) -> int:
        return len(self._item_ids)

    def __getitem__(self, index: int) -> FashionItem:
        return self.catalog.get(self._item_ids[index])
