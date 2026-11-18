"""Build CIR inputs for ranking the full Polyvore catalog of one split."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from data import (
    DataSplit,
    ItemBatch,
    RetrievalIndexExample,
    build_resnet18_transform,
    collate_items,
)
from data.polyvore import (
    POLYVORE_DATASET_ID,
    PolyvoreCatalog,
    PolyvoreRetrievalIndexDataset,
    PolyvoreTask,
    download_polyvore_resources,
    load_item_categories,
    load_outfit_token_index,
)
from training.CIR.data import RetrievalDataConfig
from training.common import EmbeddingCache


class CatalogQueryBatch(NamedTuple):
    outfits: tuple[Tensor | ItemBatch, ...]
    target_categories: tuple[str, ...]
    positive_item_ids: tuple[str, ...]


@dataclass(frozen=True)
class CatalogLoaders:
    item_ids: tuple[str, ...]
    candidates: DataLoader[str]
    queries: DataLoader[RetrievalIndexExample]


def build_catalog_loaders(
    config: RetrievalDataConfig,
    *,
    split: DataSplit,
    candidate_batch_size: int,
    token: bool | str | None,
) -> CatalogLoaders:
    config.validate()
    raw_inputs = config.feature_mode.uses_raw_inputs
    resources = download_polyvore_resources(
        task=PolyvoreTask.RETRIEVAL,
        variant=config.subset,
        split=split.value,
        dataset_root=config.dataset_root,
        cache_dir=config.cache_dir,
        token=token,
        include_items=raw_inputs,
        include_metadata=raw_inputs or config.use_category_embedding,
    )
    if resources.retrieval_path is None or resources.outfits_path is None:
        raise RuntimeError("Polyvore FITB annotation resources are incomplete")
    categories = None
    if config.use_category_embedding:
        if resources.metadata_path is None:
            raise RuntimeError("Polyvore item categories require metadata")
        categories = load_item_categories(resources.metadata_path)
    queries = PolyvoreRetrievalIndexDataset(
        resources.retrieval_path, resources.outfits_path, categories,
    )

    items: PolyvoreCatalog | EmbeddingCache
    if raw_inputs:
        if resources.item_rows is None or resources.metadata_path is None:
            raise RuntimeError("Polyvore catalog resources are incomplete")
        items = PolyvoreCatalog(
            resources.item_rows, resources.metadata_path, build_resnet18_transform(),
        )
        item_ids = items.item_ids
    else:
        items = EmbeddingCache(
            config.embedding_root / config.subset / split.value,
            expected_dataset_id=POLYVORE_DATASET_ID,
            expected_subset=config.subset,
            expected_split=split.value,
        )
        if items.embedding_dim != config.model_config.model_dim:
            raise ValueError(
                f"{split.value} embedding_dim must be "
                f"{config.model_config.model_dim}, got {items.embedding_dim}"
            )
        item_ids = tuple(items)
    expected_ids = set(load_outfit_token_index(resources.outfits_path).values())
    missing_ids = expected_ids.difference(item_ids)
    if missing_ids:
        raise ValueError(
            f"incomplete {split.value} catalog: {len(missing_ids)} missing items "
            f"(e.g. {min(missing_ids)!r})"
        )

    collator = _CatalogCollator(items)
    return CatalogLoaders(
        item_ids=item_ids,
        candidates=DataLoader(
            _CatalogItemDataset(item_ids),
            batch_size=candidate_batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=collator.candidates,
        ),
        queries=DataLoader(
            queries,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=collator.queries,
        ),
    )


@dataclass(frozen=True)
class _CatalogItemDataset(Dataset[str]):
    item_ids: tuple[str, ...]

    def __getitem__(self, index: int) -> str:
        return self.item_ids[index]

    def __len__(self) -> int:
        return len(self.item_ids)


@dataclass(frozen=True)
class _CatalogCollator:
    items: PolyvoreCatalog | EmbeddingCache

    def candidates(self, item_ids: Sequence[str]) -> ItemBatch | tuple[Tensor, ...]:
        if isinstance(self.items, PolyvoreCatalog):
            return collate_items(self.items.get_many(item_ids))
        return tuple(self.items[item_id].unsqueeze(0) for item_id in item_ids)

    def queries(self, examples: Sequence[RetrievalIndexExample]) -> CatalogQueryBatch:
        return CatalogQueryBatch(
            outfits=tuple(self._outfit(example.partial_item_ids) for example in examples),
            target_categories=tuple(example.target_category for example in examples),
            positive_item_ids=tuple(example.positive_item_id for example in examples),
        )

    def _outfit(self, item_ids: tuple[str, ...]) -> Tensor | ItemBatch:
        if isinstance(self.items, PolyvoreCatalog):
            return collate_items(self.items.get_many(item_ids))
        return torch.stack([self.items[item_id] for item_id in item_ids])
