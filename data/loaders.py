"""DataLoader factories for item, compatibility and retrieval tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from data.collate import collate_compatibility, collate_items, collate_retrieval
from data.polyvore.catalog import PolyvoreCatalog
from data.polyvore.compatibility_dataset import PolyvoreCompatibilityDataset
from data.polyvore.download import (
    DEFAULT_DATASET_ROOT,
    PolyvoreResources,
    PolyvoreSplit,
    PolyvoreTask,
    PolyvoreVariant,
    download_polyvore_resources,
)
from data.polyvore.item_dataset import PolyvoreItemDataset
from data.polyvore.retrieval_dataset import PolyvoreRetrievalDataset
from data.transforms import ImageTransform
from data.types import CompatibilityExample, FashionItem, RetrievalExample


@dataclass(frozen=True)
class LoaderConfig:
    """Task-independent PyTorch DataLoader settings."""

    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    drop_last: bool = False
    seed: int | None = None

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError(
                "persistent_workers requires num_workers greater than zero"
            )
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed cannot be negative")


def create_item_loader(
    dataset: Dataset[FashionItem],
    *,
    config: LoaderConfig | None = None,
    shuffle: bool = False,
) -> DataLoader[Any]:
    """Create a loader for individual catalog items."""
    return _create_loader(
        dataset,
        collate_fn=collate_items,
        config=config,
        shuffle=shuffle,
    )


def create_compatibility_loader(
    dataset: Dataset[CompatibilityExample],
    *,
    config: LoaderConfig | None = None,
    shuffle: bool = False,
) -> DataLoader[Any]:
    """Create a loader whose batch can be passed directly to CP training."""
    return _create_loader(
        dataset,
        collate_fn=collate_compatibility,
        config=config,
        shuffle=shuffle,
    )


def create_retrieval_loader(
    dataset: Dataset[RetrievalExample],
    *,
    config: LoaderConfig | None = None,
    shuffle: bool = False,
) -> DataLoader[Any]:
    """Create a loader for future complementary item retrieval training."""
    return _create_loader(
        dataset,
        collate_fn=collate_retrieval,
        config=config,
        shuffle=shuffle,
    )


def build_polyvore_item_loader(
    *,
    image_transform: ImageTransform,
    variant: PolyvoreVariant | str = PolyvoreVariant.DISJOINT,
    split: PolyvoreSplit | str = PolyvoreSplit.TRAIN,
    config: LoaderConfig | None = None,
    shuffle: bool = False,
    token: bool | str | None = True,
    cache_dir: str | Path | None = None,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
) -> DataLoader[Any]:
    """Resolve resources and build a Polyvore item loader."""
    resources = download_polyvore_resources(
        task=PolyvoreTask.ITEMS,
        variant=variant,
        split=split,
        token=token,
        cache_dir=cache_dir,
        dataset_root=dataset_root,
    )
    catalog = _build_catalog(resources, image_transform)
    return create_item_loader(
        PolyvoreItemDataset(catalog),
        config=config,
        shuffle=shuffle,
    )


def build_polyvore_compatibility_loader(
    *,
    image_transform: ImageTransform,
    variant: PolyvoreVariant | str = PolyvoreVariant.DISJOINT,
    split: PolyvoreSplit | str = PolyvoreSplit.TRAIN,
    config: LoaderConfig | None = None,
    shuffle: bool | None = None,
    token: bool | str | None = True,
    cache_dir: str | Path | None = None,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
) -> DataLoader[Any]:
    """Resolve resources and build the complete Polyvore CP pipeline."""
    selected_split = PolyvoreSplit(split)
    resources = download_polyvore_resources(
        task=PolyvoreTask.COMPATIBILITY,
        variant=variant,
        split=selected_split,
        token=token,
        cache_dir=cache_dir,
        dataset_root=dataset_root,
    )
    catalog = _build_catalog(resources, image_transform)
    dataset = PolyvoreCompatibilityDataset(
        catalog,
        _require_path(resources.compatibility_path, "compatibility_path"),
        _require_path(resources.outfits_path, "outfits_path"),
    )
    selected_shuffle = (
        selected_split is PolyvoreSplit.TRAIN if shuffle is None else shuffle
    )
    return create_compatibility_loader(
        dataset,
        config=config,
        shuffle=selected_shuffle,
    )


def build_polyvore_retrieval_loader(
    *,
    image_transform: ImageTransform,
    variant: PolyvoreVariant | str = PolyvoreVariant.DISJOINT,
    split: PolyvoreSplit | str = PolyvoreSplit.TRAIN,
    config: LoaderConfig | None = None,
    shuffle: bool | None = None,
    token: bool | str | None = True,
    cache_dir: str | Path | None = None,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
) -> DataLoader[Any]:
    """Resolve resources and build the complete Polyvore FITB pipeline."""
    selected_split = PolyvoreSplit(split)
    resources = download_polyvore_resources(
        task=PolyvoreTask.RETRIEVAL,
        variant=variant,
        split=selected_split,
        token=token,
        cache_dir=cache_dir,
        dataset_root=dataset_root,
    )
    catalog = _build_catalog(resources, image_transform)
    dataset = PolyvoreRetrievalDataset(
        catalog,
        _require_path(resources.retrieval_path, "retrieval_path"),
        _require_path(resources.outfits_path, "outfits_path"),
    )
    selected_shuffle = (
        selected_split is PolyvoreSplit.TRAIN if shuffle is None else shuffle
    )
    return create_retrieval_loader(
        dataset,
        config=config,
        shuffle=selected_shuffle,
    )


def _create_loader(
    dataset: Dataset[Any],
    *,
    collate_fn: Any,
    config: LoaderConfig | None,
    shuffle: bool,
) -> DataLoader[Any]:
    selected_config = config or LoaderConfig()
    selected_config.validate()
    generator = None
    if selected_config.seed is not None:
        generator = torch.Generator().manual_seed(selected_config.seed)

    return DataLoader(
        dataset,
        batch_size=selected_config.batch_size,
        shuffle=shuffle,
        num_workers=selected_config.num_workers,
        collate_fn=collate_fn,
        pin_memory=selected_config.pin_memory,
        persistent_workers=selected_config.persistent_workers,
        drop_last=selected_config.drop_last,
        generator=generator,
    )


def _build_catalog(
    resources: PolyvoreResources,
    image_transform: ImageTransform,
) -> PolyvoreCatalog:
    if resources.item_rows is None or resources.metadata_path is None:
        raise RuntimeError("Polyvore catalog resources are incomplete")
    return PolyvoreCatalog(
        resources.item_rows,
        resources.metadata_path,
        image_transform,
    )


def _require_path(path: Path | None, name: str) -> Path:
    if path is None:
        raise RuntimeError(f"downloaded resources do not contain {name}")
    return path
