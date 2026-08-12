"""Dataset-neutral CP inputs and precomputed CLIP embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from data import (
    CompatibilityIndexExample,
    DataSplit,
    DatasetRequest,
    IndexedDataset,
    LoaderConfig,
    build_resnet18_transform,
    create_compatibility_loader,
    get_dataset_source,
)
from model import TransformerConfig
from training.common import EmbeddingCache

from .config import CPTrainingConfig, FeatureMode


@dataclass(frozen=True)
class CompatibilityEmbeddingExample:
    """One CP example represented only by frozen item embeddings."""

    example_id: str
    outfit: Tensor
    label: int


@dataclass(frozen=True)
class CompatibilityEmbeddingBatch:
    """Variable-length embedding matrices and binary labels."""

    example_ids: tuple[str, ...]
    outfits: tuple[Tensor, ...]
    labels: Tensor

    def pin_memory(self) -> "CompatibilityEmbeddingBatch":
        return CompatibilityEmbeddingBatch(
            example_ids=self.example_ids,
            outfits=tuple(outfit.pin_memory() for outfit in self.outfits),
            labels=self.labels.pin_memory(),
        )


@dataclass(frozen=True)
class CompatibilityLoaders:
    train: DataLoader[Any]
    validation: DataLoader[Any]


@dataclass(frozen=True)
class CompatibilityDataConfig:
    """Data-only settings shared by CP training and evaluation."""

    dataset_name: str
    subset: str
    feature_mode: FeatureMode
    embedding_root: Path
    dataset_root: Path
    cache_dir: Path | None
    batch_size: int
    num_workers: int
    pin_memory: bool
    seed: int
    model_config: TransformerConfig

    @classmethod
    def from_training_config(
        cls,
        config: CPTrainingConfig,
    ) -> "CompatibilityDataConfig":
        return cls(
            dataset_name=config.dataset_name,
            subset=config.subset,
            feature_mode=config.feature_mode,
            embedding_root=config.embedding_root,
            dataset_root=config.dataset_root,
            cache_dir=config.cache_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            seed=config.seed,
            model_config=config.model_config,
        )

    def validate(self) -> None:
        source = get_dataset_source(self.dataset_name)
        source.descriptor.validate_subset(self.subset)
        if not isinstance(self.feature_mode, FeatureMode):
            raise TypeError("feature_mode must be a FeatureMode")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        self.model_config.validate()


class PrecomputedCompatibilityDataset(
    Dataset[CompatibilityEmbeddingExample]
):
    """Resolve official CP annotations through one embedding cache."""

    def __init__(
        self,
        examples: IndexedDataset[CompatibilityIndexExample],
        embeddings: EmbeddingCache,
    ) -> None:
        self._embeddings = embeddings
        self._examples = tuple(
            examples[index] for index in range(len(examples))
        )
        _validate_embedding_coverage(self._examples, embeddings)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> CompatibilityEmbeddingExample:
        example = self._examples[index]
        return CompatibilityEmbeddingExample(
            example_id=example.example_id,
            outfit=torch.stack(
                [self._embeddings[item_id] for item_id in example.item_ids]
            ),
            label=example.label,
        )


def build_compatibility_loaders(
    config: CPTrainingConfig,
    *,
    token: bool | str | None = True,
) -> CompatibilityLoaders:
    """Build loaders for raw inputs or precomputed CLIP features."""
    data_config = CompatibilityDataConfig.from_training_config(config)
    data_config.validate()
    if data_config.feature_mode.uses_raw_inputs:
        return CompatibilityLoaders(
            train=_build_classic_loader(
                data_config,
                split=DataSplit.TRAIN,
                shuffle=True,
                token=token,
            ),
            validation=_build_classic_loader(
                data_config,
                split=DataSplit.VALIDATION,
                shuffle=False,
                token=token,
            ),
        )

    train_loader, train_cache = _build_clip_loader(
        data_config,
        split=DataSplit.TRAIN,
        shuffle=True,
        token=token,
    )
    validation_loader, validation_cache = _build_clip_loader(
        data_config,
        split=DataSplit.VALIDATION,
        shuffle=False,
        token=token,
    )
    _require_matching_fingerprints(
        {
            DataSplit.TRAIN: train_cache,
            DataSplit.VALIDATION: validation_cache,
        }
    )
    return CompatibilityLoaders(
        train=train_loader,
        validation=validation_loader,
    )


def build_compatibility_loader(
    config: CompatibilityDataConfig,
    *,
    split: DataSplit,
    token: bool | str | None = True,
    shuffle: bool = False,
) -> DataLoader[Any]:
    """Build one CP loader for a requested dataset split."""
    config.validate()
    if not isinstance(split, DataSplit):
        raise TypeError("split must be a DataSplit")
    if config.feature_mode.uses_raw_inputs:
        return _build_classic_loader(
            config,
            split=split,
            shuffle=shuffle,
            token=token,
        )
    loader, _ = _build_clip_loader(
        config,
        split=split,
        shuffle=shuffle,
        token=token,
    )
    return loader


def _build_classic_loader(
    config: CompatibilityDataConfig,
    *,
    split: DataSplit,
    shuffle: bool,
    token: bool | str | None,
) -> DataLoader[Any]:
    loader_config = LoaderConfig(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        seed=config.seed,
    )
    source = get_dataset_source(config.dataset_name)
    dataset = source.compatibility_dataset(
        _dataset_request(config, split, token),
        build_resnet18_transform(augment=split is DataSplit.TRAIN),
    )
    return create_compatibility_loader(
        dataset,
        config=loader_config,
        shuffle=shuffle,
    )


def _build_clip_loader(
    config: CompatibilityDataConfig,
    *,
    split: DataSplit,
    shuffle: bool,
    token: bool | str | None,
) -> tuple[DataLoader[Any], EmbeddingCache]:
    source = get_dataset_source(config.dataset_name)
    cache = EmbeddingCache(
        config.embedding_root / config.subset / split.value,
        expected_dataset_id=source.descriptor.dataset_id,
        expected_subset=config.subset,
        expected_split=split.value,
    )
    if cache.embedding_dim != config.model_config.model_dim:
        raise ValueError(
            f"{split.value} embedding_dim must be "
            f"{config.model_config.model_dim}, got {cache.embedding_dim}"
        )

    dataset = PrecomputedCompatibilityDataset(
        source.compatibility_index_dataset(
            _dataset_request(config, split, token)
        ),
        cache,
    )

    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=config.num_workers,
        collate_fn=collate_compatibility_embeddings,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
    ), cache


def collate_compatibility_embeddings(
    examples: list[CompatibilityEmbeddingExample],
) -> CompatibilityEmbeddingBatch:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    return CompatibilityEmbeddingBatch(
        example_ids=tuple(example.example_id for example in examples),
        outfits=tuple(example.outfit for example in examples),
        labels=torch.tensor(
            [[example.label] for example in examples],
            dtype=torch.float32,
        ),
    )


def _validate_embedding_coverage(
    examples: tuple[CompatibilityIndexExample, ...],
    embeddings: EmbeddingCache,
) -> None:
    for example in examples:
        for item_id in example.item_ids:
            if item_id not in embeddings:
                raise ValueError(
                    f"missing embedding for item {item_id!r} "
                    f"in {example.example_id}"
                )


def _dataset_request(
    config: CompatibilityDataConfig,
    split: DataSplit,
    token: bool | str | None,
) -> DatasetRequest:
    return DatasetRequest(
        subset=config.subset,
        split=split,
        root=config.dataset_root,
        cache_dir=config.cache_dir,
        token=token,
    )


def _require_matching_fingerprints(
    caches: dict[DataSplit, EmbeddingCache],
) -> None:
    fingerprints = {cache.model_fingerprint for cache in caches.values()}
    if "" in fingerprints:
        raise ValueError("embedding manifests must contain model_fingerprint")
    if len(fingerprints) != 1:
        raise ValueError("train and validation embeddings use different models")
