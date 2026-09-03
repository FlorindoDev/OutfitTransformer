"""Dataset-neutral CIR inputs and precomputed embeddings."""

from __future__ import annotations

import math
from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from data import (
    DataSplit,
    DatasetRequest,
    IndexedDataset,
    RetrievalIndexExample,
    build_resnet18_transform,
    collate_retrieval,
    get_dataset_source,
)
from model import OutfitItem, TransformerConfig
from training.common import EmbeddingCache
from training.common.features import FeatureMode

from .config import CIRTrainingConfig


@dataclass(frozen=True)
class RetrievalEmbeddingExample:
    """One FITB example represented only by frozen item embeddings."""

    example_id: str
    partial_outfit: Tensor
    positive_item: Tensor
    negative_items: tuple[Tensor, ...]
    target_category: str


@dataclass(frozen=True)
class RetrievalEmbeddingBatch:
    """Variable-length embedding inputs for CIR training and validation."""

    example_ids: tuple[str, ...]
    partial_outfits: tuple[Tensor, ...]
    positive_items: tuple[Tensor, ...]
    negative_items: tuple[tuple[Tensor, ...], ...]
    target_categories: tuple[str, ...]

    def pin_memory(self) -> "RetrievalEmbeddingBatch":
        return RetrievalEmbeddingBatch(
            example_ids=self.example_ids,
            partial_outfits=tuple(
                outfit.pin_memory() for outfit in self.partial_outfits
            ),
            positive_items=tuple(
                item.pin_memory() for item in self.positive_items
            ),
            negative_items=tuple(
                tuple(item.pin_memory() for item in candidates)
                for candidates in self.negative_items
            ),
            target_categories=self.target_categories,
        )


@dataclass(frozen=True)
class RetrievalLoaders:
    train: DataLoader[Any]
    validation: DataLoader[Any]


@dataclass(frozen=True)
class RetrievalDataConfig:
    """Data-only settings shared by CIR training and evaluation."""

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
    use_category_embedding: bool
    distributed: bool
    model_config: TransformerConfig

    @classmethod
    def from_training_config(
        cls,
        config: CIRTrainingConfig,
    ) -> "RetrievalDataConfig":
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
            use_category_embedding=config.use_category_embedding,
            distributed=config.ddp,
            model_config=config.model_config,
        )

    def validate(self) -> None:
        source = get_dataset_source(self.dataset_name)
        source.descriptor.validate_subset(self.subset)
        if not isinstance(self.feature_mode, FeatureMode):
            raise TypeError("feature_mode must be a FeatureMode")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if not isinstance(self.use_category_embedding, bool):
            raise TypeError("use_category_embedding must be boolean")
        if not isinstance(self.distributed, bool):
            raise TypeError("distributed must be boolean")
        self.model_config.validate()


class PrecomputedRetrievalDataset(
    Dataset[RetrievalEmbeddingExample]
):
    """Resolve official FITB annotations through one embedding cache."""

    def __init__(
        self,
        examples: IndexedDataset[RetrievalIndexExample],
        embeddings: EmbeddingCache,
    ) -> None:
        self._embeddings = embeddings
        self._examples = tuple(
            examples[index] for index in range(len(examples))
        )
        _validate_embedding_coverage(self._examples, embeddings)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> RetrievalEmbeddingExample:
        example = self._examples[index]
        return RetrievalEmbeddingExample(
            example_id=example.example_id,
            partial_outfit=torch.stack(
                [
                    self._embeddings[item_id]
                    for item_id in example.partial_item_ids
                ]
            ),
            positive_item=self._embeddings[example.positive_item_id],
            negative_items=tuple(
                self._embeddings[item_id]
                for item_id in example.negative_item_ids
            ),
            target_category=example.target_category,
        )


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation examples across ranks without padding duplicates."""

    def __init__(
        self,
        dataset: Dataset[Any],
        *,
        num_replicas: int,
        rank: int,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be within num_replicas")
        if not isinstance(dataset, Sized):
            raise TypeError("distributed evaluation dataset must be sized")
        self._dataset_size = len(dataset)
        self._num_replicas = num_replicas
        self._rank = rank

    def __iter__(self) -> Iterator[int]:
        return iter(
            range(self._rank, self._dataset_size, self._num_replicas)
        )

    def __len__(self) -> int:
        remaining = max(0, self._dataset_size - self._rank)
        return math.ceil(remaining / self._num_replicas)


def build_retrieval_loaders(
    config: CIRTrainingConfig,
    *,
    token: bool | str | None = True,
) -> RetrievalLoaders:
    """Build CIR loaders for raw inputs or precomputed features."""
    data_config = RetrievalDataConfig.from_training_config(config)
    data_config.validate()
    if data_config.feature_mode.uses_raw_inputs:
        return RetrievalLoaders(
            train=_build_classic_loader(
                data_config,
                split=DataSplit.TRAIN,
                shuffle=True,
                drop_last=True,
                token=token,
            ),
            validation=_build_classic_loader(
                data_config,
                split=DataSplit.VALIDATION,
                shuffle=False,
                drop_last=False,
                token=token,
            ),
        )

    train_loader, train_cache = _build_precomputed_loader(
        data_config,
        split=DataSplit.TRAIN,
        shuffle=True,
        drop_last=True,
        token=token,
    )
    validation_loader, validation_cache = _build_precomputed_loader(
        data_config,
        split=DataSplit.VALIDATION,
        shuffle=False,
        drop_last=False,
        token=token,
    )
    _require_matching_fingerprints((train_cache, validation_cache))
    return RetrievalLoaders(train=train_loader, validation=validation_loader)


def collate_retrieval_embeddings(
    examples: list[RetrievalEmbeddingExample],
) -> RetrievalEmbeddingBatch:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    return RetrievalEmbeddingBatch(
        example_ids=tuple(example.example_id for example in examples),
        partial_outfits=tuple(example.partial_outfit for example in examples),
        positive_items=tuple(example.positive_item for example in examples),
        negative_items=tuple(
            example.negative_items for example in examples
        ),
        target_categories=tuple(
            example.target_category for example in examples
        ),
    )


def _build_classic_loader(
    config: RetrievalDataConfig,
    *,
    split: DataSplit,
    shuffle: bool,
    drop_last: bool,
    token: bool | str | None,
) -> DataLoader[Any]:
    source = get_dataset_source(config.dataset_name)
    dataset = source.retrieval_dataset(
        _dataset_request(config, split, token),
        build_resnet18_transform(augment=split is DataSplit.TRAIN),
    )
    return _create_loader(
        dataset,
        collate_fn=collate_retrieval,
        config=config,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def _build_precomputed_loader(
    config: RetrievalDataConfig,
    *,
    split: DataSplit,
    shuffle: bool,
    drop_last: bool,
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
    examples = source.retrieval_index_dataset(
        _dataset_request(config, split, token),
        include_categories=config.use_category_embedding,
    )
    dataset = PrecomputedRetrievalDataset(examples, cache)
    return (
        _create_loader(
            dataset,
            collate_fn=collate_retrieval_embeddings,
            config=config,
            shuffle=shuffle,
            drop_last=drop_last,
        ),
        cache,
    )


def _create_loader(
    dataset: Dataset[Any],
    *,
    collate_fn: Any,
    config: RetrievalDataConfig,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader[Any]:
    sampler: Sampler[int] | None = None
    if config.distributed:
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "distributed process group must be initialized before loaders"
            )
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        if shuffle:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=config.seed,
                drop_last=False,
            )
        else:
            sampler = DistributedEvalSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
            )

    generator = None
    if sampler is None and shuffle:
        generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        generator=generator,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        drop_last=drop_last,
    )


def _validate_embedding_coverage(
    examples: tuple[RetrievalIndexExample, ...],
    embeddings: EmbeddingCache,
) -> None:
    for example in examples:
        item_ids = (
            *example.partial_item_ids,
            example.positive_item_id,
            *example.negative_item_ids,
        )
        for item_id in item_ids:
            if item_id not in embeddings:
                raise ValueError(
                    f"missing embedding for item {item_id!r} "
                    f"in {example.example_id}"
                )


def _dataset_request(
    config: RetrievalDataConfig,
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
    caches: tuple[EmbeddingCache, ...],
) -> None:
    fingerprints = {cache.model_fingerprint for cache in caches}
    if "" in fingerprints:
        raise ValueError("embedding manifests must contain model_fingerprint")
    if len(fingerprints) != 1:
        raise ValueError(
            "train and validation embedding caches use different models"
        )


def as_single_item_outfits(items: tuple[Any, ...]) -> tuple[Any, ...]:
    """Convert raw items or embedding vectors to one-item outfit inputs."""
    single_item_outfits: list[Any] = []
    for item in items:
        if isinstance(item, Tensor):
            single_item_outfits.append(item.unsqueeze(0))
        elif isinstance(item, OutfitItem):
            single_item_outfits.append((item,))
        else:
            raise TypeError("items must contain tensors or OutfitItem objects")
    return tuple(single_item_outfits)
