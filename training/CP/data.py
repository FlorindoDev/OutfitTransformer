"""Polyvore CP data for classic inputs and precomputed CLIP embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from data import (
    LoaderConfig,
    build_polyvore_compatibility_loader,
    build_resnet18_transform,
)
from data.polyvore import (
    PolyvoreSplit,
    PolyvoreTask,
    PolyvoreVariant,
    download_polyvore_resources,
    load_outfit_token_index,
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

    variant: PolyvoreVariant
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
            variant=config.variant,
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
        if not isinstance(self.variant, PolyvoreVariant):
            raise TypeError("variant must be a PolyvoreVariant")
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
        compatibility_path: str | Path,
        outfits_path: str | Path,
        embeddings: EmbeddingCache,
    ) -> None:
        self._embeddings = embeddings
        token_index = load_outfit_token_index(outfits_path)
        self._annotations = _load_annotations(
            compatibility_path,
            token_index,
            embeddings,
        )

    def __len__(self) -> int:
        return len(self._annotations)

    def __getitem__(self, index: int) -> CompatibilityEmbeddingExample:
        example_id, label, item_ids = self._annotations[index]
        return CompatibilityEmbeddingExample(
            example_id=example_id,
            outfit=torch.stack([self._embeddings[item_id] for item_id in item_ids]),
            label=label,
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
                split=PolyvoreSplit.TRAIN,
                shuffle=True,
                token=token,
            ),
            validation=_build_classic_loader(
                data_config,
                split=PolyvoreSplit.VALIDATION,
                shuffle=False,
                token=token,
            ),
        )

    train_loader, train_cache = _build_clip_loader(
        data_config,
        split=PolyvoreSplit.TRAIN,
        shuffle=True,
        token=token,
    )
    validation_loader, validation_cache = _build_clip_loader(
        data_config,
        split=PolyvoreSplit.VALIDATION,
        shuffle=False,
        token=token,
    )
    _require_matching_fingerprints(
        {
            PolyvoreSplit.TRAIN: train_cache,
            PolyvoreSplit.VALIDATION: validation_cache,
        }
    )
    return CompatibilityLoaders(
        train=train_loader,
        validation=validation_loader,
    )


def build_compatibility_loader(
    config: CompatibilityDataConfig,
    *,
    split: PolyvoreSplit,
    token: bool | str | None = True,
    shuffle: bool = False,
) -> DataLoader[Any]:
    """Build one CP loader for a requested dataset split."""
    config.validate()
    if not isinstance(split, PolyvoreSplit):
        raise TypeError("split must be a PolyvoreSplit")
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
    split: PolyvoreSplit,
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
    return build_polyvore_compatibility_loader(
        image_transform=build_resnet18_transform(
            augment=split is PolyvoreSplit.TRAIN
        ),
        variant=config.variant,
        split=split,
        config=loader_config,
        shuffle=shuffle,
        token=token,
        cache_dir=config.cache_dir,
        dataset_root=config.dataset_root,
    )


def _build_clip_loader(
    config: CompatibilityDataConfig,
    *,
    split: PolyvoreSplit,
    shuffle: bool,
    token: bool | str | None,
) -> tuple[DataLoader[Any], EmbeddingCache]:
    cache = EmbeddingCache(
        config.embedding_root / config.variant.value / split.value,
        expected_variant=config.variant.value,
        expected_split=split.value,
    )
    if cache.embedding_dim != config.model_config.model_dim:
        raise ValueError(
            f"{split.value} embedding_dim must be "
            f"{config.model_config.model_dim}, got {cache.embedding_dim}"
        )

    resources = download_polyvore_resources(
        task=PolyvoreTask.COMPATIBILITY,
        variant=config.variant,
        split=split,
        token=token,
        cache_dir=config.cache_dir,
        dataset_root=config.dataset_root,
        include_items=False,
    )
    if resources.compatibility_path is None or resources.outfits_path is None:
        raise RuntimeError("downloaded CP resources are incomplete")
    dataset = PrecomputedCompatibilityDataset(
        resources.compatibility_path,
        resources.outfits_path,
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


def _load_annotations(
    path: str | Path,
    token_index: dict[str, str],
    embeddings: EmbeddingCache,
) -> tuple[tuple[str, int, tuple[str, ...]], ...]:
    selected_path = Path(path)
    if not selected_path.is_file():
        raise FileNotFoundError(
            f"compatibility annotation file does not exist: {selected_path}"
        )

    annotations: list[tuple[str, int, tuple[str, ...]]] = []
    with selected_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            fields = line.split()
            if not fields:
                continue
            if fields[0] not in {"0", "1"} or len(fields) < 2:
                raise ValueError(f"invalid compatibility line {line_number}")
            item_ids = tuple(
                _resolve_item_id(token, token_index, embeddings, line_number)
                for token in fields[1:]
            )
            annotations.append(
                (f"compatibility:{line_number}", int(fields[0]), item_ids)
            )
    if not annotations:
        raise ValueError("compatibility annotation file contains no examples")
    return tuple(annotations)


def _resolve_item_id(
    token: str,
    token_index: dict[str, str],
    embeddings: EmbeddingCache,
    line_number: int,
) -> str:
    try:
        item_id = token_index[token]
    except KeyError as error:
        raise ValueError(
            f"unknown outfit token {token!r} on line {line_number}"
        ) from error
    if item_id not in embeddings:
        raise ValueError(
            f"missing embedding for item {item_id!r} on line {line_number}"
        )
    return item_id


def _require_matching_fingerprints(
    caches: dict[PolyvoreSplit, EmbeddingCache],
) -> None:
    fingerprints = {cache.model_fingerprint for cache in caches.values()}
    if "" in fingerprints:
        raise ValueError("embedding manifests must contain model_fingerprint")
    if len(fingerprints) != 1:
        raise ValueError("train and validation embeddings use different models")
