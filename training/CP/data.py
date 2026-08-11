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
    download_polyvore_resources,
    load_outfit_token_index,
)
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
    """Build loaders for classic raw inputs or precomputed CLIP features."""
    if config.feature_mode is FeatureMode.CLASSIC:
        return _build_classic_loaders(config, token=token)
    return _build_clip_loaders(config, token=token)


def _build_classic_loaders(
    config: CPTrainingConfig,
    *,
    token: bool | str | None,
) -> CompatibilityLoaders:
    loader_config = LoaderConfig(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        seed=config.seed,
    )
    return CompatibilityLoaders(
        train=build_polyvore_compatibility_loader(
            image_transform=build_resnet18_transform(augment=True),
            variant=config.variant,
            split=PolyvoreSplit.TRAIN,
            config=loader_config,
            shuffle=True,
            token=token,
            cache_dir=config.cache_dir,
        ),
        validation=build_polyvore_compatibility_loader(
            image_transform=build_resnet18_transform(augment=False),
            variant=config.variant,
            split=PolyvoreSplit.VALIDATION,
            config=loader_config,
            shuffle=False,
            token=token,
            cache_dir=config.cache_dir,
        ),
    )


def _build_clip_loaders(
    config: CPTrainingConfig,
    *,
    token: bool | str | None,
) -> CompatibilityLoaders:
    caches: dict[PolyvoreSplit, EmbeddingCache] = {}
    datasets: dict[PolyvoreSplit, PrecomputedCompatibilityDataset] = {}
    for split in (PolyvoreSplit.TRAIN, PolyvoreSplit.VALIDATION):
        cache = EmbeddingCache(
            config.embedding_root / config.variant.value / split.value,
            expected_variant=config.variant.value,
            expected_split=split.value,
        )
        if cache.embedding_dim != config.model_config.model_dim:
            raise ValueError(
                f"{split.value} embedding_dim must be "
                f"{config.model_config.model_dim}, "
                f"got {cache.embedding_dim}"
            )
        caches[split] = cache

        resources = download_polyvore_resources(
            task=PolyvoreTask.COMPATIBILITY,
            variant=config.variant,
            split=split,
            token=token,
            cache_dir=config.cache_dir,
        )
        if resources.compatibility_path is None or resources.outfits_path is None:
            raise RuntimeError("downloaded CP resources are incomplete")
        datasets[split] = PrecomputedCompatibilityDataset(
            resources.compatibility_path,
            resources.outfits_path,
            cache,
        )

    _require_matching_fingerprints(caches)
    generator = torch.Generator().manual_seed(config.seed)
    common_loader_args = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "collate_fn": collate_compatibility_embeddings,
        "pin_memory": config.pin_memory,
        "persistent_workers": config.num_workers > 0,
    }
    return CompatibilityLoaders(
        train=DataLoader(
            datasets[PolyvoreSplit.TRAIN],
            shuffle=True,
            generator=generator,
            **common_loader_args,
        ),
        validation=DataLoader(
            datasets[PolyvoreSplit.VALIDATION],
            shuffle=False,
            **common_loader_args,
        ),
    )


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
