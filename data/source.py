"""Dataset-source contracts exposed to project workflows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, TypeVar

from torch.utils.data import Dataset

from .transforms import ImageTransform
from .types import (
    CompatibilityExample,
    CompatibilityIndexExample,
    FashionItem,
    RetrievalExample,
)

DEFAULT_DATASET_NAME = "polyvore"
_Item_co = TypeVar("_Item_co", covariant=True)


class IndexedDataset(Protocol[_Item_co]):
    """Minimal sized, indexable dataset contract."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int, /) -> _Item_co: ...


class DataSplit(str, Enum):
    """Task-independent dataset partitions used by project workflows."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class DatasetDescriptor:
    """Stable identity and supported subsets for one dataset source."""

    name: str
    dataset_id: str
    default_root: Path
    subsets: tuple[str, ...]
    default_subset: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("dataset name cannot be empty")
        if not self.dataset_id.strip():
            raise ValueError("dataset_id cannot be empty")
        if not self.subsets:
            raise ValueError("dataset must declare at least one subset")
        if any(not subset.strip() for subset in self.subsets):
            raise ValueError("dataset subsets cannot contain empty values")
        if len(set(self.subsets)) != len(self.subsets):
            raise ValueError("dataset subsets must be unique")
        if self.default_subset not in self.subsets:
            raise ValueError("default_subset must be a supported subset")

    def validate_subset(self, subset: str) -> str:
        """Return normalized subset or reject unsupported values."""
        selected = subset.strip()
        if selected not in self.subsets:
            choices = ", ".join(self.subsets)
            raise ValueError(
                f"dataset {self.name!r} subset must be one of: {choices}"
            )
        return selected


@dataclass(frozen=True)
class DatasetRequest:
    """Location and partition requested from a dataset source."""

    subset: str
    split: DataSplit
    root: Path
    cache_dir: Path | None = None
    token: bool | str | None = True

    def __post_init__(self) -> None:
        if not self.subset.strip():
            raise ValueError("dataset subset cannot be empty")
        if not isinstance(self.split, DataSplit):
            raise TypeError("split must be a DataSplit")


@dataclass(frozen=True)
class DatasetDownloadRequest:
    """Settings for downloading one complete dataset source."""

    output_dir: Path
    cache_dir: Path | None = None
    revision: str | None = None
    token: bool | str | None = True
    max_workers: int = 8
    force_download: bool = False

    def __post_init__(self) -> None:
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise NotADirectoryError(
                f"output path is not a directory: {self.output_dir}"
            )


class DatasetSource(Protocol):
    """Boundary between project workflows and dataset implementations."""

    @property
    def descriptor(self) -> DatasetDescriptor: ...

    def item_dataset(
        self,
        request: DatasetRequest,
        image_transform: ImageTransform,
    ) -> Dataset[FashionItem]: ...

    def compatibility_dataset(
        self,
        request: DatasetRequest,
        image_transform: ImageTransform,
    ) -> Dataset[CompatibilityExample]: ...

    def compatibility_index_dataset(
        self,
        request: DatasetRequest,
    ) -> IndexedDataset[CompatibilityIndexExample]: ...

    def retrieval_dataset(
        self,
        request: DatasetRequest,
        image_transform: ImageTransform,
    ) -> Dataset[RetrievalExample]: ...

    def download(self, request: DatasetDownloadRequest) -> Path: ...


def available_dataset_names() -> tuple[str, ...]:
    """Return names accepted by ``get_dataset_source``."""
    return (DEFAULT_DATASET_NAME,)


def get_dataset_source(name: str) -> DatasetSource:
    """Resolve a registered source without exposing its implementation."""
    selected = name.strip().lower()
    if selected == DEFAULT_DATASET_NAME:
        from .polyvore.source import PolyvoreSource

        return PolyvoreSource()
    choices = ", ".join(available_dataset_names())
    raise ValueError(f"dataset must be one of: {choices}")


def resolve_dataset_name(dataset_id: str) -> str:
    """Map persistent dataset ID to registered source name."""
    selected = dataset_id.strip()
    for name in available_dataset_names():
        source = get_dataset_source(name)
        if source.descriptor.dataset_id == selected:
            return name
    raise ValueError(f"unsupported dataset_id: {dataset_id!r}")
