"""Download and verify Polyvore resources without constructing batches."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .rows import ItemRows

if TYPE_CHECKING:
    from datasets import Dataset

LOGGER = logging.getLogger("data.polyvore")
DEFAULT_DATASET_ROOT = Path("datasets/polyvore-outfits")


class PolyvoreVariant(str, Enum):
    DISJOINT = "disjoint"
    NONDISJOINT = "nondisjoint"


class PolyvoreSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

    @property
    def file_stem(self) -> str:
        return "valid" if self is PolyvoreSplit.VALIDATION else self.value


class PolyvoreTask(str, Enum):
    ITEMS = "items"
    COMPATIBILITY = "compatibility"
    RETRIEVAL = "retrieval"


@dataclass(frozen=True)
class PolyvoreResources:
    """Local or downloaded rows and annotation paths for one data task."""

    item_rows: ItemRows | None = None
    metadata_path: Path | None = None
    outfits_path: Path | None = None
    compatibility_path: Path | None = None
    retrieval_path: Path | None = None


POLYVORE_DATASET_ID = "mvasil/polyvore-outfits"


def download_polyvore_resources(
    *,
    task: PolyvoreTask | str,
    variant: PolyvoreVariant | str = PolyvoreVariant.DISJOINT,
    split: PolyvoreSplit | str = PolyvoreSplit.TRAIN,
    token: bool | str | None = True,
    cache_dir: str | Path | None = None,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    include_items: bool = True,
    include_metadata: bool | None = None,
) -> PolyvoreResources:
    """Load local resources first, downloading only missing files."""
    selected_task = _coerce_enum(task, PolyvoreTask, "task")
    selected_variant = _coerce_enum(variant, PolyvoreVariant, "variant")
    selected_split = _coerce_enum(split, PolyvoreSplit, "split")
    selected_cache = str(cache_dir) if cache_dir is not None else None
    selected_root = Path(dataset_root)

    item_rows: ItemRows | None = None
    metadata_path: Path | None = None
    selected_include_metadata = (
        include_items if include_metadata is None else include_metadata
    )
    if include_items:
        item_rows = _load_item_rows(
            variant=selected_variant,
            split=selected_split,
            token=token,
            cache_dir=selected_cache,
            dataset_root=selected_root,
        )
        _verify_item_rows(item_rows)
    if selected_include_metadata:
        metadata_path = _resolve_file(
            "polyvore_item_metadata.json",
            dataset_root=selected_root,
            token=token,
            cache_dir=selected_cache,
        )
        _verify_file(metadata_path, "item metadata")

    outfits_path: Path | None = None
    compatibility_path: Path | None = None
    retrieval_path: Path | None = None
    if selected_task is not PolyvoreTask.ITEMS:
        outfits_path = _resolve_file(
            f"{selected_variant.value}/{selected_split.file_stem}.json",
            dataset_root=selected_root,
            token=token,
            cache_dir=selected_cache,
        )
        _verify_file(outfits_path, "outfit mapping")

    if selected_task is PolyvoreTask.COMPATIBILITY:
        compatibility_path = _resolve_file(
            f"{selected_variant.value}/"
            f"compatibility_{selected_split.file_stem}.txt",
            dataset_root=selected_root,
            token=token,
            cache_dir=selected_cache,
        )
        _verify_file(compatibility_path, "compatibility annotations")
    elif selected_task is PolyvoreTask.RETRIEVAL:
        retrieval_path = _resolve_file(
            f"{selected_variant.value}/"
            f"fill_in_blank_{selected_split.file_stem}.json",
            dataset_root=selected_root,
            token=token,
            cache_dir=selected_cache,
        )
        _verify_file(retrieval_path, "retrieval annotations")

    return PolyvoreResources(
        item_rows=item_rows,
        metadata_path=metadata_path,
        outfits_path=outfits_path,
        compatibility_path=compatibility_path,
        retrieval_path=retrieval_path,
    )


def _load_item_rows(
    *,
    variant: PolyvoreVariant,
    split: PolyvoreSplit,
    token: bool | str | None,
    cache_dir: str | None,
    dataset_root: Path,
) -> ItemRows:
    parquet_path = (
        dataset_root
        / "data"
        / variant.value
        / f"{split.value}.parquet"
    )
    if parquet_path.is_file():
        LOGGER.info("polyvore_source=local path=%s", parquet_path)
        return _load_local_parquet(parquet_path)

    cached_rows = _load_cached_item_rows(
        variant=variant,
        split=split,
        cache_dir=cache_dir,
    )
    if cached_rows is not None:
        return cached_rows

    LOGGER.info(
        "polyvore_source=local status=missing path=%s",
        parquet_path,
    )
    return _load_remote_item_rows(
        variant=variant,
        split=split,
        token=token,
        cache_dir=cache_dir,
    )


def _load_local_parquet(path: Path) -> Dataset:
    try:
        from datasets import Dataset
    except ImportError as error:
        raise ImportError(
            "Polyvore loading requires the 'datasets' package"
        ) from error
    return Dataset.from_parquet(str(path))


def _load_cached_item_rows(
    *,
    variant: PolyvoreVariant,
    split: PolyvoreSplit,
    cache_dir: str | None,
) -> Dataset | None:
    try:
        from datasets import Dataset, config
    except ImportError as error:
        raise ImportError(
            "Polyvore loading requires the 'datasets' package"
        ) from error

    cache_root = Path(cache_dir) if cache_dir else Path(config.HF_DATASETS_CACHE)
    dataset_cache = cache_root / "mvasil___polyvore-outfits" / variant.value
    filename = f"polyvore-outfits-{split.value}.arrow"
    candidates = tuple(dataset_cache.glob(f"**/{filename}"))
    if not candidates:
        return None
    selected_path = max(
        candidates,
        key=lambda candidate: candidate.stat().st_mtime_ns,
    )
    LOGGER.info("polyvore_source=cache path=%s", selected_path)
    return Dataset.from_file(str(selected_path))


def _load_remote_item_rows(
    *,
    variant: PolyvoreVariant,
    split: PolyvoreSplit,
    token: bool | str | None,
    cache_dir: str | None,
) -> Dataset:
    try:
        from datasets import Dataset, load_dataset
    except ImportError as error:
        raise ImportError(
            "Polyvore download requires the 'datasets' package"
        ) from error

    LOGGER.info(
        "polyvore_source=huggingface variant=%s split=%s",
        variant.value,
        split.value,
    )
    loaded_dataset = load_dataset(
        POLYVORE_DATASET_ID,
        variant.value,
        split=split.value,
        token=token,
        cache_dir=cache_dir,
    )
    if not isinstance(loaded_dataset, Dataset):
        raise TypeError("split loading must return a datasets.Dataset")
    return loaded_dataset


def _resolve_file(
    filename: str,
    *,
    dataset_root: Path,
    token: bool | str | None,
    cache_dir: str | None,
) -> Path:
    local_path = dataset_root / Path(filename)
    if local_path.is_file():
        LOGGER.info("polyvore_source=local path=%s", local_path)
        return local_path
    LOGGER.info(
        "polyvore_source=local status=missing path=%s",
        local_path,
    )
    return _download_file(
        filename,
        token=token,
        cache_dir=cache_dir,
    )


def _download_file(
    filename: str,
    *,
    token: bool | str | None,
    cache_dir: str | None,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError as error:
        raise ImportError(
            "Polyvore download requires the 'huggingface-hub' package"
        ) from error

    download_arguments = {
        "repo_id": POLYVORE_DATASET_ID,
        "repo_type": "dataset",
        "filename": filename,
        "token": token,
        "cache_dir": cache_dir,
    }
    try:
        cached_path = Path(
            hf_hub_download(
                **download_arguments,
                local_files_only=True,
            )
        )
    except LocalEntryNotFoundError:
        LOGGER.info("polyvore_source=huggingface filename=%s", filename)
    else:
        LOGGER.info("polyvore_source=cache path=%s", cached_path)
        return cached_path

    return Path(
        hf_hub_download(
            **download_arguments,
        )
    )


def _verify_item_rows(item_rows: ItemRows) -> None:
    if len(item_rows) == 0:
        raise ValueError("downloaded Polyvore item split is empty")

    columns = getattr(item_rows, "column_names", None)
    if columns is not None:
        missing = {"item_id", "image"}.difference(columns)
    else:
        first_row = item_rows[0]
        if not isinstance(first_row, dict):
            raise TypeError("Polyvore item rows must be mappings")
        missing = {"item_id", "image"}.difference(first_row)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"Polyvore item rows miss columns: {missing_names}")


def _verify_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"downloaded {description} file does not exist: {path}")


def _coerce_enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    try:
        return enum_type(value)
    except ValueError as error:
        choices = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{name} must be one of: {choices}") from error
