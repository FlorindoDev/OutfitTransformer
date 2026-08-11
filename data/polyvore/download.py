"""Download and verify Polyvore resources without constructing batches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


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
    """Downloaded rows and annotation paths needed by one data task."""

    item_rows: Sequence[Any]
    metadata_path: Path
    outfits_path: Path | None = None
    compatibility_path: Path | None = None
    retrieval_path: Path | None = None


_DATASET_ID = "mvasil/polyvore-outfits"


def download_polyvore_resources(
    *,
    task: PolyvoreTask | str,
    variant: PolyvoreVariant | str = PolyvoreVariant.DISJOINT,
    split: PolyvoreSplit | str = PolyvoreSplit.TRAIN,
    token: bool | str | None = True,
    cache_dir: str | Path | None = None,
) -> PolyvoreResources:
    """Fetch the minimum catalog and annotation files required by ``task``."""
    selected_task = _coerce_enum(task, PolyvoreTask, "task")
    selected_variant = _coerce_enum(variant, PolyvoreVariant, "variant")
    selected_split = _coerce_enum(split, PolyvoreSplit, "split")
    selected_cache = str(cache_dir) if cache_dir is not None else None

    item_rows = _load_item_rows(
        variant=selected_variant,
        split=selected_split,
        token=token,
        cache_dir=selected_cache,
    )
    _verify_item_rows(item_rows)

    metadata_path = _download_file(
        "polyvore_item_metadata.json",
        token=token,
        cache_dir=selected_cache,
    )
    _verify_file(metadata_path, "item metadata")

    outfits_path: Path | None = None
    compatibility_path: Path | None = None
    retrieval_path: Path | None = None
    if selected_task is not PolyvoreTask.ITEMS:
        outfits_path = _download_file(
            f"{selected_variant.value}/{selected_split.file_stem}.json",
            token=token,
            cache_dir=selected_cache,
        )
        _verify_file(outfits_path, "outfit mapping")

    if selected_task is PolyvoreTask.COMPATIBILITY:
        compatibility_path = _download_file(
            f"{selected_variant.value}/"
            f"compatibility_{selected_split.file_stem}.txt",
            token=token,
            cache_dir=selected_cache,
        )
        _verify_file(compatibility_path, "compatibility annotations")
    elif selected_task is PolyvoreTask.RETRIEVAL:
        retrieval_path = _download_file(
            f"{selected_variant.value}/"
            f"fill_in_blank_{selected_split.file_stem}.json",
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
) -> Sequence[Any]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(
            "Polyvore download requires the 'datasets' package"
        ) from error

    return load_dataset(
        _DATASET_ID,
        variant.value,
        split=split.value,
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
    except ImportError as error:
        raise ImportError(
            "Polyvore download requires the 'huggingface-hub' package"
        ) from error

    return Path(
        hf_hub_download(
            repo_id=_DATASET_ID,
            repo_type="dataset",
            filename=filename,
            token=token,
            cache_dir=cache_dir,
        )
    )


def _verify_item_rows(item_rows: Sequence[Any]) -> None:
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
