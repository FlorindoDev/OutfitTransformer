from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


PolyvoreVariant = Literal["nondisjoint", "disjoint"]
PolyvoreSplit = Literal["train", "validation", "test"]

_DATASET_ID = "mvasil/polyvore-outfits"
_TEXT_SPLIT_NAMES: dict[PolyvoreSplit, str] = {
    "train": "train",
    "validation": "valid",
    "test": "test",
}


@dataclass(frozen=True)
class PolyvoreResources:
    outfit_rows: Any
    compatibility_path: Path
    item_metadata_path: Path
    outfit_mapping_path: Path


def download_polyvore_resources(
    variant: PolyvoreVariant = "nondisjoint",
    split: PolyvoreSplit = "train",
    *,
    token: bool | str | None = True,
    cache_dir: str | Path | None = None,
) -> PolyvoreResources:
    """Download or reuse every Polyvore resource required by one split."""
    _validate_selection(variant, split)
    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError(
            "Hugging Face loading requires the 'datasets' and "
            "'huggingface_hub' packages"
        ) from error

    normalized_cache_dir = str(cache_dir) if cache_dir is not None else None
    text_split = _TEXT_SPLIT_NAMES[split]
    outfit_rows = load_dataset(
        _DATASET_ID,
        variant,
        split=split,
        token=token,
        cache_dir=normalized_cache_dir,
    )
    compatibility_path = hf_hub_download(
        repo_id=_DATASET_ID,
        filename=f"{variant}/compatibility_{text_split}.txt",
        repo_type="dataset",
        token=token,
        cache_dir=normalized_cache_dir,
    )
    item_metadata_path = hf_hub_download(
        repo_id=_DATASET_ID,
        filename="polyvore_item_metadata.json",
        repo_type="dataset",
        token=token,
        cache_dir=normalized_cache_dir,
    )
    outfit_mapping_path = hf_hub_download(
        repo_id=_DATASET_ID,
        filename=f"{variant}/{text_split}.json",
        repo_type="dataset",
        token=token,
        cache_dir=normalized_cache_dir,
    )
    return PolyvoreResources(
        outfit_rows=outfit_rows,
        compatibility_path=Path(compatibility_path),
        item_metadata_path=Path(item_metadata_path),
        outfit_mapping_path=Path(outfit_mapping_path),
    )


def _validate_selection(variant: str, split: str) -> None:
    if variant not in ("nondisjoint", "disjoint"):
        raise ValueError("variant must be 'nondisjoint' or 'disjoint'")
    if split not in ("train", "validation", "test"):
        raise ValueError("split must be 'train', 'validation', or 'test'")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download every resource required by one Polyvore split",
    )
    parser.add_argument(
        "--variant",
        choices=("nondisjoint", "disjoint"),
        default="disjoint",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="train",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    resources = download_polyvore_resources(
        variant=args.variant,
        split=args.split,
        cache_dir=args.cache_dir,
    )
    print(f"examples={len(resources.outfit_rows)}")
    print(f"compatibility={resources.compatibility_path}")
    print(f"metadata={resources.item_metadata_path}")
    print(f"mapping={resources.outfit_mapping_path}")


if __name__ == "__main__":
    main()
