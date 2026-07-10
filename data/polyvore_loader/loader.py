from pathlib import Path
from typing import cast

from torch.utils.data import DataLoader

from ..batch import CompatibilityBatch, collate_compatibility
from ..transforms import ImageTransform, build_image_transform
from .dataset import PolyvoreCompatibilityDataset
from .download import (
    PolyvoreSplit,
    PolyvoreVariant,
    download_polyvore_resources,
)


def load_polyvore_compatibility_dataset(
    variant: PolyvoreVariant = "nondisjoint",
    split: PolyvoreSplit = "train",
    *,
    token: bool | str | None = True,
    cache_dir: str | Path | None = None,
    image_transform: ImageTransform | None = None,
) -> PolyvoreCompatibilityDataset:
    """Download Polyvore resources and adapt them to model-ready examples."""
    resources = download_polyvore_resources(
        variant=variant,
        split=split,
        token=token,
        cache_dir=cache_dir,
    )
    return PolyvoreCompatibilityDataset(
        outfit_rows=resources.outfit_rows,
        compatibility_path=resources.compatibility_path,
        image_transform=image_transform or build_image_transform(),
        item_metadata_path=resources.item_metadata_path,
        outfit_mapping_path=resources.outfit_mapping_path,
    )


def create_polyvore_compatibility_loader(
    variant: PolyvoreVariant = "nondisjoint",
    split: PolyvoreSplit = "train",
    *,
    batch_size: int = 32,
    shuffle: bool | None = None,
    workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    token: bool | str | None = True,
    cache_dir: str | Path | None = None,
    image_transform: ImageTransform | None = None,
) -> DataLoader[CompatibilityBatch]:
    """Create a DataLoader returning CompatibilityBatch instances."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if workers < 0:
        raise ValueError("workers must be non-negative")

    dataset = load_polyvore_compatibility_dataset(
        variant=variant,
        split=split,
        token=token,
        cache_dir=cache_dir,
        image_transform=image_transform,
    )
    should_shuffle = split == "train" if shuffle is None else shuffle
    return cast(
        DataLoader[CompatibilityBatch],
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            num_workers=workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_compatibility,
        ),
    )
