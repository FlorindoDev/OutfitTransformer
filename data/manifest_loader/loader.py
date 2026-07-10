from pathlib import Path
from typing import cast

from torch.utils.data import DataLoader

from ..batch import OutfitBatch, collate_outfits
from ..transforms import ImageTransform, build_image_transform
from .dataset import OutfitDataset


def create_manifest_outfit_loader(
    manifest_path: str | Path,
    image_root: str | Path,
    *,
    batch_size: int = 8,
    shuffle: bool = False,
    workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    image_transform: ImageTransform | None = None,
) -> DataLoader[OutfitBatch]:
    """Create a DataLoader for a local outfit manifest."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if workers < 0:
        raise ValueError("workers must be non-negative")

    dataset = OutfitDataset(
        manifest_path=manifest_path,
        image_root=image_root,
        image_transform=image_transform or build_image_transform(),
    )
    return cast(
        DataLoader[OutfitBatch],
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_outfits,
        ),
    )
