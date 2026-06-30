import json
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from ..batch import OutfitExample
from ..transforms import ImageTransform, build_image_transform


class OutfitDataset(Dataset[OutfitExample]):
    """Loads outfits described by a small, model-independent JSON manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path,
        image_transform: ImageTransform | None = None,
    ) -> None:
        self._manifest_path = Path(manifest_path)
        self._image_root = Path(image_root)
        self._image_transform: ImageTransform = (
            image_transform or build_image_transform()
        )
        self._records = self._load_records(self._manifest_path)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> OutfitExample:
        record = self._records[index]
        images: list[Tensor] = []
        descriptions: list[str] = []

        for item in record["items"]:
            image_path = self._image_root / item["image"]
            with Image.open(image_path) as image:
                images.append(self._image_transform(image.convert("RGB")))
            descriptions.append(item["text"])

        return OutfitExample(
            outfit_id=record["outfit_id"],
            images=torch.stack(images),
            descriptions=tuple(descriptions),
        )

    @staticmethod
    def _load_records(manifest_path: Path) -> list[dict[str, Any]]:
        with manifest_path.open(encoding="utf-8") as manifest_file:
            records = json.load(manifest_file)

        if not isinstance(records, list):
            raise ValueError("manifest root must be a list of outfits")
        for record in records:
            OutfitDataset._validate_record(record)
        return records

    @staticmethod
    def _validate_record(record: Any) -> None:
        if not isinstance(record, dict):
            raise ValueError("each outfit must be a JSON object")
        if not isinstance(record.get("outfit_id"), str):
            raise ValueError("each outfit requires a string outfit_id")

        items = record.get("items")
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)) or not items:
            raise ValueError("each outfit requires a non-empty items list")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("each item must be a JSON object")
            if not isinstance(item.get("image"), str):
                raise ValueError("each item requires a string image path")
            if not isinstance(item.get("text"), str):
                raise ValueError("each item requires a string text description")
