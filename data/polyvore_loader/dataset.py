from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from ..batch import OutfitBatch, OutfitExample, collate_outfits
from ..transforms import ImageTransform, build_image_transform

PolyvoreVariant = Literal["nondisjoint", "disjoint"]
PolyvoreSplit = Literal["train", "validation", "test"]

_DATASET_ID = "mvasil/polyvore-outfits"
_TEXT_SPLIT_NAMES: dict[PolyvoreSplit, str] = {
    "train": "train",
    "validation": "valid",
    "test": "test",
}


@dataclass(frozen=True)
class CompatibilityExample:
    outfit_id: str
    images: Tensor
    descriptions: tuple[str, ...]
    label: float

    def __post_init__(self) -> None:
        OutfitExample(self.outfit_id, self.images, self.descriptions)
        if self.label not in (0.0, 1.0):
            raise ValueError("compatibility label must be 0 or 1")


@dataclass(frozen=True)
class CompatibilityBatch:
    outfits: OutfitBatch
    labels: Tensor

    @property
    def outfit_ids(self) -> tuple[str, ...]:
        return self.outfits.outfit_ids

    @property
    def images(self) -> Tensor:
        return self.outfits.images

    @property
    def descriptions(self) -> tuple[tuple[str, ...], ...]:
        return self.outfits.descriptions

    @property
    def padding_mask(self) -> Tensor:
        return self.outfits.padding_mask

    def to(self, device: torch.device | str) -> "CompatibilityBatch":
        return CompatibilityBatch(
            outfits=self.outfits.to(device),
            labels=self.labels.to(device),
        )


@dataclass(frozen=True)
class _CompatibilityQuestion:
    label: float
    item_tokens: tuple[str, ...]
    line_number: int


class PolyvoreCompatibilityDataset(Dataset[CompatibilityExample]):
    """Prepare official Polyvore compatibility questions for CP training."""

    def __init__(
        self,
        outfit_rows: Any,
        compatibility_path: str | Path,
        image_transform: ImageTransform | None = None,
        image_root: str | Path | None = None,
        item_metadata_path: str | Path | None = None,
        outfit_mapping_path: str | Path | None = None,
    ) -> None:
        self._outfit_rows = outfit_rows
        self._compatibility_path = Path(compatibility_path)
        self._image_transform: ImageTransform = (
            image_transform or build_image_transform()
        )
        self._image_root = Path(image_root) if image_root is not None else None
        self._item_metadata = self._load_item_metadata(item_metadata_path)
        self._token_to_item = self._load_outfit_mapping(outfit_mapping_path)
        self._questions = self._load_questions(self._compatibility_path)
        self._uses_mapped_item_rows = bool(self._token_to_item)
        self._uses_item_rows = (
            False
            if self._uses_mapped_item_rows
            else self._detect_item_rows(outfit_rows)
        )
        self._item_id_to_row = (
            self._index_rows_by_item_id(outfit_rows)
            if self._uses_mapped_item_rows
            else {}
        )
        self._set_to_row = (
            {}
            if self._uses_mapped_item_rows or self._uses_item_rows
            else self._index_outfit_rows(outfit_rows)
        )
        self._token_to_row = (
            self._index_item_rows(outfit_rows) if self._uses_item_rows else {}
        )

    def __len__(self) -> int:
        return len(self._questions)

    def __getitem__(self, index: int) -> CompatibilityExample:
        question = self._questions[index]
        source_rows: dict[str, Mapping[str, Any]] = {}
        images: list[Tensor] = []
        descriptions: list[str] = []

        for token in question.item_tokens:
            if self._uses_mapped_item_rows:
                item = self._get_mapped_item(token)
                row = self._get_item_image_row(item)
                image = self._extract_image(row, item, position=0)
                images.append(self._image_transform(image))
                descriptions.append(
                    self._extract_description(row, item, position=0)
                )
                continue

            if self._uses_item_rows:
                row = self._get_item_row(token)
                item = row
                position = 0
                image = self._extract_image(row, item, position)
                images.append(self._image_transform(image))
                descriptions.append(
                    self._extract_description(row, item, position)
                )
                continue

            set_id, item_index = self._parse_item_token(token)
            row = source_rows.get(set_id)
            if row is None:
                row = self._get_outfit_row(set_id)
                source_rows[set_id] = row

            item, position = self._find_item(row, item_index, token)
            image = self._extract_image(row, item, position)
            images.append(self._image_transform(image))
            descriptions.append(self._extract_description(row, item, position))

        return CompatibilityExample(
            outfit_id=f"{self._compatibility_path.stem}:{question.line_number}",
            images=torch.stack(images),
            descriptions=tuple(descriptions),
            label=question.label,
        )

    @staticmethod
    def _load_outfit_mapping(
        path: str | Path | None,
    ) -> dict[str, Mapping[str, Any]]:
        if path is None:
            return {}
        with Path(path).open(encoding="utf-8") as mapping_file:
            mapping_data = json.load(mapping_file)
        return PolyvoreCompatibilityDataset._index_items_from_outfit_rows(
            mapping_data
        )

    @staticmethod
    def _load_questions(path: Path) -> list[_CompatibilityQuestion]:
        questions: list[_CompatibilityQuestion] = []
        with path.open(encoding="utf-8") as compatibility_file:
            for line_number, raw_line in enumerate(compatibility_file, start=1):
                parts = raw_line.split()
                if not parts:
                    continue
                if len(parts) < 3:
                    raise ValueError(
                        f"{path}:{line_number}: expected a label and at least two items"
                    )
                try:
                    label = float(parts[0])
                except ValueError as error:
                    raise ValueError(
                        f"{path}:{line_number}: compatibility label is not numeric"
                    ) from error
                if label not in (0.0, 1.0):
                    raise ValueError(
                        f"{path}:{line_number}: compatibility label must be 0 or 1"
                    )
                questions.append(
                    _CompatibilityQuestion(
                        label=label,
                        item_tokens=tuple(parts[1:]),
                        line_number=line_number,
                    )
                )

        if not questions:
            raise ValueError(f"{path} does not contain compatibility questions")
        return questions

    @staticmethod
    def _detect_item_rows(outfit_rows: Any) -> bool:
        column_names = set(getattr(outfit_rows, "column_names", ()))
        if column_names:
            return "items" not in column_names and {
                "set_id",
                "item_id",
                "index",
            }.issubset(column_names)
        if len(outfit_rows) == 0:
            return False
        first_row = outfit_rows[0]
        return (
            isinstance(first_row, Mapping)
            and "items" not in first_row
            and all(key in first_row for key in ("set_id", "item_id", "index"))
        )

    @staticmethod
    def _index_rows_by_item_id(item_rows: Any) -> dict[str, int]:
        column_names = getattr(item_rows, "column_names", ())
        if "item_id" in column_names:
            item_ids = item_rows["item_id"]
            return {
                str(item_id): row_index
                for row_index, item_id in enumerate(item_ids)
            }

        index: dict[str, int] = {}
        for row_index in range(len(item_rows)):
            row = item_rows[row_index]
            if not isinstance(row, Mapping):
                raise TypeError("Polyvore item rows must be mappings")
            item_id = row.get("item_id")
            if item_id is None:
                raise ValueError("each Polyvore item row requires an item_id")
            index[str(item_id)] = row_index
        return index

    @staticmethod
    def _index_items_from_outfit_rows(
        outfit_rows: Any,
    ) -> dict[str, Mapping[str, Any]]:
        index: dict[str, Mapping[str, Any]] = {}
        for row in PolyvoreCompatibilityDataset._iter_outfit_mapping_rows(
            outfit_rows
        ):
            set_id = PolyvoreCompatibilityDataset._extract_set_id(row)
            items = PolyvoreCompatibilityDataset._normalize_items(
                row.get("items")
            )
            for position, item in enumerate(items):
                item_index = item.get("index", position + 1)
                item_id = item.get("item_id")
                if item_id is None:
                    raise ValueError(
                        f"Polyvore outfit {set_id!r} has an item without item_id"
                    )
                normalized_item = dict(item)
                normalized_item["index"] = item_index
                normalized_item["item_id"] = str(item_id)
                index[f"{set_id}_{item_index}"] = normalized_item

        if not index:
            raise ValueError("Polyvore outfit mapping does not contain items")
        return index

    @staticmethod
    def _iter_outfit_mapping_rows(value: Any) -> list[Mapping[str, Any]]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if all(isinstance(row, Mapping) for row in value):
                return list(value)
        if isinstance(value, Mapping):
            if "items" in value:
                return [value]
            rows: list[Mapping[str, Any]] = []
            for set_id, row_value in value.items():
                if isinstance(row_value, Mapping):
                    row = dict(row_value)
                    row.setdefault("set_id", set_id)
                    rows.append(row)
                elif (
                    isinstance(row_value, Sequence)
                    and not isinstance(row_value, (str, bytes))
                ):
                    rows.append({"set_id": set_id, "items": row_value})
            if rows:
                return rows
        raise ValueError("Polyvore outfit mapping must contain outfit rows")

    @staticmethod
    def _extract_set_id(row: Mapping[str, Any]) -> str:
        for key in ("set_id", "outfit_id", "id"):
            value = row.get(key)
            if value is not None:
                return str(value)
        raise ValueError("each Polyvore outfit mapping row requires a set_id")

    @staticmethod
    def _index_outfit_rows(outfit_rows: Any) -> dict[str, int]:
        column_names = getattr(outfit_rows, "column_names", ())
        if "set_id" in column_names:
            set_ids = outfit_rows["set_id"]
            return {
                str(set_id): row_index
                for row_index, set_id in enumerate(set_ids)
            }

        index: dict[str, int] = {}
        for row_index in range(len(outfit_rows)):
            row = outfit_rows[row_index]
            if not isinstance(row, Mapping) or "set_id" not in row:
                raise ValueError(
                    "each Polyvore outfit row requires a set_id; "
                    "if rows contain only item_id/image, pass "
                    "outfit_mapping_path so compatibility tokens can be "
                    "resolved to item_id"
                )
            index[str(row["set_id"])] = row_index
        return index

    @staticmethod
    def _index_item_rows(item_rows: Any) -> dict[str, int]:
        column_names = getattr(item_rows, "column_names", ())
        if "set_id" in column_names and "index" in column_names:
            set_ids = item_rows["set_id"]
            item_indices = item_rows["index"]
            return {
                f"{set_id}_{item_index}": row_index
                for row_index, (set_id, item_index) in enumerate(
                    zip(set_ids, item_indices, strict=True)
                )
            }

        index: dict[str, int] = {}
        for row_index in range(len(item_rows)):
            row = item_rows[row_index]
            if not isinstance(row, Mapping):
                raise TypeError("Polyvore item rows must be mappings")
            token = f"{row['set_id']}_{row['index']}"
            index[token] = row_index
        return index

    def _get_outfit_row(self, set_id: str) -> Mapping[str, Any]:
        try:
            row_index = self._set_to_row[set_id]
        except KeyError as error:
            raise KeyError(
                f"compatibility question references unknown set_id {set_id!r}"
            ) from error

        row = self._outfit_rows[row_index]
        if not isinstance(row, Mapping):
            raise TypeError("Polyvore outfit rows must be mappings")
        return row

    def _get_item_row(self, token: str) -> Mapping[str, Any]:
        try:
            row_index = self._token_to_row[token]
        except KeyError as error:
            raise KeyError(
                f"compatibility question references unknown item {token!r}"
            ) from error
        row = self._outfit_rows[row_index]
        if not isinstance(row, Mapping):
            raise TypeError("Polyvore item rows must be mappings")
        return row

    def _get_mapped_item(self, token: str) -> Mapping[str, Any]:
        try:
            return self._token_to_item[token]
        except KeyError as error:
            raise KeyError(
                f"compatibility question references unknown item {token!r}"
            ) from error

    def _get_item_image_row(self, item: Mapping[str, Any]) -> Mapping[str, Any]:
        item_id = str(item.get("item_id", ""))
        try:
            row_index = self._item_id_to_row[item_id]
        except KeyError as error:
            raise KeyError(
                f"mapped Polyvore item_id {item_id!r} has no image row"
            ) from error
        row = self._outfit_rows[row_index]
        if not isinstance(row, Mapping):
            raise TypeError("Polyvore item rows must be mappings")
        return row

    @staticmethod
    def _parse_item_token(token: str) -> tuple[str, int]:
        try:
            set_id, raw_index = token.rsplit("_", maxsplit=1)
            item_index = int(raw_index)
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"invalid Polyvore item token {token!r}; expected set_id_index"
            ) from error
        if not set_id:
            raise ValueError(f"invalid Polyvore item token {token!r}")
        return set_id, item_index

    @staticmethod
    def _find_item(
        row: Mapping[str, Any],
        item_index: int,
        token: str,
    ) -> tuple[Mapping[str, Any], int]:
        items = PolyvoreCompatibilityDataset._normalize_items(row.get("items"))

        for position, item in enumerate(items):
            if str(item.get("index")) == str(item_index):
                return item, position

        fallback_position = item_index - 1
        if 0 <= fallback_position < len(items):
            return items[fallback_position], fallback_position
        raise KeyError(f"compatibility question references unknown item {token!r}")

    @staticmethod
    def _normalize_items(value: Any) -> list[Mapping[str, Any]]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if all(isinstance(item, Mapping) for item in value):
                return list(value)
        if isinstance(value, Mapping):
            lengths = [
                len(column)
                for column in value.values()
                if isinstance(column, Sequence)
                and not isinstance(column, (str, bytes))
            ]
            if lengths and len(set(lengths)) == 1:
                return [
                    {
                        key: column[position]
                        for key, column in value.items()
                        if isinstance(column, Sequence)
                        and not isinstance(column, (str, bytes))
                    }
                    for position in range(lengths[0])
                ]
        raise ValueError("Polyvore outfit row has no valid items sequence")

    def _extract_image(
        self,
        row: Mapping[str, Any],
        item: Mapping[str, Any],
        position: int,
    ) -> Image.Image:
        image_value = item.get("image")
        if (
            isinstance(image_value, str)
            and image_value.startswith(("http://", "https://"))
        ):
            image_value = None
        if image_value is None:
            image_value = self._aligned_value(row, ("images", "image"), position)
        if image_value is None:
            image_value = self._scalar_value(row, ("image", "images"))
        if image_value is None and self._image_root is not None:
            item_id = item.get("item_id")
            if item_id is not None:
                image_value = self._image_root / f"{item_id}.jpg"
        elif (
            self._image_root is not None
            and isinstance(image_value, (str, Path))
            and not Path(image_value).is_absolute()
        ):
            image_value = self._image_root / image_value
        if image_value is None:
            raise ValueError(
                f"item {item.get('item_id')!r} does not contain an image"
            )
        return self._open_image(image_value)

    def _extract_description(
        self,
        row: Mapping[str, Any],
        item: Mapping[str, Any],
        position: int,
    ) -> str:
        description = self._first_text(
            item,
            ("description", "text", "title", "name", "url_name"),
        )
        if description is None:
            aligned = self._aligned_value(
                row,
                ("descriptions", "texts", "titles", "names"),
                position,
            )
            if isinstance(aligned, str) and aligned.strip():
                description = aligned.strip()

        item_id = str(item.get("item_id", ""))
        metadata = self._item_metadata.get(item_id, {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        if description is None:
            description = self._first_text(
                metadata,
                ("description", "title", "name", "url_name", "semantic_category"),
            )
        if description is None:
            description = self._first_text(item, ("semantic_category", "category"))
        return description or "fashion item"

    @staticmethod
    def _aligned_value(
        row: Mapping[str, Any],
        keys: Sequence[str],
        position: int,
    ) -> Any:
        for key in keys:
            values = row.get(key)
            if (
                isinstance(values, Sequence)
                and not isinstance(values, (str, bytes))
                and position < len(values)
            ):
                return values[position]
        return None

    @staticmethod
    def _scalar_value(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
        for key in keys:
            value = row.get(key)
            if value is None:
                continue
            if isinstance(value, (str, bytes)):
                return value
            if isinstance(value, Mapping) or isinstance(value, Image.Image):
                return value
            if not isinstance(value, Sequence):
                return value
        return None

    @staticmethod
    def _first_text(
        values: Mapping[str, Any],
        keys: Sequence[str],
    ) -> str | None:
        for key in keys:
            value = values.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _open_image(value: Any) -> Image.Image:
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, Mapping):
            raw_bytes = value.get("bytes")
            if isinstance(raw_bytes, bytes):
                with Image.open(BytesIO(raw_bytes)) as image:
                    return image.convert("RGB")
            value = value.get("path")
        if isinstance(value, (str, Path)):
            with Image.open(value) as image:
                return image.convert("RGB")
        raise TypeError(f"unsupported Polyvore image value: {type(value).__name__}")

    @staticmethod
    def _load_item_metadata(
        path: str | Path | None,
    ) -> Mapping[str, Mapping[str, Any]]:
        if path is None:
            return {}
        return _read_item_metadata(str(Path(path).resolve()))


def collate_compatibility(
    examples: Sequence[CompatibilityExample],
) -> CompatibilityBatch:
    if not examples:
        raise ValueError("cannot collate an empty compatibility batch")
    outfits = collate_outfits(
        [
            OutfitExample(
                outfit_id=example.outfit_id,
                images=example.images,
                descriptions=example.descriptions,
            )
            for example in examples
        ]
    )
    labels = torch.tensor(
        [example.label for example in examples],
        dtype=torch.float32,
    )
    return CompatibilityBatch(outfits=outfits, labels=labels)


def load_polyvore_compatibility_dataset(
    variant: PolyvoreVariant = "nondisjoint",
    split: PolyvoreSplit = "train",
    *,
    token: bool | str | None = True,
    cache_dir: str | Path | None = None,
    image_transform: ImageTransform | None = None,
) -> PolyvoreCompatibilityDataset:
    """Load a gated Hugging Face split and its official CP question file."""
    if variant not in ("nondisjoint", "disjoint"):
        raise ValueError("variant must be 'nondisjoint' or 'disjoint'")
    if split not in ("train", "validation", "test"):
        raise ValueError("split must be 'train', 'validation', or 'test'")

    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError(
            "Hugging Face loading requires the 'datasets' and "
            "'huggingface_hub' packages"
        ) from error

    normalized_cache_dir = str(cache_dir) if cache_dir is not None else None
    outfit_rows = load_dataset(
        _DATASET_ID,
        variant,
        split=split,
        token=token,
        cache_dir=normalized_cache_dir,
    )
    question_filename = (
        f"{variant}/compatibility_{_TEXT_SPLIT_NAMES[split]}.txt"
    )
    outfit_mapping_filename = f"{variant}/{_TEXT_SPLIT_NAMES[split]}.json"
    compatibility_path = hf_hub_download(
        repo_id=_DATASET_ID,
        filename=question_filename,
        repo_type="dataset",
        token=token,
        cache_dir=normalized_cache_dir,
    )
    metadata_path = hf_hub_download(
        repo_id=_DATASET_ID,
        filename="polyvore_item_metadata.json",
        repo_type="dataset",
        token=token,
        cache_dir=normalized_cache_dir,
    )
    outfit_mapping_path = hf_hub_download(
        repo_id=_DATASET_ID,
        filename=outfit_mapping_filename,
        repo_type="dataset",
        token=token,
        cache_dir=normalized_cache_dir,
    )
    return PolyvoreCompatibilityDataset(
        outfit_rows=outfit_rows,
        compatibility_path=compatibility_path,
        image_transform=image_transform,
        item_metadata_path=metadata_path,
        outfit_mapping_path=outfit_mapping_path,
    )


@lru_cache(maxsize=4)
def _read_item_metadata(path: str) -> Mapping[str, Mapping[str, Any]]:
    with Path(path).open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if not isinstance(metadata, Mapping):
        raise ValueError("Polyvore item metadata must be a JSON object")
    return metadata
