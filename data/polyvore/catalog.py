"""Polyvore catalog mapping item identifiers to model-neutral records."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from PIL import Image

from data.transforms import ImageTransform
from data.types import FashionItem
from preprocessing import load_image_from_bytes, load_image_from_path, normalize_image

from .rows import ItemRows


@dataclass(frozen=True, slots=True)
class _ItemMetadata:
    description: str
    category: str


_DEFAULT_METADATA = _ItemMetadata(description="fashion item", category="unknown")


class _ColumnarItemRows(Protocol):
    """Optional fast column access exposed by Hugging Face datasets."""

    column_names: Sequence[str]

    def __getitem__(self, column_name: str, /) -> Sequence[Any]: ...


class PolyvoreCatalog:
    """Resolve a Polyvore ``item_id`` to image, description and category."""

    def __init__(
        self,
        item_rows: ItemRows,
        metadata_path: str | Path,
        image_transform: ImageTransform,
    ) -> None:
        if not callable(image_transform):
            raise TypeError("image_transform must be callable")

        self._item_rows = item_rows
        self._image_transform = image_transform
        self._item_ids = _extract_item_ids(item_rows)
        if not self._item_ids:
            raise ValueError("item_rows cannot be empty")
        if len(set(self._item_ids)) != len(self._item_ids):
            raise ValueError("item_rows contains duplicate item_id values")
        self._row_by_item_id = {
            item_id: index for index, item_id in enumerate(self._item_ids)
        }
        self._metadata = _load_metadata(metadata_path, self._item_ids)

    def __len__(self) -> int:
        return len(self._item_ids)

    @property
    def item_ids(self) -> tuple[str, ...]:
        return self._item_ids

    def __contains__(self, item_id: object) -> bool:
        return str(item_id) in self._row_by_item_id

    def get(self, item_id: str) -> FashionItem:
        """Load and transform one item, failing clearly for unknown identifiers."""
        normalized_id = str(item_id)
        try:
            row_index = self._row_by_item_id[normalized_id]
        except KeyError as error:
            raise KeyError(f"unknown Polyvore item_id: {normalized_id}") from error

        row = self._item_rows[row_index]
        if not isinstance(row, Mapping):
            raise TypeError(f"row for item {normalized_id} must be a mapping")
        image = _decode_image(row.get("image"))
        image_tensor = self._image_transform(image)
        metadata = self._metadata.get(
            normalized_id,
            _DEFAULT_METADATA,
        )

        return FashionItem(
            item_id=normalized_id,
            image=image_tensor,
            description=metadata.description,
            category=metadata.category,
        )

    def get_many(self, item_ids: Sequence[str]) -> tuple[FashionItem, ...]:
        return tuple(self.get(item_id) for item_id in item_ids)


def load_outfit_token_index(path: str | Path) -> dict[str, str]:
    """Map official ``set_id_item_index`` tokens to catalog item identifiers."""
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError("Polyvore outfit mapping must contain a JSON list")

    token_index: dict[str, str] = {}
    for outfit_position, outfit in enumerate(payload):
        if not isinstance(outfit, Mapping):
            raise ValueError(f"outfit {outfit_position} must be an object")
        set_id = str(outfit.get("set_id", "")).strip()
        items = outfit.get("items")
        if not set_id or not isinstance(items, list) or not items:
            raise ValueError(
                f"outfit {outfit_position} requires set_id and non-empty items"
            )

        for item_position, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"item {item_position} in outfit {set_id} must be an object"
                )
            item_id = str(item.get("item_id", "")).strip()
            item_index = item.get("index")
            if not item_id or item_index is None:
                raise ValueError(
                    f"item {item_position} in outfit {set_id} misses item_id or index"
                )
            token = f"{set_id}_{item_index}"
            previous = token_index.setdefault(token, item_id)
            if previous != item_id:
                raise ValueError(f"outfit token maps to multiple items: {token}")

    return token_index


def load_item_categories(path: str | Path) -> dict[str, str]:
    """Load semantic categories without constructing image-backed items."""
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("Polyvore metadata must contain a JSON object")

    categories: dict[str, str] = {}
    for raw_item_id, value in payload.items():
        item_id = str(raw_item_id).strip()
        if item_id and isinstance(value, Mapping):
            categories[item_id] = _extract_category(value)
    return categories


def _extract_item_ids(item_rows: ItemRows) -> tuple[str, ...]:
    columns = getattr(item_rows, "column_names", None)
    if columns is not None:
        if "item_id" not in columns:
            raise ValueError("item_rows misses the item_id column")
        columnar_rows = cast(_ColumnarItemRows, item_rows)
        raw_ids = columnar_rows["item_id"]
    else:
        raw_ids = []
        for row_index, row in enumerate(item_rows):
            if not isinstance(row, Mapping) or "item_id" not in row:
                raise ValueError(f"item row {row_index} misses item_id")
            raw_ids.append(row["item_id"])

    item_ids = tuple(str(item_id).strip() for item_id in raw_ids)
    if any(not item_id for item_id in item_ids):
        raise ValueError("item_rows contains an empty item_id")
    return item_ids


def _load_metadata(
    path: str | Path,
    item_ids: Sequence[str],
) -> dict[str, _ItemMetadata]:
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("Polyvore metadata must contain a JSON object")

    metadata: dict[str, _ItemMetadata] = {}
    for item_id in item_ids:
        value = payload.get(item_id)
        if isinstance(value, Mapping):
            metadata[item_id] = _ItemMetadata(
                description=_build_description(value),
                category=_extract_category(value),
            )
    return metadata


def _read_json(path: str | Path) -> Any:
    selected_path = Path(path)
    if not selected_path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {selected_path}")
    try:
        with selected_path.open(encoding="utf-8") as source:
            return json.load(source)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON file: {selected_path}") from error


def _decode_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return normalize_image(value, "RGB")
    if isinstance(value, (bytes, bytearray)):
        return load_image_from_bytes(bytes(value), "RGB")
    if isinstance(value, (str, Path)):
        return load_image_from_path(value, "RGB")
    if isinstance(value, Mapping):
        image_bytes = value.get("bytes")
        if isinstance(image_bytes, (bytes, bytearray)):
            return load_image_from_bytes(bytes(image_bytes), "RGB")
        image_path = value.get("path")
        if image_path:
            return load_image_from_path(image_path, "RGB")
    raise TypeError("image must be a PIL image, bytes, path, or Hugging Face image")


def _build_description(metadata: Mapping[str, Any]) -> str:
    title = _first_text(metadata, "title", "url_name")
    details = _first_text(metadata, "description")
    if title and details and details.casefold() != title.casefold():
        return f"{title}. {details}"
    if title or details:
        return title or details

    category = _first_text(metadata, "semantic_category")
    return category or "fashion item"


def _extract_category(metadata: Mapping[str, Any]) -> str:
    return _first_text(metadata, "semantic_category", "category_id") or "unknown"


def _first_text(metadata: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return ""
