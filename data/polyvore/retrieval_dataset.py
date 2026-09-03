"""Polyvore fill-in-the-blank dataset for complementary item retrieval."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from data.types import RetrievalExample, RetrievalIndexExample

from .catalog import PolyvoreCatalog, load_outfit_token_index


@dataclass(frozen=True)
class _RetrievalAnnotation:
    example_id: str
    query_item_ids: tuple[str, ...]
    positive_item_id: str
    negative_item_ids: tuple[str, ...]


class PolyvoreRetrievalDataset(Dataset[RetrievalExample]):
    """Return official FITB queries with positive and negative completions."""

    def __init__(
        self,
        catalog: PolyvoreCatalog,
        retrieval_path: str | Path,
        outfits_path: str | Path,
    ) -> None:
        self.catalog = catalog
        token_index = load_outfit_token_index(outfits_path)
        self._annotations = _load_annotations(
            retrieval_path,
            token_index,
        )
        _validate_catalog_coverage(self._annotations, catalog)

    def __len__(self) -> int:
        return len(self._annotations)

    def __getitem__(self, index: int) -> RetrievalExample:
        annotation = self._annotations[index]
        return RetrievalExample(
            example_id=annotation.example_id,
            partial_outfit=self.catalog.get_many(annotation.query_item_ids),
            positive_item=self.catalog.get(annotation.positive_item_id),
            negative_items=self.catalog.get_many(annotation.negative_item_ids),
        )


class PolyvoreRetrievalIndexDataset(Dataset[RetrievalIndexExample]):
    """Return official FITB examples without decoding catalog images."""

    def __init__(
        self,
        retrieval_path: str | Path,
        outfits_path: str | Path,
        category_by_item_id: Mapping[str, str] | None = None,
    ) -> None:
        token_index = load_outfit_token_index(outfits_path)
        self._annotations = _load_annotations(retrieval_path, token_index)
        categories = category_by_item_id or {}
        self._target_categories = tuple(
            categories.get(annotation.positive_item_id, "unknown")
            for annotation in self._annotations
        )

    def __len__(self) -> int:
        return len(self._annotations)

    def __getitem__(self, index: int) -> RetrievalIndexExample:
        annotation = self._annotations[index]
        return RetrievalIndexExample(
            example_id=annotation.example_id,
            partial_item_ids=annotation.query_item_ids,
            positive_item_id=annotation.positive_item_id,
            negative_item_ids=annotation.negative_item_ids,
            target_category=self._target_categories[index],
        )


def _load_annotations(
    path: str | Path,
    token_index: dict[str, str],
) -> tuple[_RetrievalAnnotation, ...]:
    selected_path = Path(path)
    if not selected_path.is_file():
        raise FileNotFoundError(
            f"retrieval annotation file does not exist: {selected_path}"
        )
    try:
        with selected_path.open(encoding="utf-8") as source:
            payload = json.load(source)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid retrieval JSON file: {selected_path}") from error
    if not isinstance(payload, list) or not payload:
        raise ValueError("retrieval annotations must contain a non-empty JSON list")

    annotations = [
        _parse_annotation(index, value, token_index)
        for index, value in enumerate(payload)
    ]
    return tuple(annotations)


def _parse_annotation(
    index: int,
    value: Any,
    token_index: dict[str, str],
) -> _RetrievalAnnotation:
    if not isinstance(value, Mapping):
        raise ValueError(f"retrieval example {index} must be an object")
    question = _string_list(value.get("question"), "question", index)
    answers = _string_list(value.get("answers"), "answers", index)
    if len(answers) < 2:
        raise ValueError(f"retrieval example {index} needs at least two answers")

    blank_position = value.get("blank_position")
    if not isinstance(blank_position, int) or blank_position <= 0:
        raise ValueError(
            f"retrieval example {index} requires a positive integer blank_position"
        )
    source_set = _token_set_id(question[0], index)
    if any(_token_set_id(token, index) != source_set for token in question):
        raise ValueError(f"retrieval example {index} mixes question outfit IDs")
    positive_token = f"{source_set}_{blank_position}"
    if answers.count(positive_token) != 1:
        raise ValueError(
            f"retrieval example {index} must contain answer {positive_token!r} once"
        )

    query_ids = tuple(
        _resolve_token(token, token_index, index) for token in question
    )
    positive_id = _resolve_token(positive_token, token_index, index)
    negative_ids = tuple(
        _resolve_token(token, token_index, index)
        for token in answers
        if token != positive_token
    )
    return _RetrievalAnnotation(
        example_id=f"retrieval:{index}",
        query_item_ids=query_ids,
        positive_item_id=positive_id,
        negative_item_ids=negative_ids,
    )


def _string_list(value: Any, name: str, index: int) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"retrieval example {index} requires non-empty {name}")
    strings = [str(element).strip() for element in value]
    if any(not element for element in strings):
        raise ValueError(f"retrieval example {index} contains an empty {name} token")
    return strings


def _token_set_id(token: str, index: int) -> str:
    parts = token.rsplit("_", maxsplit=1)
    if len(parts) != 2 or not parts[0] or not parts[1].isdigit():
        raise ValueError(f"invalid outfit token {token!r} in retrieval example {index}")
    return parts[0]


def _resolve_token(
    token: str,
    token_index: dict[str, str],
    index: int,
) -> str:
    try:
        item_id = token_index[token]
    except KeyError as error:
        raise ValueError(
            f"unknown outfit token {token!r} in retrieval example {index}"
        ) from error
    return item_id


def _validate_catalog_coverage(
    annotations: tuple[_RetrievalAnnotation, ...],
    catalog: PolyvoreCatalog,
) -> None:
    for annotation in annotations:
        item_ids = (
            *annotation.query_item_ids,
            annotation.positive_item_id,
            *annotation.negative_item_ids,
        )
        for item_id in item_ids:
            if item_id not in catalog:
                raise ValueError(
                    f"item {item_id!r} in {annotation.example_id} "
                    "is absent from the image split"
                )
