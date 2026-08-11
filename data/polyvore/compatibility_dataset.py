"""Polyvore compatibility prediction dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset

from data.types import CompatibilityExample

from .catalog import PolyvoreCatalog, load_outfit_token_index


@dataclass(frozen=True)
class _CompatibilityAnnotation:
    example_id: str
    label: int
    item_ids: tuple[str, ...]


class PolyvoreCompatibilityDataset(Dataset[CompatibilityExample]):
    """Return positive and synthetic-negative outfits from official annotations."""

    def __init__(
        self,
        catalog: PolyvoreCatalog,
        compatibility_path: str | Path,
        outfits_path: str | Path,
    ) -> None:
        self.catalog = catalog
        token_index = load_outfit_token_index(outfits_path)
        self._annotations = _load_annotations(
            compatibility_path,
            token_index,
            catalog,
        )

    def __len__(self) -> int:
        return len(self._annotations)

    def __getitem__(self, index: int) -> CompatibilityExample:
        annotation = self._annotations[index]
        return CompatibilityExample(
            example_id=annotation.example_id,
            outfit=self.catalog.get_many(annotation.item_ids),
            label=annotation.label,
        )


def _load_annotations(
    path: str | Path,
    token_index: dict[str, str],
    catalog: PolyvoreCatalog,
) -> tuple[_CompatibilityAnnotation, ...]:
    selected_path = Path(path)
    if not selected_path.is_file():
        raise FileNotFoundError(
            f"compatibility annotation file does not exist: {selected_path}"
        )

    annotations: list[_CompatibilityAnnotation] = []
    with selected_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            fields = line.split()
            if not fields:
                continue
            if fields[0] not in {"0", "1"}:
                raise ValueError(
                    f"compatibility line {line_number} must start with 0 or 1"
                )
            if len(fields) < 2:
                raise ValueError(
                    f"compatibility line {line_number} has no outfit items"
                )

            item_ids = tuple(
                _resolve_token(token, token_index, catalog, line_number)
                for token in fields[1:]
            )
            annotations.append(
                _CompatibilityAnnotation(
                    example_id=f"compatibility:{line_number}",
                    label=int(fields[0]),
                    item_ids=item_ids,
                )
            )

    if not annotations:
        raise ValueError("compatibility annotation file contains no examples")
    return tuple(annotations)


def _resolve_token(
    token: str,
    token_index: dict[str, str],
    catalog: PolyvoreCatalog,
    line_number: int,
) -> str:
    try:
        item_id = token_index[token]
    except KeyError as error:
        raise ValueError(
            f"unknown outfit token {token!r} on compatibility line {line_number}"
        ) from error
    if item_id not in catalog:
        raise ValueError(
            f"item {item_id!r} on compatibility line {line_number} "
            "is absent from the image split"
        )
    return item_id
