"""Read validated item embeddings produced by the precomputation job."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


class EmbeddingCache(Mapping[str, Tensor]):
    """Read-only, memory-mapped mapping from item IDs to embedding rows."""

    def __init__(
        self,
        directory: str | Path,
        *,
        expected_variant: str | None = None,
        expected_split: str | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.manifest = _load_manifest(self.directory / "manifest.json")
        _validate_manifest_identity(
            self.manifest,
            expected_variant=expected_variant,
            expected_split=expected_split,
        )
        self.embedding_dim = _positive_int(
            self.manifest.get("embedding_dim"),
            "manifest embedding_dim",
        )
        self.model_fingerprint = str(
            self.manifest.get("model_fingerprint", "")
        ).strip()
        self._embeddings: dict[str, Tensor] = {}
        self._shard_payloads: list[Mapping[str, Any]] = []
        self._load_shards()

    def __getitem__(self, item_id: str) -> Tensor:
        try:
            return self._embeddings[str(item_id)]
        except KeyError as error:
            raise KeyError(f"embedding not found for item_id {item_id!r}") from error

    def __iter__(self) -> Iterator[str]:
        return iter(self._embeddings)

    def __len__(self) -> int:
        return len(self._embeddings)

    def _load_shards(self) -> None:
        shard_entries = self.manifest.get("shards")
        if not isinstance(shard_entries, list) or not shard_entries:
            raise ValueError("manifest shards must be a non-empty list")

        for shard_index, entry in enumerate(shard_entries):
            if not isinstance(entry, Mapping):
                raise TypeError(f"manifest shard {shard_index} must be an object")
            filename = entry.get("file")
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError(f"manifest shard {shard_index} has invalid file")
            shard_path = self.directory / filename
            if not shard_path.is_file():
                raise FileNotFoundError(f"embedding shard does not exist: {shard_path}")
            payload = torch.load(
                shard_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            if not isinstance(payload, Mapping):
                raise TypeError(f"embedding shard must contain a mapping: {shard_path}")
            self._add_shard(payload, shard_path)
            self._shard_payloads.append(payload)

        expected_count = _positive_int(
            self.manifest.get("count"),
            "manifest count",
        )
        if len(self._embeddings) != expected_count:
            raise ValueError(
                "embedding count differs from manifest: "
                f"{len(self._embeddings)} != {expected_count}"
            )

    def _add_shard(self, payload: Mapping[str, Any], path: Path) -> None:
        item_ids = payload.get("item_ids")
        embeddings = payload.get("embeddings")
        if not isinstance(item_ids, (list, tuple)):
            raise TypeError(f"item_ids must be a list or tuple: {path}")
        if not isinstance(embeddings, Tensor) or embeddings.ndim != 2:
            raise TypeError(f"embeddings must be a rank-2 tensor: {path}")
        if embeddings.shape != (len(item_ids), self.embedding_dim):
            raise ValueError(f"item_ids and embeddings shape differ: {path}")
        if not torch.is_floating_point(embeddings):
            raise TypeError(f"embeddings must be floating point: {path}")
        if not bool(torch.isfinite(embeddings).all()):
            raise ValueError(f"embeddings must contain only finite values: {path}")

        for row, raw_item_id in enumerate(item_ids):
            item_id = str(raw_item_id).strip()
            if not item_id:
                raise ValueError(f"embedding shard contains an empty item_id: {path}")
            if item_id in self._embeddings:
                raise ValueError(f"duplicate embedding item_id: {item_id}")
            self._embeddings[item_id] = embeddings[row]


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"embedding manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid embedding manifest: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError("embedding manifest must contain an object")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported embedding manifest schema_version")
    return payload


def _validate_manifest_identity(
    manifest: Mapping[str, Any],
    *,
    expected_variant: str | None,
    expected_split: str | None,
) -> None:
    if expected_variant is not None and manifest.get("variant") != expected_variant:
        raise ValueError(
            f"embedding variant must be {expected_variant!r}, "
            f"got {manifest.get('variant')!r}"
        )
    if expected_split is not None and manifest.get("split") != expected_split:
        raise ValueError(
            f"embedding split must be {expected_split!r}, "
            f"got {manifest.get('split')!r}"
        )


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value

