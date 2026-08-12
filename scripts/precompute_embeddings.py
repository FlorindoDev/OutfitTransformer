"""Precompute normalized FashionCLIP image/text embeddings for Polyvore items."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from data import LoaderConfig, build_fashion_clip_transform
from data.loaders import build_polyvore_item_loader
from data.polyvore import (
    DEFAULT_DATASET_ROOT,
    PolyvoreSplit,
    PolyvoreVariant,
)
from data.types import ItemBatch
from model import FashionCLIPTextEncoder, FashionCLIPVisualEncoder
from model.common import TextEncoder, VisualEncoder

OutputDType = Literal["float32", "float16"]

LOGGER = logging.getLogger("precompute_embeddings")
DEFAULT_MODEL_NAME = "patrickjohncyh/fashion-clip"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PrecomputeConfig:
    """Validated settings for one Polyvore embedding job."""

    variant: PolyvoreVariant
    split: PolyvoreSplit
    model_name: str
    output_dir: Path
    dataset_root: Path
    cache_dir: Path | None
    batch_size: int
    num_workers: int
    shard_size: int
    output_dtype: OutputDType
    device: str
    token: bool | str | None
    limit: int | None
    overwrite: bool
    log_every: int

    def validate(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.shard_size <= 0:
            raise ValueError("shard_size must be positive")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")

    @property
    def target_dir(self) -> Path:
        model_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", self.model_name).strip("-")
        return self.output_dir / model_slug / self.variant.value / self.split.value


class EmbeddingShardWriter:
    """Validate, buffer and atomically persist item embedding shards."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        shard_size: int,
        output_dtype: OutputDType,
        overwrite: bool = False,
    ) -> None:
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        if output_dtype not in {"float32", "float16"}:
            raise ValueError("output_dtype must be float32 or float16")

        self.output_dir = Path(output_dir)
        self.shard_size = shard_size
        self.output_dtype = output_dtype
        self._torch_dtype = getattr(torch, output_dtype)
        self._item_ids: list[str] = []
        self._embedding_chunks: list[Tensor] = []
        self._seen_item_ids: set[str] = set()
        self._shards: list[dict[str, Any]] = []
        self._embedding_dim: int | None = None
        self._buffered_count = 0
        self._written_count = 0
        self._finalized = False
        self._prepare_output_directory(overwrite)

    @property
    def count(self) -> int:
        return self._written_count + self._buffered_count

    def add(self, item_ids: Sequence[str], embeddings: Tensor) -> None:
        """Append one encoded batch, flushing complete shards as needed."""
        if self._finalized:
            raise RuntimeError("cannot add embeddings after finalize")
        normalized_ids = tuple(str(item_id).strip() for item_id in item_ids)
        self._validate_batch(normalized_ids, embeddings)

        duplicate_ids = self._seen_item_ids.intersection(normalized_ids)
        if duplicate_ids or len(set(normalized_ids)) != len(normalized_ids):
            duplicate_id = next(iter(duplicate_ids), "within current batch")
            raise ValueError(f"duplicate item_id: {duplicate_id}")
        self._seen_item_ids.update(normalized_ids)

        cpu_embeddings = embeddings.detach().to(
            device="cpu",
            dtype=self._torch_dtype,
        )
        offset = 0
        while offset < len(normalized_ids):
            capacity = self.shard_size - self._buffered_count
            take = min(capacity, len(normalized_ids) - offset)
            self._item_ids.extend(normalized_ids[offset : offset + take])
            self._embedding_chunks.append(
                cpu_embeddings[offset : offset + take].contiguous()
            )
            self._buffered_count += take
            offset += take
            if self._buffered_count == self.shard_size:
                self._flush()

    def finalize(self, metadata: Mapping[str, Any]) -> Path:
        """Flush remaining data and atomically write the cache manifest."""
        if self._finalized:
            raise RuntimeError("writer already finalized")
        if self.count == 0:
            raise ValueError("cannot finalize an empty embedding cache")
        if self._buffered_count:
            self._flush()

        manifest = dict(metadata)
        manifest.update(
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.now(UTC).isoformat(),
                "count": self._written_count,
                "embedding_dim": self._embedding_dim,
                "dtype": self.output_dtype,
                "shards": self._shards,
            }
        )
        manifest_path = self.output_dir / "manifest.json"
        temporary_path = self.output_dir / "manifest.json.tmp"
        try:
            temporary_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary_path.replace(manifest_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        self._finalized = True
        return manifest_path

    def _validate_batch(
        self,
        item_ids: Sequence[str],
        embeddings: Tensor,
    ) -> None:
        if not item_ids:
            raise ValueError("item_ids cannot be empty")
        if any(not item_id for item_id in item_ids):
            raise ValueError("item_ids cannot contain empty values")
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape [items, features]")
        if embeddings.size(0) != len(item_ids):
            raise ValueError("item_ids and embeddings must have equal length")
        if embeddings.size(1) == 0:
            raise ValueError("embeddings must contain features")
        if not torch.is_floating_point(embeddings):
            raise TypeError("embeddings must be floating point")
        if not bool(torch.isfinite(embeddings).all()):
            raise ValueError("embeddings must contain only finite values")
        if self._embedding_dim is None:
            self._embedding_dim = embeddings.size(1)
        elif embeddings.size(1) != self._embedding_dim:
            raise ValueError(
                f"embedding dimension changed from {self._embedding_dim} "
                f"to {embeddings.size(1)}"
            )

    def _flush(self) -> None:
        embeddings = torch.cat(self._embedding_chunks, dim=0)
        shard_index = len(self._shards)
        filename = f"shard-{shard_index:05d}.pt"
        shard_path = self.output_dir / filename
        temporary_path = self.output_dir / f"{filename}.tmp"
        start = self._written_count
        payload = {
            "schema_version": SCHEMA_VERSION,
            "item_ids": tuple(self._item_ids),
            "embeddings": embeddings,
        }
        try:
            torch.save(payload, temporary_path)
            temporary_path.replace(shard_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        self._written_count += self._buffered_count
        self._shards.append(
            {
                "file": filename,
                "count": self._buffered_count,
                "start": start,
                "end": self._written_count,
            }
        )
        self._item_ids.clear()
        self._embedding_chunks.clear()
        self._buffered_count = 0

    def _prepare_output_directory(self, overwrite: bool) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        existing = tuple(self.output_dir.iterdir())
        if existing and not overwrite:
            raise FileExistsError(
                f"output directory is not empty: {self.output_dir}; "
                "use --overwrite to replace an existing cache"
            )
        if not overwrite:
            return

        for path in existing:
            managed_file = (
                path.is_file()
                and (
                    path.name == "manifest.json"
                    or path.name == "manifest.json.tmp"
                    or bool(re.fullmatch(r"shard-\d{5}\.pt(?:\.tmp)?", path.name))
                )
            )
            if not managed_file:
                raise ValueError(
                    f"refusing to overwrite unmanaged output path: {path}"
                )
            path.unlink()


@torch.inference_mode()
def encode_item_batch(
    batch: ItemBatch,
    visual_encoder: VisualEncoder,
    text_encoder: TextEncoder,
    device: torch.device,
) -> Tensor:
    """Return ``[visual_L2, text_L2]`` embeddings for one item batch."""
    if not batch.model_items:
        raise ValueError("batch cannot be empty")
    images = [item.image for item in batch.model_items]
    descriptions = [item.text for item in batch.model_items]
    if any(image is None for image in images):
        raise ValueError("all batch items must contain images")
    if any(text is None or not text.strip() for text in descriptions):
        raise ValueError("all batch items must contain text")

    image_tensor = torch.stack(
        [image for image in images if image is not None]
    ).to(device=device, non_blocking=True)
    text_values = [text for text in descriptions if text is not None]
    visual_embeddings = visual_encoder(image_tensor)
    text_embeddings = text_encoder(text_values)
    _validate_encoder_pair(
        visual_embeddings,
        text_embeddings,
        expected_items=len(batch.model_items),
    )
    return torch.cat(
        (
            _l2_normalize(visual_embeddings, "visual embeddings"),
            _l2_normalize(text_embeddings, "text embeddings"),
        ),
        dim=-1,
    )


def run(config: PrecomputeConfig) -> Path:
    """Execute one end-to-end Polyvore precomputation job."""
    config.validate()
    device = resolve_device(config.device)
    LOGGER.info("device=%s", device)
    writer = EmbeddingShardWriter(
        config.target_dir,
        shard_size=config.shard_size,
        output_dtype=config.output_dtype,
        overwrite=config.overwrite,
    )

    image_transform = build_fashion_clip_transform(config.model_name)
    loader = build_polyvore_item_loader(
        image_transform=image_transform,
        variant=config.variant,
        split=config.split,
        config=LoaderConfig(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.num_workers > 0,
        ),
        shuffle=False,
        token=config.token,
        cache_dir=config.cache_dir,
        dataset_root=config.dataset_root,
    )

    visual_encoder = FashionCLIPVisualEncoder(
        config.model_name,
        trainable=False,
    ).to(device)
    text_encoder = FashionCLIPTextEncoder(
        config.model_name,
        trainable=False,
    ).to(device)
    visual_encoder.eval()
    text_encoder.eval()
    if visual_encoder.output_dim != text_encoder.output_dim:
        raise ValueError("FashionCLIP visual and text projection dimensions differ")

    for batch_index, batch in enumerate(loader, start=1):
        selected_batch = _apply_limit(batch, config.limit, writer.count)
        if selected_batch is None:
            break
        embeddings = encode_item_batch(
            selected_batch,
            visual_encoder,
            text_encoder,
            device,
        )
        writer.add(selected_batch.item_ids, embeddings)
        if batch_index % config.log_every == 0:
            LOGGER.info("encoded_items=%d", writer.count)
        if config.limit is not None and writer.count >= config.limit:
            break

    metadata = _build_manifest_metadata(
        config,
        visual_encoder,
        text_encoder,
    )
    manifest_path = writer.finalize(metadata)
    LOGGER.info("saved_items=%d manifest=%s", writer.count, manifest_path)
    return manifest_path


def resolve_device(value: str) -> torch.device:
    """Resolve ``auto`` or validate an explicit PyTorch device."""
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")
    if device.type == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        raise ValueError("MPS device requested but MPS is unavailable")
    return device


def parse_args(argv: Sequence[str] | None = None) -> PrecomputeConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute concatenated FashionCLIP visual/text embeddings for "
            "Polyvore items."
        )
    )
    parser.add_argument(
        "--variant",
        choices=[variant.value for variant in PolyvoreVariant],
        default=PolyvoreVariant.DISJOINT.value,
    )
    parser.add_argument(
        "--split",
        choices=[split.value for split in PolyvoreSplit],
        default=PolyvoreSplit.TRAIN.value,
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("precomputed_embeddings"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument(
        "--output-dtype",
        choices=["float32", "float16"],
        default="float32",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    authentication = parser.add_mutually_exclusive_group()
    authentication.add_argument("--token")
    authentication.add_argument("--no-token", action="store_true")
    arguments = parser.parse_args(argv)
    token: bool | str | None = False if arguments.no_token else arguments.token or True

    config = PrecomputeConfig(
        variant=PolyvoreVariant(arguments.variant),
        split=PolyvoreSplit(arguments.split),
        model_name=arguments.model_name,
        output_dir=arguments.output_dir,
        dataset_root=arguments.dataset_root,
        cache_dir=arguments.cache_dir,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        shard_size=arguments.shard_size,
        output_dtype=arguments.output_dtype,
        device=arguments.device,
        token=token,
        limit=arguments.limit,
        overwrite=arguments.overwrite,
        log_every=arguments.log_every,
    )
    config.validate()
    return config


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run(parse_args(argv))
    return 0


def _apply_limit(
    batch: ItemBatch,
    limit: int | None,
    completed: int,
) -> ItemBatch | None:
    if limit is None:
        return batch
    remaining = limit - completed
    if remaining <= 0:
        return None
    if remaining >= len(batch.item_ids):
        return batch
    return ItemBatch(
        item_ids=batch.item_ids[:remaining],
        categories=batch.categories[:remaining],
        model_items=batch.model_items[:remaining],
    )


def _validate_encoder_pair(
    visual_embeddings: Tensor,
    text_embeddings: Tensor,
    *,
    expected_items: int,
) -> None:
    if visual_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("CLIP encoders must return [items, features]")
    if visual_embeddings.size(0) != expected_items:
        raise ValueError("visual encoder returned the wrong item count")
    if text_embeddings.size(0) != expected_items:
        raise ValueError("text encoder returned the wrong item count")
    if visual_embeddings.size(1) != text_embeddings.size(1):
        raise ValueError("visual and text embedding dimensions must match")


def _l2_normalize(embeddings: Tensor, name: str) -> Tensor:
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError(f"{name} must contain only finite values")
    norms = torch.linalg.vector_norm(embeddings, dim=-1)
    if bool((norms <= 1e-12).any()):
        raise ValueError(f"{name} cannot contain zero vectors")
    return F.normalize(embeddings, p=2, dim=-1, eps=1e-12)


def _build_manifest_metadata(
    config: PrecomputeConfig,
    visual_encoder: FashionCLIPVisualEncoder,
    text_encoder: FashionCLIPTextEncoder,
) -> dict[str, Any]:
    encoder_metadata = {
        "model_name": config.model_name,
        "visual_encoder": type(visual_encoder).__name__,
        "text_encoder": type(text_encoder).__name__,
        "visual_commit": getattr(visual_encoder.backbone.config, "_commit_hash", None),
        "text_commit": getattr(text_encoder.backbone.config, "_commit_hash", None),
        "modality_dim": visual_encoder.output_dim,
        "normalization": "l2_per_modality",
        "aggregation": "concat_visual_then_text",
    }
    fingerprint_source = json.dumps(
        encoder_metadata,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "dataset": "mvasil/polyvore-outfits",
        "dataset_root": str(config.dataset_root),
        "variant": config.variant.value,
        "split": config.split.value,
        "limit": config.limit,
        "encoder": encoder_metadata,
        "model_fingerprint": hashlib.sha256(fingerprint_source).hexdigest(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
