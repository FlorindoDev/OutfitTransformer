"""Inference loop and report model for CIR FITB evaluation."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

from data import DataSplit
from metrics import RetrievalMetrics, retrieval_metrics, retrieval_rank
from training.CIR import CIRTrainingModel
from training.CIR.data import as_single_item_outfits, flatten_retrieval_candidates
from training.common.features import FeatureMode

LOGGER = logging.getLogger("evaluation.CIR")


class BatchLoader(Protocol):
    """Minimal loader contract needed by the evaluation loop."""

    def __iter__(self) -> Iterator[Any]: ...

    def __len__(self) -> int: ...


@dataclass(frozen=True)
class CIREvaluationResult:
    """Serializable summary of one checkpoint's FITB evaluation."""

    checkpoint: Path
    checkpoint_epoch: int
    output_path: Path
    split: DataSplit
    dataset_name: str
    dataset_id: str
    subset: str
    feature_mode: FeatureMode
    use_category_embedding: bool
    metrics: RetrievalMetrics

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint),
            "checkpoint_epoch": self.checkpoint_epoch,
            "dataset": self.dataset_name,
            "dataset_id": self.dataset_id,
            "split": self.split.value,
            "subset": self.subset,
            "feature_mode": self.feature_mode.value,
            "use_category_embedding": self.use_category_embedding,
            "protocol": "polyvore_fitb",
            "distance": "euclidean",
            "tie_policy": "pessimistic",
            "metrics": self.metrics.as_dict(),
        }


@torch.inference_mode()
def evaluate(
    model: CIRTrainingModel,
    loader: BatchLoader,
    device: torch.device,
    *,
    candidate_batch_size: int = 512,
    log_every: int = 10,
) -> RetrievalMetrics:
    """Rank official completions and aggregate metrics over every query."""
    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    if log_every <= 0:
        raise ValueError("log_every must be positive")
    if len(loader) == 0:
        raise ValueError("evaluation loader cannot be empty")

    model.to(device)
    model.eval()
    ranks: list[int] = []
    for batch_index, batch in enumerate(loader, start=1):
        candidate_items, candidate_counts = flatten_retrieval_candidates(batch)
        if not candidate_counts:
            raise ValueError("FITB evaluation batches cannot be empty")
        queries = model.embed_queries(batch.partial_outfits, batch.target_categories)
        _validate_embeddings(queries, len(candidate_counts))
        candidates = _embed_candidates(model, candidate_items, candidate_batch_size)
        if queries.shape[1] != candidates.shape[1]:
            raise ValueError("query and candidate embedding dimensions must match")

        for query, group in zip(
            queries.float(), candidates.split(candidate_counts), strict=True,
        ):
            distances = torch.linalg.vector_norm(group - query.unsqueeze(0), dim=1)
            ranks.append(retrieval_rank(distances, positive_index=0))
        if batch_index % log_every == 0 or batch_index == len(loader):
            LOGGER.info("batch=%d/%d", batch_index, len(loader))

    return retrieval_metrics(torch.tensor(ranks, dtype=torch.long))


def _embed_candidates(
    model: CIRTrainingModel,
    items: tuple[Any, ...],
    batch_size: int,
) -> Tensor:
    """Bound candidate inference memory independently of the query batch."""
    embeddings: list[Tensor] = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        values = model.embed_items(as_single_item_outfits(chunk))
        _validate_embeddings(values, len(chunk))
        embeddings.append(values.float())
    return torch.cat(embeddings)


def _validate_embeddings(embeddings: Tensor, expected_rows: int) -> None:
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_rows:
        raise ValueError("retrieval embeddings must have shape [examples, features]")
    if embeddings.shape[1] == 0 or not torch.is_floating_point(embeddings):
        raise ValueError("retrieval embeddings must have floating-point features")
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("retrieval embeddings must contain only finite values")
