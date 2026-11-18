"""Inference loops and report models for CIR evaluation."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

from data import DataSplit, ItemBatch
from metrics import (
    RetrievalMetrics,
    mean_reciprocal_rank,
    recall_at_k,
    retrieval_metrics,
    retrieval_rank,
)
from training.CIR import CIRTrainingModel
from training.CIR.data import as_single_item_outfits, flatten_retrieval_candidates
from training.common.features import FeatureMode

from .catalog_data import CatalogLoaders

LOGGER = logging.getLogger("evaluation.CIR")


class BatchLoader(Protocol):
    """Minimal loader contract needed by the evaluation loop."""

    def __iter__(self) -> Iterator[Any]: ...

    def __len__(self) -> int: ...


@dataclass(frozen=True)
class CatalogRetrievalMetrics:
    """Recall and mean reciprocal rank across the evaluated catalog."""

    recall: dict[int, float]
    mrr: float
    examples: int
    catalog_items: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "examples": self.examples,
            "catalog_items": self.catalog_items,
            "mrr": self.mrr,
            **{f"recall_at_{k}": value for k, value in self.recall.items()},
        }


@dataclass(frozen=True)
class CIREvaluationResult:
    """Serializable summary of one checkpoint's CIR evaluation."""

    checkpoint: Path
    checkpoint_epoch: int
    output_path: Path
    split: DataSplit
    dataset_name: str
    dataset_id: str
    subset: str
    feature_mode: FeatureMode
    use_category_embedding: bool
    metrics: RetrievalMetrics | CatalogRetrievalMetrics

    @property
    def protocol(self) -> str:
        if isinstance(self.metrics, CatalogRetrievalMetrics):
            return "polyvore_full_catalog"
        return "polyvore_fitb"

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
            "protocol": self.protocol,
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


@torch.inference_mode()
def evaluate_catalog(
    model: CIRTrainingModel,
    loaders: CatalogLoaders,
    device: torch.device,
    *,
    log_every: int = 10,
) -> CatalogRetrievalMetrics:
    """Encode each catalog item once and rank the relevant item for every query."""
    if log_every <= 0:
        raise ValueError("log_every must be positive")
    if not len(loaders.candidates) or not len(loaders.queries):
        raise ValueError("catalog evaluation requires candidates and queries")

    model.to(device)
    model.eval()
    candidate_embeddings: list[Tensor] = []
    for batch_index, batch in enumerate(loaders.candidates, start=1):
        outfits = (
            batch.as_single_item_outfits() if isinstance(batch, ItemBatch) else batch
        )
        values = model.embed_items(outfits)
        _validate_embeddings(values, len(outfits))
        candidate_embeddings.append(values.float())
        if batch_index % log_every == 0 or batch_index == len(loaders.candidates):
            LOGGER.info("catalog_batch=%d/%d", batch_index, len(loaders.candidates))
    catalog = torch.cat(candidate_embeddings)
    del candidate_embeddings
    index_by_id = {item_id: index for index, item_id in enumerate(loaders.item_ids)}

    ranks: list[Tensor] = []
    for batch_index, batch in enumerate(loaders.queries, start=1):
        outfits = tuple(
            outfit.model_items if isinstance(outfit, ItemBatch) else outfit
            for outfit in batch.outfits
        )
        queries = model.embed_queries(outfits, batch.target_categories)
        _validate_embeddings(queries, len(outfits))
        positive_indices = torch.tensor(
            [[index_by_id[item_id]] for item_id in batch.positive_item_ids],
            device=queries.device,
        )
        ranks.append(_catalog_ranks(queries.float(), catalog, positive_indices).cpu())
        if batch_index % log_every == 0 or batch_index == len(loaders.queries):
            LOGGER.info("query_batch=%d/%d", batch_index, len(loaders.queries))

    all_ranks = torch.cat(ranks)
    return CatalogRetrievalMetrics(
        recall={k: recall_at_k(all_ranks, k) for k in (1, 2, 10, 30, 50, 100)},
        mrr=mean_reciprocal_rank(all_ranks),
        examples=all_ranks.numel(),
        catalog_items=len(loaders.item_ids),
    )


def _catalog_ranks(queries: Tensor, catalog: Tensor, positive_indices: Tensor) -> Tensor:
    if queries.shape[1] != catalog.shape[1]:
        raise ValueError("query and candidate embedding dimensions must match")
    distances = torch.cdist(queries, catalog)
    if not bool(torch.isfinite(distances).all()):
        raise ValueError("catalog distances must contain only finite values")
    positive_distances = distances.gather(1, positive_indices)
    # The count includes the positive itself and puts tied competitors ahead of it.
    return (distances <= positive_distances).sum(dim=1)


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
