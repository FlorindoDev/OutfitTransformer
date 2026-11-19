"""Inference loops and report models for CIR evaluation."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

from data import DataSplit, ItemBatch, RetrievalBatch
from metrics import (
    RetrievalMetrics,
    mean_reciprocal_rank,
    recall_at_k,
    retrieval_metrics,
    retrieval_rank,
)
from training.CIR.data import (
    RetrievalEmbeddingBatch,
    as_single_item_outfits,
    flatten_retrieval_candidates,
)
from training.common.features import FeatureMode

from .catalog_data import CatalogLoaders, CatalogQueryBatch
from .config import DEFAULT_ERROR_NEIGHBORS

LOGGER = logging.getLogger("evaluation.CIR")
_FITB_CUTOFFS = {"fitb_accuracy": 1, "recall_at_2": 2}
_CATALOG_CUTOFFS = {f"recall_at_{k}": k for k in (1, 2, 10, 30, 50, 100)}


class BatchLoader(Protocol):
    """Minimal loader contract needed by the evaluation loop."""

    def __iter__(self) -> Iterator[Any]: ...

    def __len__(self) -> int: ...


class _EvaluationModel(Protocol):
    """Model operations used by both CIR evaluation protocols."""

    def to(self, device: torch.device, /) -> object: ...

    def eval(self) -> object: ...

    def embed_queries(self, outfits: Any, target_categories: tuple[str, ...], /) -> Tensor: ...

    def embed_items(self, outfits: Any, /) -> Tensor: ...


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
    errors: list[dict[str, Any]] | None = None

    @property
    def protocol(self) -> str:
        if isinstance(self.metrics, CatalogRetrievalMetrics):
            return "polyvore_full_catalog"
        return "polyvore_fitb"

    def as_dict(self) -> dict[str, Any]:
        report: dict[str, Any] = {
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
        if self.errors is not None:
            report["errors"] = self.errors
        return report


@torch.inference_mode()
def evaluate(
    model: _EvaluationModel,
    loader: BatchLoader,
    device: torch.device,
    *,
    candidate_batch_size: int = 512,
    log_every: int = 10,
    errors: list[dict[str, Any]] | None = None,
    error_neighbors: int = DEFAULT_ERROR_NEIGHBORS,
) -> RetrievalMetrics:
    """Rank official completions and aggregate metrics over every query."""
    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    if log_every <= 0:
        raise ValueError("log_every must be positive")
    if error_neighbors < 0:
        raise ValueError("error_neighbors cannot be negative")
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

        groups = zip(queries.float(), candidates.split(candidate_counts), strict=True)
        for row, (query, group) in enumerate(groups):
            distances = torch.linalg.vector_norm(group - query.unsqueeze(0), dim=1)
            rank = retrieval_rank(distances, positive_index=0)
            ranks.append(rank)
            if errors is not None and rank > 1:
                errors.append(_fitb_error(batch, row, rank, group, distances, error_neighbors))
        if batch_index % log_every == 0 or batch_index == len(loader):
            LOGGER.info("batch=%d/%d", batch_index, len(loader))

    return retrieval_metrics(torch.tensor(ranks, dtype=torch.long))


def _fitb_error(
    batch: RetrievalBatch | RetrievalEmbeddingBatch,
    row: int,
    rank: int,
    candidates: Tensor,
    distances: Tensor,
    neighbor_count: int,
) -> dict[str, Any]:
    item_ids = (batch.positive_item_ids[row], *batch.negative_item_ids[row])
    # A missed query has a competitor at least as close as the correct item.
    prediction = 1 + int(distances[1:].argmin().item())
    predicted_item_id = item_ids[prediction]
    neighbors = []
    if neighbor_count:
        from_prediction = torch.linalg.vector_norm(candidates - candidates[prediction], dim=1)
        distance_by_id = dict(zip(item_ids, from_prediction.tolist(), strict=True))
        del distance_by_id[predicted_item_id]
        neighbors = sorted(distance_by_id.items(), key=itemgetter(1))[:neighbor_count]
    error = _query_error(
        batch, row, rank, _FITB_CUTOFFS,
        predicted_item_id=predicted_item_id,
        predicted_distance=float(distances[prediction].item()),
        neighbors=neighbors,
    )
    error["predicted_distractor_index"] = prediction - 1
    return error


@torch.inference_mode()
def evaluate_catalog(
    model: _EvaluationModel,
    loaders: CatalogLoaders,
    device: torch.device,
    *,
    log_every: int = 10,
    errors: list[dict[str, Any]] | None = None,
    error_neighbors: int = DEFAULT_ERROR_NEIGHBORS,
) -> CatalogRetrievalMetrics:
    """Encode each catalog item once and rank the relevant item for every query."""
    if log_every <= 0:
        raise ValueError("log_every must be positive")
    if error_neighbors < 0:
        raise ValueError("error_neighbors cannot be negative")
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
        batch_ranks, predictions = _catalog_ranking(
            queries.float(), catalog, positive_indices, collect_predictions=errors is not None,
        )
        batch_ranks = batch_ranks.cpu()
        ranks.append(batch_ranks)
        if errors is not None and predictions is not None:
            errors.extend(_catalog_errors(
                batch, batch_ranks, predictions, catalog, loaders.item_ids, error_neighbors,
            ))
        if batch_index % log_every == 0 or batch_index == len(loaders.queries):
            LOGGER.info("query_batch=%d/%d", batch_index, len(loaders.queries))

    all_ranks = torch.cat(ranks)
    return CatalogRetrievalMetrics(
        recall={k: recall_at_k(all_ranks, k) for k in _CATALOG_CUTOFFS.values()},
        mrr=mean_reciprocal_rank(all_ranks),
        examples=all_ranks.numel(),
        catalog_items=len(loaders.item_ids),
    )


def _catalog_ranking(
    queries: Tensor,
    catalog: Tensor,
    positive_indices: Tensor,
    *,
    collect_predictions: bool,
) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
    if queries.shape[1] != catalog.shape[1]:
        raise ValueError("query and candidate embedding dimensions must match")
    distances = torch.cdist(queries, catalog)
    if not bool(torch.isfinite(distances).all()):
        raise ValueError("catalog distances must contain only finite values")
    positive_distances = distances.gather(1, positive_indices)
    # The count includes the positive itself and puts tied competitors ahead of it.
    ranks = (distances <= positive_distances).sum(dim=1)
    if collect_predictions:
        # For misses, removing the positive also resolves its ties against it.
        distances.scatter_(1, positive_indices, torch.inf)
        return ranks, distances.min(dim=1)
    return ranks, None


def _catalog_errors(
    batch: CatalogQueryBatch,
    ranks: Tensor,
    predictions: tuple[Tensor, Tensor],
    catalog: Tensor,
    item_ids: tuple[str, ...],
    neighbor_count: int,
) -> list[dict[str, Any]]:
    ranks_by_row = ranks.tolist()
    failed_rows = [row for row, rank in enumerate(ranks_by_row) if rank > 1]
    if not failed_rows:
        return []

    predicted_distances, predicted_indices = predictions
    predicted_indices = predicted_indices[failed_rows]
    neighbor_count = min(neighbor_count, len(item_ids) - 1)
    neighbors: list[list[tuple[str, float]]] = [[] for _ in failed_rows]
    if neighbor_count:
        from_prediction = torch.cdist(catalog[predicted_indices], catalog)
        from_prediction.scatter_(1, predicted_indices.unsqueeze(1), torch.inf)
        neighbor_distances, neighbor_indices = from_prediction.topk(neighbor_count, largest=False)
        neighbors = [
            [(item_ids[index], distance) for index, distance in zip(row_indices, row_values, strict=True)]
            for row_indices, row_values in zip(
                neighbor_indices.tolist(), neighbor_distances.tolist(), strict=True,
            )
        ]

    return [
        _query_error(
            batch, row, ranks_by_row[row], _CATALOG_CUTOFFS,
            predicted_item_id=item_ids[prediction],
            predicted_distance=distance,
            neighbors=nearby,
        )
        for row, prediction, distance, nearby in zip(
            failed_rows, predicted_indices.tolist(), predicted_distances[failed_rows].tolist(),
            neighbors, strict=True,
        )
    ]


def _query_error(
    batch: CatalogQueryBatch | RetrievalBatch | RetrievalEmbeddingBatch,
    row: int,
    rank: int,
    cutoffs: Mapping[str, int],
    *,
    predicted_item_id: str,
    predicted_distance: float,
    neighbors: Iterable[tuple[str, float]],
) -> dict[str, Any]:
    return {
        "example_id": batch.example_ids[row],
        "partial_item_ids": batch.partial_item_ids[row],
        "target_category": batch.target_categories[row],
        "positive_item_id": batch.positive_item_ids[row],
        "positive_rank": rank,
        "predicted_item_id": predicted_item_id,
        "predicted_distance": predicted_distance,
        "failed_metrics": ["mrr", *(name for name, k in cutoffs.items() if rank > k)],
        "neighbors": [{"item_id": item_id, "distance": distance} for item_id, distance in neighbors],
    }


def _embed_candidates(
    model: _EvaluationModel,
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
