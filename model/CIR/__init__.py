"""Complementary Item Retrieval components."""

from ..common.config import DEFAULT_CIR_CONFIG, ComplementaryItemConfig
from .category_embedding import POLYVORE_CATEGORIES, PolyvoreCategoryEmbedding
from .head import RetrievalEmbeddingHead
from .in_batch_triplet_margin_loss import InBatchTripletMarginLoss
from .transformer import ComplementaryItemTransformer

__all__ = [
    "ComplementaryItemConfig",
    "ComplementaryItemTransformer",
    "DEFAULT_CIR_CONFIG",
    "InBatchTripletMarginLoss",
    "POLYVORE_CATEGORIES",
    "PolyvoreCategoryEmbedding",
    "RetrievalEmbeddingHead",
]
