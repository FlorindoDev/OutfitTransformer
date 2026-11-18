"""Complementary Item Retrieval components."""

from ..common.config import DEFAULT_CIR_CONFIG, ComplementaryItemConfig
from .category_embedding import POLYVORE_CATEGORIES, PolyvoreCategoryEmbedding
from .head import RetrievalEmbeddingHead
from .in_batch_info_nce_loss import InBatchInfoNCELoss, JointInBatchRetrievalLoss
from .in_batch_triplet_margin_loss import InBatchTripletMarginLoss
from .progressive_in_batch_triplet_margin_loss import (
    ProgressiveInBatchTripletMarginLoss,
)
from .transformer import ComplementaryItemTransformer

__all__ = [
    "ComplementaryItemConfig",
    "ComplementaryItemTransformer",
    "DEFAULT_CIR_CONFIG",
    "InBatchInfoNCELoss",
    "InBatchTripletMarginLoss",
    "JointInBatchRetrievalLoss",
    "POLYVORE_CATEGORIES",
    "PolyvoreCategoryEmbedding",
    "ProgressiveInBatchTripletMarginLoss",
    "RetrievalEmbeddingHead",
]
