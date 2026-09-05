"""Task-specific Transformer for Complementary Item Retrieval."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..common.config import (
    DEFAULT_CIR_CONFIG,
    DEFAULT_MODEL_CONFIG,
    ComplementaryItemConfig,
    TransformerConfig,
)
from ..common.embeddings import OutfitEmbeddingBatch
from ..common.output_validation import validate_outfit_batch
from ..common.task_embedding import TaskEmbedding
from ..common.transformer_encoder import build_transformer_encoder
from .category_embedding import PolyvoreCategoryEmbedding
from .head import RetrievalEmbeddingHead


class ComplementaryItemTransformer(nn.Module):
    """Embed partial outfits and target items in one retrieval space."""

    def __init__(
        self,
        config: TransformerConfig | None = None,
        cir_config: ComplementaryItemConfig | None = None,
        task_embedding: TaskEmbedding | None = None,
        use_category_embedding: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(use_category_embedding, bool):
            raise TypeError("use_category_embedding must be boolean")

        self.config = config or DEFAULT_MODEL_CONFIG.transformer
        self.cir_config = cir_config or DEFAULT_CIR_CONFIG
        self.use_category_embedding = use_category_embedding
        self.config.validate()
        self.cir_config.validate()

        token_part_dim = self.config.modality_embedding_dim
        selected_task_embedding = (
            task_embedding
            if task_embedding is not None
            else TaskEmbedding(
                token_part_dim,
                initialization_std=self.config.embedding_initialization_std,
            )
        )
        if not isinstance(selected_task_embedding, TaskEmbedding):
            raise TypeError("task_embedding must be a TaskEmbedding")
        if selected_task_embedding.embedding_dim != token_part_dim:
            raise ValueError(
                "task_embedding dimension must equal modality_embedding_dim "
                f"({token_part_dim})"
            )

        self.task_embedding = selected_task_embedding
        self.embed_emb = _embedding_parameter(
            token_part_dim,
            initialization_std=self.config.embedding_initialization_std,
        )
        self.category_embedding = (
            PolyvoreCategoryEmbedding(
                token_part_dim,
                initialization_std=self.config.embedding_initialization_std,
            )
            if self.use_category_embedding
            else None
        )
        self.encoder = build_transformer_encoder(self.config)
        self.head = RetrievalEmbeddingHead(
            input_dim=self.config.model_dim,
            output_dim=self.cir_config.embedding_dim,
            normalize=self.cir_config.normalize_embeddings,
            normalization_epsilon=self.config.normalization_epsilon,
        )

    def forward(
        self,
        outfit_batch: OutfitEmbeddingBatch,
        target_categories: Sequence[str] | None = None,
    ) -> Tensor:
        """Return retrieval embeddings for a batch of partial outfits."""
        return self.embed_query(outfit_batch, target_categories)

    def embed_query(
        self,
        outfit_batch: OutfitEmbeddingBatch,
        target_categories: Sequence[str] | None = None,
    ) -> Tensor:
        """Return missing-item embeddings conditioned on partial outfits."""
        query_representations = self.encode_query(
            outfit_batch,
            target_categories,
        )
        return self.head(query_representations)

    def encode_query(
        self,
        outfit_batch: OutfitEmbeddingBatch,
        target_categories: Sequence[str] | None = None,
    ) -> Tensor:
        """Return missing-item token states after outfit self-attention."""
        item_embeddings, padding_mask = validate_outfit_batch(
            outfit_batch,
            expected_dim=self.config.model_dim,
        )
        batch_size = item_embeddings.size(0)
        missing_item_tokens = self._missing_item_tokens(
            batch_size,
            target_categories,
        )
        transformer_input = torch.cat(
            (missing_item_tokens, item_embeddings),
            dim=1,
        )

        token_mask = torch.zeros(
            (batch_size, 1),
            device=padding_mask.device,
            dtype=torch.bool,
        )
        transformer_mask = torch.cat((token_mask, padding_mask), dim=1)
        encoded = self.encoder(
            transformer_input,
            src_key_padding_mask=transformer_mask,
        )
        return encoded[:, 0, :]

    def _missing_item_tokens(
        self,
        batch_size: int,
        target_categories: Sequence[str] | None,
    ) -> Tensor:
        task_embeddings = self.task_embedding().view(1, -1).expand(
            batch_size,
            -1,
        )
        retrieval_embeddings = self.embed_emb.view(1, -1).expand(
            batch_size,
            -1,
        )

        if self.category_embedding is not None:
            if target_categories is None:
                raise ValueError(
                    "target_categories are required when "
                    "use_category_embedding is enabled"
                )
            category_embeddings = self.category_embedding(target_categories)
            if category_embeddings.size(0) != batch_size:
                raise ValueError(
                    "target_categories length must equal the outfit batch size"
                )
            retrieval_embeddings = retrieval_embeddings + category_embeddings

        tokens = torch.cat(
            (task_embeddings, retrieval_embeddings),
            dim=-1,
        )
        return F.normalize(
            tokens,
            p=2,
            dim=-1,
            eps=self.config.normalization_epsilon,
        ).unsqueeze(1)

    def embed_items(self, outfit_batch: OutfitEmbeddingBatch) -> Tensor:
        """Embed batches whose elements contain exactly one target item."""
        item_representations = self.encode_items(outfit_batch)
        return self.head(item_representations)

    def encode_items(self, outfit_batch: OutfitEmbeddingBatch) -> Tensor:
        """Return CIR states for independent catalog or positive items."""
        item_embeddings, padding_mask = validate_outfit_batch(
            outfit_batch,
            expected_dim=self.config.model_dim,
        )
        real_item_counts = (~padding_mask).sum(dim=1)
        if not bool((real_item_counts == 1).all()):
            raise ValueError(
                "item outfit_batch must contain exactly one real item "
                "per batch element"
            )

        encoded = self.encoder(
            item_embeddings,
            src_key_padding_mask=padding_mask,
        )
        return encoded[:, 0, :]

    @property
    def task_emb(self) -> nn.Parameter:
        """Expose the parameter shared through ``model.common``."""
        return self.task_embedding.embedding

    @property
    def category_emb(self) -> nn.Parameter | None:
        """Expose Polyvore category vectors when conditioning is enabled."""
        if self.category_embedding is None:
            return None
        return self.category_embedding.weight


def _embedding_parameter(
    size: int,
    *,
    initialization_std: float,
) -> nn.Parameter:
    embedding = nn.Parameter(torch.empty(size))
    nn.init.normal_(embedding, std=initialization_std)
    return embedding
