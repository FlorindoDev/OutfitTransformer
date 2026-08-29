"""Task-specific Transformer for Complementary Item Retrieval."""

import torch
from torch import Tensor, nn

from ..common.config import DEFAULT_MODEL_CONFIG, TransformerConfig
from ..common.output_validation import validate_common_output
from ..common.task_embedding import TaskEmbedding
from ..common.transformer import (
    OutfitTransformerOutput,
    build_transformer_encoder,
)
from .config import DEFAULT_CIR_CONFIG, ComplementaryItemConfig
from .head import RetrievalEmbeddingHead


class ComplementaryItemTransformer(nn.Module):
    """Embed partial outfits and target items in one retrieval space."""

    def __init__(
        self,
        config: TransformerConfig | None = None,
        cir_config: ComplementaryItemConfig | None = None,
        task_embedding: TaskEmbedding | None = None,
    ) -> None:
        super().__init__()
        self.config = config or DEFAULT_MODEL_CONFIG.transformer
        self.cir_config = cir_config or DEFAULT_CIR_CONFIG
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
        self.encoder = build_transformer_encoder(self.config)
        self.head = RetrievalEmbeddingHead(
            input_dim=self.config.model_dim,
            output_dim=self.cir_config.embedding_dim,
            normalize=self.cir_config.normalize_embeddings,
            normalization_epsilon=self.config.normalization_epsilon,
        )

    def forward(self, common_output: OutfitTransformerOutput) -> Tensor:
        """Return retrieval embeddings for a batch of partial outfits."""
        return self.embed_query(common_output)

    def embed_query(self, common_output: OutfitTransformerOutput) -> Tensor:
        """Return missing-item embeddings conditioned on partial outfits."""
        query_representations = self.encode_query(common_output)
        return self.head(query_representations)

    def encode_query(self, common_output: OutfitTransformerOutput) -> Tensor:
        """Return missing-item token states after outfit self-attention."""
        item_embeddings, padding_mask = validate_common_output(
            common_output,
            expected_dim=self.config.model_dim,
        )
        batch_size = item_embeddings.size(0)
        missing_item_token = torch.cat(
            (self.task_embedding(), self.embed_emb),
            dim=0,
        )
        missing_item_tokens = missing_item_token.view(1, 1, -1).expand(
            batch_size,
            -1,
            -1,
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

    def embed_items(self, common_output: OutfitTransformerOutput) -> Tensor:
        """Embed batches whose elements contain exactly one target item."""
        item_representations = self.encode_items(common_output)
        return self.head(item_representations)

    def encode_items(self, common_output: OutfitTransformerOutput) -> Tensor:
        """Return CIR states for independent catalog or positive items."""
        item_embeddings, padding_mask = validate_common_output(
            common_output,
            expected_dim=self.config.model_dim,
        )
        real_item_counts = (~padding_mask).sum(dim=1)
        if not bool((real_item_counts == 1).all()):
            raise ValueError(
                "item common_output must contain exactly one real item "
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


def _embedding_parameter(
    size: int,
    *,
    initialization_std: float,
) -> nn.Parameter:
    embedding = nn.Parameter(torch.empty(size))
    nn.init.normal_(embedding, std=initialization_std)
    return embedding
