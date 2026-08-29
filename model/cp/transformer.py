"""Task-specific Transformer for compatibility prediction."""

import torch
from torch import Tensor, nn

from ..common.config import DEFAULT_MODEL_CONFIG, TransformerConfig
from ..common.output_validation import validate_common_output
from ..common.task_embedding import TaskEmbedding
from ..common.transformer import (
    OutfitTransformerOutput,
    build_transformer_encoder,
)

from .head import CompatibilityHead


class CompatibilityTransformer(nn.Module):
    """Predict compatibility from contextual embeddings produced by common."""

    def __init__(
        self,
        config: TransformerConfig | None = None,
        task_embedding: TaskEmbedding | None = None,
    ) -> None:
        super().__init__()
        self.config = config or DEFAULT_MODEL_CONFIG.transformer
        self.config.validate()

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
        self.predict_emb = _embedding_parameter(
            token_part_dim,
            initialization_std=self.config.embedding_initialization_std,
        )

        self.encoder = build_transformer_encoder(self.config)
        self.head = CompatibilityHead(
            input_dim=self.config.model_dim,
            dropout=self.config.dropout,
        )

    def forward(self, common_output: OutfitTransformerOutput) -> Tensor:
        """Return compatibility probabilities with shape ``[batch, 1]``."""
        outfit_embedding = self.encode(common_output)
        return self.head(outfit_embedding)

    def encode(self, common_output: OutfitTransformerOutput) -> Tensor:
        """Return CP token state after attending to every real outfit item."""
        item_embeddings, padding_mask = validate_common_output(
            common_output,
            expected_dim=self.config.model_dim,
        )
        batch_size = item_embeddings.size(0)
        cp_token = torch.cat((self.task_embedding(), self.predict_emb), dim=0)
        cp_tokens = cp_token.view(1, 1, -1).expand(batch_size, -1, -1)
        transformer_input = torch.cat((cp_tokens, item_embeddings), dim=1)

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

    @property
    def task_emb(self) -> nn.Parameter:
        """Expose the shared parameter using its conceptual name."""
        return self.task_embedding.embedding


def _embedding_parameter(
    size: int,
    *,
    initialization_std: float,
) -> nn.Parameter:
    embedding = nn.Parameter(torch.empty(size))
    nn.init.normal_(embedding, std=initialization_std)
    return embedding
