"""Task-specific Transformer for compatibility prediction."""

import torch
from torch import Tensor, nn

from ..common.task_embedding import TaskEmbedding
from ..common.transformer import OutfitTransformerOutput, TransformerConfig

from .head import CompatibilityHead


class CompatibilityTransformer(nn.Module):
    """Predict compatibility from contextual embeddings produced by common."""

    def __init__(
        self,
        config: TransformerConfig | None = None,
        task_embedding: TaskEmbedding | None = None,
    ) -> None:
        super().__init__()
        self.config = config or TransformerConfig()
        self.config.validate()

        token_part_dim = self.config.modality_embedding_dim
        selected_task_embedding = (
            task_embedding
            if task_embedding is not None
            else TaskEmbedding(token_part_dim)
        )
        if not isinstance(selected_task_embedding, TaskEmbedding):
            raise TypeError("task_embedding must be a TaskEmbedding")
        if selected_task_embedding.embedding_dim != token_part_dim:
            raise ValueError(
                "task_embedding dimension must equal modality_embedding_dim "
                f"({token_part_dim})"
            )
        self.task_embedding = selected_task_embedding
        self.predict_emb = _embedding_parameter(token_part_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=self.config.model_dim,
            nhead=self.config.attention_heads,
            dim_feedforward=self.config.feedforward_dim,
            dropout=self.config.dropout,
            activation=nn.Mish(),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=self.config.layers,
            norm=nn.LayerNorm(self.config.model_dim),
            enable_nested_tensor=False,
        )
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
        item_embeddings, padding_mask = _validate_common_output(
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


def _embedding_parameter(size: int) -> nn.Parameter:
    embedding = nn.Parameter(torch.empty(size))
    nn.init.normal_(embedding, std=0.02)
    return embedding


def _validate_common_output(
    common_output: OutfitTransformerOutput,
    *,
    expected_dim: int,
) -> tuple[Tensor, Tensor]:
    if not isinstance(common_output, OutfitTransformerOutput):
        raise TypeError("common_output must be an OutfitTransformerOutput")

    item_embeddings = common_output.contextual_embeddings
    padding_mask = common_output.padding_mask
    if item_embeddings.ndim != 3 or item_embeddings.size(2) != expected_dim:
        raise ValueError(
            "common contextual_embeddings must have shape "
            f"[batch, items, {expected_dim}]"
        )
    if item_embeddings.size(0) == 0 or item_embeddings.size(1) == 0:
        raise ValueError("common contextual_embeddings cannot be empty")
    if padding_mask.shape != item_embeddings.shape[:2]:
        raise ValueError(
            "common padding_mask must match contextual_embeddings batch and items"
        )
    if padding_mask.dtype != torch.bool:
        raise TypeError("common padding_mask must be boolean")
    if padding_mask.device != item_embeddings.device:
        raise ValueError(
            "common padding_mask and contextual_embeddings must share a device"
        )
    if bool(padding_mask.all(dim=1).any()):
        raise ValueError("each outfit must contain at least one real item")
    if not bool(torch.isfinite(item_embeddings).all()):
        raise ValueError("common contextual_embeddings must contain finite values")
    return item_embeddings, padding_mask
