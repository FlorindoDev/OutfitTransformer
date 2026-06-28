import torch
from torch import Tensor, nn


class OutfitContextEncoder(nn.Module):
    """Permutation-equivariant outfit encoder: no positional embeddings."""

    def __init__(
        self,
        embedding_dim: int = 128,
        layers: int = 6,
        attention_heads: int = 16,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )

    def forward(self, item_embeddings: Tensor, padding_mask: Tensor) -> Tensor:
        if item_embeddings.ndim != 3:
            raise ValueError("item_embeddings must have shape [batch, items, features]")
        if padding_mask.shape != item_embeddings.shape[:2]:
            raise ValueError("padding_mask must have shape [batch, items]")
        if padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be a boolean tensor")
        if padding_mask.all(dim=1).any():
            raise ValueError("each outfit must contain at least one non-padded item")

        contextual_embeddings = self.encoder(
            item_embeddings,
            src_key_padding_mask=padding_mask,
        )
        return contextual_embeddings.masked_fill(padding_mask.unsqueeze(-1), 0.0)
