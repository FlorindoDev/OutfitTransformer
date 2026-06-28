from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..common.config import OutfitEncoderConfig
from ..common.heads import TaskMLP
from ..common.outfit_encoder import OutfitEncoder


@dataclass(frozen=True)
class CompatibilityOutput:
    logits: Tensor
    compatibility_score: Tensor
    outfit_embedding: Tensor


class CompatibilityPredictor(nn.Module):
    """Predict a compatibility score from the global outfit embedding."""

    def __init__(
        self,
        encoder: OutfitEncoder | None = None,
        config: OutfitEncoderConfig | None = None,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if encoder is not None and config is not None:
            raise ValueError("provide either an encoder or a config, not both")

        self.encoder = encoder or OutfitEncoder(config)
        embedding_dim = self.encoder.config.item_embedding_dim
        self.classifier = TaskMLP(
            input_dim=embedding_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        images: Tensor,
        descriptions: Sequence[Sequence[str]],
        padding_mask: Tensor,
    ) -> CompatibilityOutput:
        encoder_output = self.encoder(images, descriptions, padding_mask)
        logits = self.classifier(encoder_output.outfit_embedding).squeeze(-1)

        return CompatibilityOutput(
            logits=logits,
            compatibility_score=torch.sigmoid(logits),
            outfit_embedding=encoder_output.outfit_embedding,
        )
