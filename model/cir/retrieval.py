from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..common.config import OutfitEncoderConfig
from ..common.heads import TaskMLP
from ..common.outfit_encoder import OutfitEncoder


@dataclass(frozen=True)
class RetrievalOutput:
    target_embedding: Tensor
    target_token: Tensor
    item_embeddings: Tensor
    contextual_embeddings: Tensor
    padding_mask: Tensor


class ComplementaryItemRetriever(nn.Module):
    """Generate a contextual target embedding for complementary retrieval."""

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
        image_dim = self.encoder.config.image_embedding_dim
        self.empty_image_embedding = nn.Parameter(torch.empty(1, image_dim))
        self.target_projection = TaskMLP(
            input_dim=embedding_dim,
            output_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )
        nn.init.normal_(self.empty_image_embedding, std=0.02)

    def forward(
        self,
        images: Tensor,
        descriptions: Sequence[Sequence[str]],
        padding_mask: Tensor,
        target_descriptions: Sequence[str],
    ) -> RetrievalOutput:
        self._validate_target_descriptions(
            target_descriptions,
            batch_size=images.size(0),
        )
        item_embeddings = self.encoder.encode_items(
            images,
            descriptions,
            padding_mask,
        )
        target_token = self._build_target_token(target_descriptions)
        transformer_inputs = torch.cat(
            (target_token.unsqueeze(1), item_embeddings),
            dim=1,
        )
        transformer_mask = self._prepend_unmasked_token(padding_mask)
        contextual_embeddings = self.encoder.contextualize(
            transformer_inputs,
            transformer_mask,
        )
        target_embedding = self.target_projection(contextual_embeddings[:, 0])

        return RetrievalOutput(
            target_embedding=target_embedding,
            target_token=target_token,
            item_embeddings=item_embeddings,
            contextual_embeddings=contextual_embeddings,
            padding_mask=transformer_mask,
        )

    def encode_candidates(
        self,
        images: Tensor,
        descriptions: Sequence[str],
    ) -> Tensor:
        """Encode positive, negative, or catalog items for distance search."""
        return self.encoder.encode_flat_items(images, descriptions)

    def _build_target_token(
        self,
        target_descriptions: Sequence[str],
    ) -> Tensor:
        text_embeddings = self.encoder.encode_text(target_descriptions)
        empty_image_embeddings = self.empty_image_embedding.expand(
            len(target_descriptions),
            -1,
        )
        return torch.cat((empty_image_embeddings, text_embeddings), dim=-1)

    @staticmethod
    def _prepend_unmasked_token(padding_mask: Tensor) -> Tensor:
        token_mask = torch.zeros(
            (padding_mask.size(0), 1),
            dtype=torch.bool,
            device=padding_mask.device,
        )
        return torch.cat((token_mask, padding_mask), dim=1)

    @staticmethod
    def _validate_target_descriptions(
        target_descriptions: Sequence[str],
        batch_size: int,
    ) -> None:
        if len(target_descriptions) != batch_size:
            raise ValueError("each outfit must have one target description")
        if any(
            not isinstance(description, str) or not description.strip()
            for description in target_descriptions
        ):
            raise ValueError("target descriptions must be non-empty strings")
