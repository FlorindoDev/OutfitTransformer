from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import OutfitEncoderConfig
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder
from .transformer_encoder import OutfitContextEncoder


@dataclass(frozen=True)
class OutfitEncoderOutput:
    item_embeddings: Tensor
    contextual_embeddings: Tensor
    outfit_embedding: Tensor
    padding_mask: Tensor


class OutfitEncoder(nn.Module):
    """Image + text item fusion followed by outfit-level self-attention."""

    def __init__(self, config: OutfitEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or OutfitEncoderConfig()
        self.config.validate()

        self.image_encoder = ImageEncoder(
            embedding_dim=self.config.image_embedding_dim,
            pretrained=self.config.pretrained_image_encoder,
        )
        self.text_encoder = TextEncoder(
            embedding_dim=self.config.text_embedding_dim,
            model_name=self.config.text_model_name,
        )
        self.outfit_token = nn.Parameter(
            torch.empty(1, 1, self.config.item_embedding_dim)
        )
        nn.init.normal_(self.outfit_token, std=0.02)
        self.context_encoder = OutfitContextEncoder(
            embedding_dim=self.config.item_embedding_dim,
            layers=self.config.transformer_layers,
            attention_heads=self.config.attention_heads,
            feedforward_dim=self.config.feedforward_dim,
            dropout=self.config.dropout,
        )

    def forward(
        self,
        images: Tensor,
        descriptions: Sequence[Sequence[str]],
        padding_mask: Tensor,
    ) -> OutfitEncoderOutput:
        item_embeddings = self.encode_items(images, descriptions, padding_mask)
        transformer_input, transformer_padding_mask = self._prepend_outfit_token(
            item_embeddings,
            padding_mask,
        )
        transformer_output = self.contextualize(
            transformer_input,
            transformer_padding_mask,
        )
        outfit_embedding = transformer_output[:, 0]
        contextual_embeddings = transformer_output[:, 1:]

        return OutfitEncoderOutput(
            item_embeddings=item_embeddings,
            contextual_embeddings=contextual_embeddings,
            outfit_embedding=outfit_embedding,
            padding_mask=padding_mask,
        )

    def encode_items(
        self,
        images: Tensor,
        descriptions: Sequence[Sequence[str]],
        padding_mask: Tensor,
    ) -> Tensor:
        """Encode padded outfit batches into multimodal item embeddings."""
        self._validate_inputs(images, descriptions, padding_mask)

        valid_mask = ~padding_mask
        flat_valid_mask = valid_mask.reshape(-1)
        valid_images = images.reshape(-1, *images.shape[2:])[flat_valid_mask]
        valid_descriptions = self._flatten_descriptions(descriptions)
        valid_item_embeddings = self.encode_flat_items(
            valid_images,
            valid_descriptions,
        )

        return self._restore_batch_shape(
            valid_item_embeddings,
            flat_valid_mask,
            batch_size=images.size(0),
            max_items=images.size(1),
        )

    def encode_flat_items(
        self,
        images: Tensor,
        descriptions: Sequence[str],
    ) -> Tensor:
        """Encode catalog items independently, without outfit context."""
        if images.ndim != 4:
            raise ValueError("images must have shape [items, channels, height, width]")
        if images.size(0) == 0:
            raise ValueError("at least one item is required")
        if len(descriptions) != images.size(0):
            raise ValueError("each image must have exactly one text description")

        image_embeddings = self.image_encoder(images)
        text_embeddings = self.encode_text(descriptions)
        return torch.cat((image_embeddings, text_embeddings), dim=-1)

    def encode_text(self, descriptions: Sequence[str]) -> Tensor:
        """Encode descriptions with frozen SentenceBERT and its trainable FC."""
        return self.text_encoder(descriptions)

    def contextualize(
        self,
        embeddings: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        """Run self-attention over item embeddings and optional task tokens."""
        return self.context_encoder(embeddings, padding_mask)

    def _prepend_outfit_token(
        self,
        item_embeddings: Tensor,
        padding_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size = item_embeddings.size(0)
        outfit_tokens = self.outfit_token.expand(batch_size, -1, -1)
        transformer_input = torch.cat((outfit_tokens, item_embeddings), dim=1)

        token_mask = padding_mask.new_zeros((batch_size, 1))
        transformer_padding_mask = torch.cat((token_mask, padding_mask), dim=1)
        return transformer_input, transformer_padding_mask

    @staticmethod
    def _flatten_descriptions(
        descriptions: Sequence[Sequence[str]],
    ) -> list[str]:
        return [
            description
            for outfit_descriptions in descriptions
            for description in outfit_descriptions
        ]

    @staticmethod
    def _restore_batch_shape(
        valid_embeddings: Tensor,
        flat_valid_mask: Tensor,
        batch_size: int,
        max_items: int,
    ) -> Tensor:
        flat_embeddings = valid_embeddings.new_zeros(
            (batch_size * max_items, valid_embeddings.size(-1))
        )
        valid_indices = flat_valid_mask.nonzero(as_tuple=False).squeeze(1)
        flat_embeddings = flat_embeddings.index_copy(
            0,
            valid_indices,
            valid_embeddings,
        )
        return flat_embeddings.view(batch_size, max_items, -1)

    @staticmethod
    def _validate_inputs(
        images: Tensor,
        descriptions: Sequence[Sequence[str]],
        padding_mask: Tensor,
    ) -> None:
        if images.ndim != 5:
            raise ValueError(
                "images must have shape [batch, items, channels, height, width]"
            )
        if padding_mask.shape != images.shape[:2]:
            raise ValueError("padding_mask must have shape [batch, items]")
        if padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be a boolean tensor")
        if len(descriptions) != images.size(0):
            raise ValueError("descriptions must contain one sequence per outfit")
        if padding_mask.all(dim=1).any():
            raise ValueError("each outfit must contain at least one item")
        if (
            padding_mask[:, :-1] & ~padding_mask[:, 1:]
        ).any():
            raise ValueError("padding positions must be trailing")

        for outfit_descriptions, outfit_mask in zip(
            descriptions,
            padding_mask,
            strict=True,
        ):
            required_descriptions = int((~outfit_mask).sum().item())
            if len(outfit_descriptions) != required_descriptions:
                raise ValueError(
                    "each valid image must have exactly one text description"
                )
