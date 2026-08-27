"""Multimodal fusion and position-free outfit Transformer."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from pydantic import BaseModel, ConfigDict, model_validator
from torch import Tensor, nn
from torch.nn import functional as F

from .config import DEFAULT_MODEL_CONFIG, TransformerConfig
from .text_encoder import SentenceTransformerTextEncoder, TextEncoder
from .visual_encoder import ResNet18VisualEncoder, VisualEncoder


class OutfitItem(BaseModel):
    """One raw item or one precomputed multimodal item embedding."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Tensor | None = None
    text: str | None = None
    embedding: Tensor | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "OutfitItem":
        has_image = self.image is not None
        has_text = self.text is not None
        has_embedding = self.embedding is not None

        if has_image != has_text:
            raise ValueError("image and text must be provided together")
        if has_embedding == has_image:
            raise ValueError(
                "provide either image and text, or one precomputed embedding"
            )
        if self.image is not None and self.image.ndim != 3:
            raise ValueError("image must have shape [channels, height, width]")
        if self.text is not None and not self.text.strip():
            raise ValueError("text cannot be empty")
        if self.embedding is not None and self.embedding.ndim != 1:
            raise ValueError("embedding must be one-dimensional")
        return self


@dataclass(frozen=True)
class OutfitTransformerOutput:
    """Padded inputs and contextual item representations."""

    item_embeddings: Tensor
    contextual_embeddings: Tensor
    padding_mask: Tensor
    lengths: Tensor
    truncated: Tensor


class OutfitContextTransformer(nn.Module):
    """Pre-norm Transformer encoder without positional embeddings."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.padding_embedding = nn.Parameter(torch.empty(config.model_dim))
        nn.init.normal_(
            self.padding_embedding,
            std=config.embedding_initialization_std,
        )

        self.encoder = build_transformer_encoder(config)

    def forward(
        self,
        outfits: Sequence[Tensor],
    ) -> OutfitTransformerOutput:
        item_embeddings, padding_mask, lengths, truncated = self._pad(outfits)
        normalized_embeddings = _l2_normalize(
            item_embeddings,
            name="Transformer input",
            epsilon=self.config.normalization_epsilon,
        )
        contextual_embeddings = self.encoder(
            normalized_embeddings,
            src_key_padding_mask=padding_mask,
        )

        normalized_padding = _l2_normalize(
            self.padding_embedding,
            name="padding embedding",
            epsilon=self.config.normalization_epsilon,
        )
        contextual_embeddings = torch.where(
            padding_mask.unsqueeze(-1),
            normalized_padding.view(1, 1, -1),
            contextual_embeddings,
        )
        return OutfitTransformerOutput(
            item_embeddings=normalized_embeddings,
            contextual_embeddings=contextual_embeddings,
            padding_mask=padding_mask,
            lengths=lengths,
            truncated=truncated,
        )

    def _pad(
        self,
        outfits: Sequence[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if not outfits:
            raise ValueError("outfits cannot be empty")

        device = self.padding_embedding.device
        dtype = self.padding_embedding.dtype
        batch_size = len(outfits)
        padded = self.padding_embedding.view(1, 1, -1).expand(
            batch_size,
            self.config.max_items,
            -1,
        ).clone()
        padding_mask = torch.ones(
            (batch_size, self.config.max_items),
            device=device,
            dtype=torch.bool,
        )
        lengths: list[int] = []
        truncated: list[bool] = []

        for index, outfit in enumerate(outfits):
            if outfit.ndim != 2 or outfit.size(1) != self.config.model_dim:
                raise ValueError(
                    f"outfit {index} must have shape [items, {self.config.model_dim}]"
                )
            if outfit.size(0) == 0:
                raise ValueError(f"outfit {index} cannot be empty")

            kept_items = min(outfit.size(0), self.config.max_items)
            padded[index, :kept_items] = outfit[:kept_items].to(
                device=device,
                dtype=dtype,
            )
            padding_mask[index, :kept_items] = False
            lengths.append(kept_items)
            truncated.append(outfit.size(0) > self.config.max_items)

        return (
            padded,
            padding_mask,
            torch.tensor(lengths, device=device, dtype=torch.long),
            torch.tensor(truncated, device=device, dtype=torch.bool),
        )


class OutfitTransformer(nn.Module):
    """Encode image/text items, fuse them, then contextualize each outfit."""

    def __init__(
        self,
        visual_encoder: VisualEncoder | None = None,
        text_encoder: TextEncoder | None = None,
        config: TransformerConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or DEFAULT_MODEL_CONFIG.transformer
        self.config.validate()

        self.visual_encoder = visual_encoder or ResNet18VisualEncoder()
        self.text_encoder = text_encoder or SentenceTransformerTextEncoder(
            output_dim=self.config.modality_embedding_dim
        )
        self.visual_projection = _projection(
            self.visual_encoder.output_dim,
            self.config.modality_embedding_dim,
        )
        self.text_projection = _projection(
            self.text_encoder.output_dim,
            self.config.modality_embedding_dim,
        )
        self.context_transformer = OutfitContextTransformer(self.config)

    def forward(
        self,
        outfits: Sequence[Sequence[OutfitItem]],
    ) -> OutfitTransformerOutput:
        """Encode a batch of non-empty outfits."""
        item_embeddings = self.encode_items(outfits)
        return self.context_transformer(item_embeddings)

    def encode_items(
        self,
        outfits: Sequence[Sequence[OutfitItem]],
    ) -> list[Tensor]:
        """Return one unpadded multimodal embedding matrix per outfit."""
        flat_items, outfit_lengths = _flatten_outfits(outfits)
        device = self.context_transformer.padding_embedding.device
        dtype = self.context_transformer.padding_embedding.dtype
        raw_positions = [
            index for index, item in enumerate(flat_items) if item.embedding is None
        ]
        encoded_items: list[Tensor | None] = [None] * len(flat_items)

        if raw_positions:
            raw_items = [flat_items[index] for index in raw_positions]
            images = _stack_images(raw_items).to(device=device, dtype=dtype)
            descriptions = [item.text for item in raw_items]
            if any(description is None for description in descriptions):
                raise RuntimeError("validated raw items must contain text")

            visual_features = self.visual_projection(self.visual_encoder(images))
            text_features = self.text_projection(
                self.text_encoder(
                    [
                        description
                        for description in descriptions
                        if description is not None
                    ]
                )
            )
            _validate_encoder_output(
                visual_features,
                expected_items=len(raw_items),
                expected_dim=self.config.modality_embedding_dim,
                name="visual encoder",
            )
            _validate_encoder_output(
                text_features,
                expected_items=len(raw_items),
                expected_dim=self.config.modality_embedding_dim,
                name="text encoder",
            )

            visual_features = _l2_normalize(
                visual_features,
                name="visual embeddings",
                epsilon=self.config.normalization_epsilon,
            )
            text_features = _l2_normalize(
                text_features,
                name="text embeddings",
                epsilon=self.config.normalization_epsilon,
            )
            fused_features = torch.cat((visual_features, text_features), dim=-1)
            for position, embedding in zip(
                raw_positions,
                fused_features,
                strict=True,
            ):
                encoded_items[position] = embedding

        for position, item in enumerate(flat_items):
            if item.embedding is None:
                continue
            if item.embedding.numel() != self.config.model_dim:
                raise ValueError(
                    "precomputed embedding must contain "
                    f"{self.config.model_dim} features"
                )
            embedding = item.embedding.to(device=device, dtype=dtype)
            visual_embedding, text_embedding = embedding.split(
                self.config.modality_embedding_dim
            )
            encoded_items[position] = torch.cat(
                (
                    _l2_normalize(
                        visual_embedding,
                        name="precomputed visual embedding",
                        epsilon=self.config.normalization_epsilon,
                    ),
                    _l2_normalize(
                        text_embedding,
                        name="precomputed text embedding",
                        epsilon=self.config.normalization_epsilon,
                    ),
                )
            )

        if any(embedding is None for embedding in encoded_items):
            raise RuntimeError("failed to encode one or more outfit items")
        complete_embeddings = [
            embedding for embedding in encoded_items if embedding is not None
        ]
        return _restore_outfits(complete_embeddings, outfit_lengths)


def _projection(input_dim: int, output_dim: int) -> nn.Module:
    if input_dim <= 0:
        raise ValueError("encoder output_dim must be positive")
    if input_dim == output_dim:
        return nn.Identity()
    return nn.Linear(input_dim, output_dim)


def build_transformer_encoder(config: TransformerConfig) -> nn.TransformerEncoder:
    """Build one Transformer encoder from the shared validated config."""
    config.validate()
    layer = nn.TransformerEncoderLayer(
        d_model=config.model_dim,
        nhead=config.attention_heads,
        dim_feedforward=config.feedforward_dim,
        dropout=config.dropout,
        activation=_activation(config.activation),
        batch_first=True,
        norm_first=config.norm_first,
    )
    return nn.TransformerEncoder(
        encoder_layer=layer,
        num_layers=config.layers,
        norm=nn.LayerNorm(
            config.model_dim,
            eps=config.layer_norm_epsilon,
        ),
        enable_nested_tensor=False,
    )


def _activation(name: str) -> nn.Module:
    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "mish": nn.Mish,
    }
    try:
        return activations[name]()
    except KeyError as error:
        raise ValueError(f"unsupported activation: {name}") from error


def _flatten_outfits(
    outfits: Sequence[Sequence[OutfitItem]],
) -> tuple[list[OutfitItem], list[int]]:
    if not outfits:
        raise ValueError("outfits cannot be empty")

    flat_items: list[OutfitItem] = []
    lengths: list[int] = []
    for outfit_index, outfit in enumerate(outfits):
        if not outfit:
            raise ValueError(f"outfit {outfit_index} cannot be empty")
        if any(not isinstance(item, OutfitItem) for item in outfit):
            raise TypeError("outfits must contain OutfitItem objects")
        flat_items.extend(outfit)
        lengths.append(len(outfit))
    return flat_items, lengths


def _stack_images(items: Sequence[OutfitItem]) -> Tensor:
    images = [item.image for item in items]
    if any(image is None for image in images):
        raise RuntimeError("validated raw items must contain images")
    try:
        return torch.stack([image for image in images if image is not None])
    except RuntimeError as error:
        raise ValueError("all images in a batch must have the same shape") from error


def _restore_outfits(
    embeddings: Sequence[Tensor],
    outfit_lengths: Sequence[int],
) -> list[Tensor]:
    outfits: list[Tensor] = []
    offset = 0
    for length in outfit_lengths:
        outfits.append(torch.stack(list(embeddings[offset : offset + length])))
        offset += length
    return outfits


def _validate_encoder_output(
    embeddings: Tensor,
    *,
    expected_items: int,
    expected_dim: int,
    name: str,
) -> None:
    expected_shape = (expected_items, expected_dim)
    if embeddings.shape != expected_shape:
        raise ValueError(f"{name} must return shape {expected_shape}")


def _l2_normalize(embeddings: Tensor, *, name: str, epsilon: float) -> Tensor:
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError(f"{name} must contain only finite values")
    norms = torch.linalg.vector_norm(embeddings, dim=-1, keepdim=True)
    if bool((norms <= epsilon).any()):
        raise ValueError(f"{name} cannot contain zero vectors")
    return F.normalize(embeddings, p=2, dim=-1, eps=epsilon)
