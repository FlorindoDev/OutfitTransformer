"""Composition of common embeddings and CIR Transformer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import Tensor, nn

from model import (
    ComplementaryItemConfig,
    ComplementaryItemTransformer,
    MultimodalOutfitEncoder,
    OutfitEmbeddingBatcher,
    TextEncoder,
    TransformerConfig,
    VisualEncoder,
)
from training.common.features import FeatureMode, default_transformer_config


class CIRTrainingModel(nn.Module):
    """Compose CIR with runtime encoders or precomputed embeddings."""

    def __init__(
        self,
        config: TransformerConfig | None = None,
        cir_config: ComplementaryItemConfig | None = None,
        *,
        feature_mode: FeatureMode = FeatureMode.NEW_CLASSIC,
        use_category_embedding: bool = False,
        visual_encoder: VisualEncoder | None = None,
        text_encoder: TextEncoder | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(feature_mode, FeatureMode):
            raise TypeError("feature_mode must be a FeatureMode")
        self.config = config or default_transformer_config(feature_mode)
        self.config.validate()
        self.feature_mode = feature_mode
        self.use_category_embedding = use_category_embedding

        if feature_mode.uses_raw_inputs:
            self.common: nn.Module = MultimodalOutfitEncoder(
                visual_encoder=visual_encoder,
                text_encoder=text_encoder,
                config=self.config,
            )
        else:
            if visual_encoder is not None or text_encoder is not None:
                raise ValueError(
                    "precomputed mode cannot receive runtime encoders"
                )
            self.common = OutfitEmbeddingBatcher(self.config)
        self.cir = ComplementaryItemTransformer(
            self.config,
            cir_config,
            use_category_embedding=use_category_embedding,
        )

    def forward(
        self,
        partial_outfits: Any,
        positive_outfits: Any,
        target_categories: Sequence[str] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return aligned query and positive item retrieval embeddings."""
        return (
            self.embed_queries(partial_outfits, target_categories),
            self.embed_items(positive_outfits),
        )

    def embed_queries(
        self,
        partial_outfits: Any,
        target_categories: Sequence[str] | None = None,
    ) -> Tensor:
        """Embed partial outfits, optionally conditioned by target category."""
        selected_categories = (
            target_categories if self.use_category_embedding else None
        )
        return self.cir.embed_query(
            self.common(partial_outfits),
            selected_categories,
        )

    def embed_items(self, single_item_outfits: Any) -> Tensor:
        """Embed independent candidate items in the CIR retrieval space."""
        return self.cir.embed_items(self.common(single_item_outfits))
