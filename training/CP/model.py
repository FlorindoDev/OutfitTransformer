"""Composition of common and CP Transformers for both feature modes."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from model import (
    CompatibilityTransformer,
    OutfitTransformer,
    TextEncoder,
    TransformerConfig,
    VisualEncoder,
)
from model.common.transformer import OutfitContextTransformer

from .config import FeatureMode, default_transformer_config


class CPTrainingModel(nn.Module):
    """Compose CP with classic encoders or precomputed CLIP embeddings."""

    def __init__(
        self,
        config: TransformerConfig | None = None,
        *,
        feature_mode: FeatureMode = FeatureMode.CLIP,
        visual_encoder: VisualEncoder | None = None,
        text_encoder: TextEncoder | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(feature_mode, FeatureMode):
            raise TypeError("feature_mode must be a FeatureMode")
        self.config = config or default_transformer_config(feature_mode)
        self.config.validate()
        self.feature_mode = feature_mode
        if feature_mode is FeatureMode.CLASSIC:
            self.common: nn.Module = OutfitTransformer(
                visual_encoder=visual_encoder,
                text_encoder=text_encoder,
                config=self.config,
            )
        else:
            if visual_encoder is not None or text_encoder is not None:
                raise ValueError(
                    "CLIP mode cannot receive runtime visual or text encoders"
                )
            self.common = OutfitContextTransformer(self.config)
        self.cp = CompatibilityTransformer(self.config)

    def forward(self, outfits: Any) -> Tensor:
        """Return compatibility probabilities shaped ``[batch, 1]``."""
        return self.cp(self.common(outfits))
