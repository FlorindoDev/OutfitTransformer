"""Feature-source profiles shared by task-specific training jobs."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from model.common.config import DEFAULT_MODEL_CONFIG, TransformerConfig


DEFAULT_PRECOMPUTED_EMBEDDING_ROOT = (
    Path("precomputed_embeddings")
    / DEFAULT_MODEL_CONFIG.encoders.fashion_clip_model_name.replace("/", "-")
)


class FeatureMode(str, Enum):
    """Source used to create multimodal item representations."""

    CLASSIC = "classic"
    NEW_CLASSIC = "new_classic"
    PRECOMPUTED = "precomputed"

    @property
    def uses_raw_inputs(self) -> bool:
        """Return whether images and descriptions are encoded during training."""
        return self in {
            FeatureMode.CLASSIC,
            FeatureMode.NEW_CLASSIC,
        }

    @property
    def uses_precomputed_embeddings(self) -> bool:
        """Return whether item embeddings are loaded from cache."""
        return self is FeatureMode.PRECOMPUTED

    @classmethod
    def from_serialized(cls, value: str) -> "FeatureMode":
        """Parse current and legacy checkpoint feature-mode names."""
        legacy_modes = {
            "fashion_clip_approach": cls.NEW_CLASSIC,
            "clip": cls.PRECOMPUTED,
            "openrouter": cls.PRECOMPUTED,
        }
        if value in legacy_modes:
            return legacy_modes[value]
        return cls(value)


def default_transformer_config(mode: FeatureMode) -> TransformerConfig:
    """Return architecture dimensions matching the selected feature source."""
    if not isinstance(mode, FeatureMode):
        raise TypeError("mode must be a FeatureMode")
    if mode is FeatureMode.CLASSIC:
        return DEFAULT_MODEL_CONFIG.classic_transformer
    if mode is FeatureMode.NEW_CLASSIC:
        return DEFAULT_MODEL_CONFIG.new_classic_transformer
    return DEFAULT_MODEL_CONFIG.transformer


def feature_config(mode: FeatureMode) -> dict[str, Any]:
    """Serialize the encoder profile without including authentication data."""
    encoders = DEFAULT_MODEL_CONFIG.encoders
    if mode.uses_raw_inputs:
        modality_embedding_dim = default_transformer_config(
            mode
        ).modality_embedding_dim
        return {
            "mode": mode.value,
            "visual_encoder": "ResNet18VisualEncoder",
            "visual_pretrained": encoders.resnet18_pretrained,
            "visual_trainable": encoders.resnet18_trainable,
            "visual_embedding_dim": modality_embedding_dim,
            "text_encoder": "SentenceTransformerTextEncoder",
            "text_model_name": encoders.sentence_transformer_model_name,
            "text_backbone_trainable": (
                encoders.sentence_transformer_trainable
            ),
            "text_projection_trainable": True,
            "text_embedding_dim": modality_embedding_dim,
        }
    return {
        "mode": mode.value,
        "encoder": "from_embedding_manifest",
        "precomputed": True,
        "trainable": False,
    }
