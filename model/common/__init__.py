"""Shared multimodal components for CP and CIR tasks."""

from .config import (
    DEFAULT_CIR_CONFIG,
    DEFAULT_COMPATIBILITY_CONFIG,
    DEFAULT_MODEL_CONFIG,
    ComplementaryItemConfig,
    CompatibilityConfig,
    EncoderConfig,
    ModelConfig,
    TransformerConfig,
)
from .task_embedding import TaskEmbedding
from .text_encoder import (
    FashionCLIPTextEncoder,
    OpenRouterTextEncoder,
    SentenceTransformerTextEncoder,
    TextEncoder,
)
from .transformer import (
    OutfitItem,
    OutfitTransformer,
    OutfitTransformerOutput,
)
from .visual_encoder import (
    FashionCLIPVisualEncoder,
    OpenRouterVisualEncoder,
    ResNet18VisualEncoder,
    VisualEncoder,
)

__all__ = [
    "ComplementaryItemConfig",
    "CompatibilityConfig",
    "DEFAULT_CIR_CONFIG",
    "DEFAULT_COMPATIBILITY_CONFIG",
    "DEFAULT_MODEL_CONFIG",
    "EncoderConfig",
    "FashionCLIPTextEncoder",
    "FashionCLIPVisualEncoder",
    "ModelConfig",
    "OpenRouterTextEncoder",
    "OpenRouterVisualEncoder",
    "OutfitItem",
    "OutfitTransformer",
    "OutfitTransformerOutput",
    "ResNet18VisualEncoder",
    "SentenceTransformerTextEncoder",
    "TaskEmbedding",
    "TextEncoder",
    "TransformerConfig",
    "VisualEncoder",
]
