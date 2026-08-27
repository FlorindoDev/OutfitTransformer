"""Shared multimodal components for CP and CIR tasks."""

from .config import (
    DEFAULT_MODEL_CONFIG,
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
    "CompatibilityConfig",
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
