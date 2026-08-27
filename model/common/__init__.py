"""Shared multimodal components for CP and CIR tasks."""

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
    TransformerConfig,
)
from .visual_encoder import (
    FashionCLIPVisualEncoder,
    OpenRouterVisualEncoder,
    ResNet18VisualEncoder,
    VisualEncoder,
)

__all__ = [
    "FashionCLIPTextEncoder",
    "FashionCLIPVisualEncoder",
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
