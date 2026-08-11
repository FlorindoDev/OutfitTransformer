"""Shared multimodal components for CP and CIR tasks."""

from .task_embedding import TaskEmbedding
from .text_encoder import (
    FashionCLIPTextEncoder,
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
    ResNet18VisualEncoder,
    VisualEncoder,
)

__all__ = [
    "FashionCLIPTextEncoder",
    "FashionCLIPVisualEncoder",
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
