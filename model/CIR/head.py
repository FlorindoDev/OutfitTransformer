"""Projection head for Complementary Item Retrieval."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..common.config import DEFAULT_MODEL_CONFIG
from .config import DEFAULT_CIR_CONFIG


class RetrievalEmbeddingHead(nn.Module):
    """Project a contextual state into the shared retrieval space."""

    def __init__(
        self,
        input_dim: int = DEFAULT_MODEL_CONFIG.transformer.model_dim,
        output_dim: int = DEFAULT_CIR_CONFIG.embedding_dim,
        *,
        normalize: bool = DEFAULT_CIR_CONFIG.normalize_embeddings,
        normalization_epsilon: float = (
            DEFAULT_MODEL_CONFIG.transformer.normalization_epsilon
        ),
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if not isinstance(normalize, bool):
            raise TypeError("normalize must be boolean")
        if normalization_epsilon <= 0.0:
            raise ValueError("normalization_epsilon must be positive")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize = normalize
        self.normalization_epsilon = normalization_epsilon
        self.projection = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, representations: Tensor) -> Tensor:
        """Return one retrieval embedding for each input representation."""
        if representations.ndim != 2:
            raise ValueError("representations must have shape [batch, features]")
        if representations.size(0) == 0:
            raise ValueError("representations cannot contain an empty batch")
        if representations.size(1) != self.input_dim:
            raise ValueError(
                f"representations must have {self.input_dim} features"
            )
        if not torch.is_floating_point(representations):
            raise TypeError("representations must be a floating-point tensor")
        if not bool(torch.isfinite(representations).all()):
            raise ValueError("representations must contain only finite values")

        embeddings = self.projection(representations)
        if self.normalize:
            embeddings = F.normalize(
                embeddings,
                p=2,
                dim=-1,
                eps=self.normalization_epsilon,
            )
        return embeddings
