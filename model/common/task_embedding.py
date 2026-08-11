"""Trainable task embedding shared by task-specific modules."""

import torch
from torch import Tensor, nn


class TaskEmbedding(nn.Module):
    """Own one trainable vector that can be shared across CP and CIR."""

    def __init__(
        self,
        embedding_dim: int = 512,
        initialization_std: float = 0.02,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if initialization_std <= 0.0:
            raise ValueError("initialization_std must be positive")

        self.embedding_dim = embedding_dim
        self.embedding = nn.Parameter(torch.empty(embedding_dim))
        nn.init.normal_(self.embedding, std=initialization_std)

    def forward(self) -> Tensor:
        """Return the shared trainable vector."""
        return self.embedding
