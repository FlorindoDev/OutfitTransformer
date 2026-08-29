"""Configuration owned by Complementary Item Retrieval."""

from dataclasses import dataclass
from typing import Literal


Reduction = Literal["none", "mean", "sum"]


@dataclass(frozen=True)
class ComplementaryItemConfig:
    """Defaults specific to Complementary Item Retrieval."""

    embedding_dim: int = 128
    normalize_embeddings: bool = False
    triplet_margin: float = 2.0
    loss_reduction: Reduction = "mean"

    def validate(self) -> None:
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not isinstance(self.normalize_embeddings, bool):
            raise TypeError("normalize_embeddings must be boolean")
        if self.triplet_margin < 0.0:
            raise ValueError("triplet_margin cannot be negative")
        if self.loss_reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                "loss_reduction must be 'none', 'mean', or 'sum'"
            )


DEFAULT_CIR_CONFIG = ComplementaryItemConfig()
DEFAULT_CIR_CONFIG.validate()
