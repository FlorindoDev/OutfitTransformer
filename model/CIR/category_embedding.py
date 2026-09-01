"""Trainable target-category embeddings for Polyvore CIR queries."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# Values extracted from ``polyvore_item_metadata.json`` semantic_category fields.
# Stable order keeps checkpoint rows reproducible across training and inference.
POLYVORE_CATEGORIES = (
    "accessories",
    "all-body",
    "bags",
    "bottoms",
    "hats",
    "jewellery",
    "outerwear",
    "scarves",
    "shoes",
    "sunglasses",
    "tops",
)


class PolyvoreCategoryEmbedding(nn.Module):
    """Map Polyvore semantic categories to trainable embedding vectors."""

    def __init__(self, embedding_dim: int, *, initialization_std: float) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if initialization_std <= 0.0:
            raise ValueError("initialization_std must be positive")

        self.embedding_dim = embedding_dim
        self._category_to_index = {
            category: index for index, category in enumerate(POLYVORE_CATEGORIES)
        }
        self.weight = nn.Parameter(
            torch.empty(len(POLYVORE_CATEGORIES), embedding_dim)
        )
        nn.init.normal_(self.weight, std=initialization_std)

    def forward(self, categories: Sequence[str]) -> Tensor:
        """Return one trainable vector for each target category."""
        indices = self.encode(categories)
        return F.embedding(indices, self.weight)

    def encode(self, categories: Sequence[str]) -> Tensor:
        """Convert category names to stable indices on the parameter device."""
        if isinstance(categories, (str, bytes)) or not isinstance(
            categories,
            Sequence,
        ):
            raise TypeError("categories must be a sequence of category names")
        if len(categories) == 0:
            raise ValueError("categories cannot be empty")

        normalized_categories = tuple(
            self._normalize_category(category) for category in categories
        )
        unsupported = sorted(
            {
                category
                for category in normalized_categories
                if category not in self._category_to_index
            }
        )
        if unsupported:
            supported = ", ".join(POLYVORE_CATEGORIES)
            raise ValueError(
                f"unsupported Polyvore categories: {unsupported}; "
                f"supported categories: {supported}"
            )

        return torch.tensor(
            [self._category_to_index[category] for category in normalized_categories],
            device=self.weight.device,
            dtype=torch.long,
        )

    @staticmethod
    def _normalize_category(category: str) -> str:
        if not isinstance(category, str):
            raise TypeError("each category must be a string")
        normalized = " ".join(category.split()).casefold()
        if not normalized:
            raise ValueError("categories cannot contain empty values")
        return normalized
