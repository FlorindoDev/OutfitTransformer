"""Classification head for compatibility prediction."""

from torch import Tensor, nn


class CompatibilityHead(nn.Module):
    """Map one global outfit representation to a compatibility probability."""

    def __init__(self, input_dim: int = 1024, dropout: float = 0.3) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_dim = input_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, outfit_embedding: Tensor) -> Tensor:
        """Return one compatibility probability for each outfit."""
        if outfit_embedding.ndim != 2:
            raise ValueError("outfit_embedding must have shape [batch, features]")
        if outfit_embedding.size(0) == 0:
            raise ValueError("outfit_embedding cannot contain an empty batch")
        if outfit_embedding.size(1) != self.input_dim:
            raise ValueError(
                f"outfit_embedding must have {self.input_dim} features"
            )
        return self.classifier(outfit_embedding)
