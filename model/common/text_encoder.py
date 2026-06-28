from collections.abc import Sequence

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor, nn


class TextEncoder(nn.Module):
    """Frozen SentenceBERT backbone followed by the trainable FC from the paper."""

    def __init__(
        self,
        embedding_dim: int = 64,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        super().__init__()
        self.backbone = SentenceTransformer(model_name)
        backbone_dim = self.backbone.get_embedding_dimension()
        if backbone_dim is None:
            raise ValueError("SentenceBERT model does not expose its embedding dimension")

        self.projection = nn.Linear(backbone_dim, embedding_dim)
        self.backbone.requires_grad_(False)

    def train(self, mode: bool = True) -> "TextEncoder":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, descriptions: Sequence[str]) -> Tensor:
        if not descriptions:
            raise ValueError("descriptions cannot be empty")

        device = self.projection.weight.device
        with torch.no_grad():
            sentence_features = self.backbone.encode(
                list(descriptions),
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False,
            )
        return self.projection(sentence_features)
