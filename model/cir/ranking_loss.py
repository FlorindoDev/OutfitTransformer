import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..common.loss_reduction import Reduction, reduce_loss, validate_reduction


class SetWiseRankingLoss(nn.Module):
    """Paper set-wise loss over all negatives and the hardest negative."""

    def __init__(
        self,
        margin: float = 2.0,
        reduction: Reduction = "mean",
    ) -> None:
        super().__init__()
        if margin < 0.0:
            raise ValueError("margin must be non-negative")
        validate_reduction(reduction)
        self.margin = margin
        self.reduction: Reduction = reduction

    def forward(
        self,
        target_embeddings: Tensor,
        positive_embeddings: Tensor,
        negative_embeddings: Tensor,
    ) -> Tensor:
        self._validate_shapes(
            target_embeddings,
            positive_embeddings,
            negative_embeddings,
        )
        positive_distances = torch.linalg.vector_norm(
            target_embeddings - positive_embeddings,
            dim=-1,
        )
        negative_distances = torch.linalg.vector_norm(
            target_embeddings.unsqueeze(1) - negative_embeddings,
            dim=-1,
        )

        all_negative_loss = F.relu(
            positive_distances.unsqueeze(1)
            - negative_distances
            + self.margin
        ).mean(dim=1)
        hard_negative_distances = negative_distances.min(dim=1).values
        hard_negative_loss = F.relu(
            positive_distances - hard_negative_distances + self.margin
        )

        return reduce_loss(
            all_negative_loss + hard_negative_loss,
            self.reduction,
        )

    @staticmethod
    def _validate_shapes(
        target_embeddings: Tensor,
        positive_embeddings: Tensor,
        negative_embeddings: Tensor,
    ) -> None:
        if target_embeddings.ndim != 2:
            raise ValueError("target embeddings must have shape [batch, features]")
        if positive_embeddings.shape != target_embeddings.shape:
            raise ValueError(
                "positive embeddings must have shape [batch, features]"
            )
        if negative_embeddings.ndim != 3:
            raise ValueError(
                "negative embeddings must have shape [batch, negatives, features]"
            )
        if negative_embeddings.size(0) != target_embeddings.size(0):
            raise ValueError("negative embeddings must use the same batch size")
        if negative_embeddings.size(1) == 0:
            raise ValueError("at least one negative embedding is required")
        if negative_embeddings.size(2) != target_embeddings.size(1):
            raise ValueError("all embeddings must use the same feature dimension")
