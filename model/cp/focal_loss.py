import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..common.loss_reduction import Reduction, reduce_loss, validate_reduction


class BinaryFocalLoss(nn.Module):
    """Numerically stable focal loss for binary logits."""

    def __init__(
        self,
        alpha: float | None = 0.25,
        gamma: float = 2.0,
        reduction: Reduction = "mean",
    ) -> None:
        super().__init__()
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1] or None")
        if gamma < 0.0:
            raise ValueError("gamma must be non-negative")
        validate_reduction(reduction)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction: Reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if logits.shape != targets.shape:
            raise ValueError("logits and targets must have the same shape")
        if not torch.is_floating_point(targets):
            targets = targets.to(dtype=logits.dtype)
        if ((targets < 0.0) | (targets > 1.0)).any():
            raise ValueError("targets must be in [0, 1]")

        cross_entropy = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        probabilities = torch.sigmoid(logits)
        correct_class_probability = (
            probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        )
        focal_weight = (1.0 - correct_class_probability).pow(self.gamma)

        if self.alpha is not None:
            alpha_weight = (
                self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            )
            focal_weight = focal_weight * alpha_weight

        return reduce_loss(cross_entropy * focal_weight, self.reduction)
