"""Binary focal loss for compatibility prediction."""

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


Reduction = Literal["none", "mean", "sum"]


class FocalLoss(nn.Module):
    """Apply binary focal loss to probabilities produced by a sigmoid."""

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 2.0,
        reduction: Reduction = "mean",
    ) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if gamma < 0.0:
            raise ValueError("gamma must be non-negative")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, probabilities: Tensor, targets: Tensor) -> Tensor:
        """Return focal loss for equally shaped probabilities and binary targets."""
        if probabilities.shape != targets.shape:
            raise ValueError("probabilities and targets must have the same shape")
        if probabilities.numel() == 0:
            raise ValueError("probabilities and targets cannot be empty")
        if not torch.is_floating_point(probabilities):
            raise TypeError("probabilities must be a floating-point tensor")
        if not bool(torch.isfinite(probabilities).all()):
            raise ValueError("probabilities must contain only finite values")
        if bool(((probabilities < 0.0) | (probabilities > 1.0)).any()):
            raise ValueError("probabilities must be in [0, 1]")
        if not bool(torch.isfinite(targets).all()):
            raise ValueError("targets must contain only finite values")
        if not bool(((targets == 0) | (targets == 1)).all()):
            raise ValueError("targets must contain only 0 and 1")

        targets = targets.to(
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        binary_cross_entropy = F.binary_cross_entropy(
            probabilities,
            targets,
            reduction="none",
        )
        probability_of_target = (
            probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        )
        target_alpha = self.alpha * targets + (1.0 - self.alpha) * (
            1.0 - targets
        )
        loss = (
            target_alpha
            * (1.0 - probability_of_target).pow(self.gamma)
            * binary_cross_entropy
        )

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
