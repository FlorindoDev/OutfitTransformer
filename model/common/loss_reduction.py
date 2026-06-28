from typing import Literal

from torch import Tensor

Reduction = Literal["none", "mean", "sum"]


def validate_reduction(reduction: str) -> None:
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def reduce_loss(loss: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
