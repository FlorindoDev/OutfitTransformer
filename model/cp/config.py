"""Configuration owned by Compatibility Prediction."""

from dataclasses import dataclass
from typing import Literal


Reduction = Literal["none", "mean", "sum"]


@dataclass(frozen=True)
class CompatibilityConfig:
    """Defaults specific to compatibility prediction."""

    focal_alpha: float = 0.5
    focal_gamma: float = 2.0
    focal_reduction: Reduction = "mean"

    def validate(self) -> None:
        if not 0.0 <= self.focal_alpha <= 1.0:
            raise ValueError("focal_alpha must be in [0, 1]")
        if self.focal_gamma < 0.0:
            raise ValueError("focal_gamma cannot be negative")
        if self.focal_reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                "focal_reduction must be 'none', 'mean', or 'sum'"
            )


DEFAULT_COMPATIBILITY_CONFIG = CompatibilityConfig()
DEFAULT_COMPATIBILITY_CONFIG.validate()
