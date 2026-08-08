from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CPResumeExtension:
    """Describe an explicit epoch-budget extension from a checkpoint."""

    checkpoint_epoch: int
    additional_epochs: int
    phase_source_epoch: int = 0

    def __post_init__(self) -> None:
        if self.checkpoint_epoch < 0:
            raise ValueError("checkpoint_epoch must be non-negative")
        if self.additional_epochs <= 0:
            raise ValueError("additional_epochs must be positive")
        if not 0 <= self.phase_source_epoch <= self.checkpoint_epoch:
            raise ValueError(
                "phase_source_epoch must be between zero and checkpoint_epoch"
            )

    @property
    def final_epoch(self) -> int:
        return self.checkpoint_epoch + self.additional_epochs

    @property
    def total_phase_epochs(self) -> int:
        return self.final_epoch - self.phase_source_epoch


__all__ = ["CPResumeExtension"]
