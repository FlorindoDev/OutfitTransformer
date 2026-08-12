"""Configuration for Compatibility Prediction evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from data.polyvore import PolyvoreSplit


@dataclass(frozen=True)
class CPEvaluationConfig:
    """Validated runtime settings for one CP evaluation."""

    checkpoint: Path
    split: PolyvoreSplit = PolyvoreSplit.TEST
    embedding_root: Path | None = None
    output_path: Path | None = None
    cache_dir: Path | None = None
    batch_size: int = 512
    threshold: float = 0.5
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = False
    device: str = "auto"
    log_every: int = 10

    def validate(self) -> None:
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"evaluation checkpoint does not exist: {self.checkpoint}"
            )
        if self.split not in {PolyvoreSplit.VALIDATION, PolyvoreSplit.TEST}:
            raise ValueError("evaluation split must be validation or test")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if not self.device.strip():
            raise ValueError("device cannot be empty")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")

