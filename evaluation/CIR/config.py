"""Configuration for Complementary Item Retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from data import DataSplit


@dataclass(frozen=True)
class CIREvaluationConfig:
    """Validated runtime settings for one CIR FITB evaluation."""

    checkpoint: Path
    split: DataSplit = DataSplit.TEST
    embedding_root: Path | None = None
    dataset_root: Path | None = None
    output_path: Path | None = None
    cache_dir: Path | None = None
    batch_size: int = 512
    candidate_batch_size: int = 512
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
        if self.split not in {DataSplit.VALIDATION, DataSplit.TEST}:
            raise ValueError("evaluation split must be validation or test")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.candidate_batch_size <= 0:
            raise ValueError("candidate_batch_size must be positive")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if not self.device.strip():
            raise ValueError("device cannot be empty")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
