"""Configuration for Complementary Item Retrieval training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from data import DEFAULT_DATASET_NAME, get_dataset_source
from model.common.config import (
    DEFAULT_CIR_CONFIG,
    ComplementaryItemConfig,
    TransformerConfig,
)
from training.common.features import (
    DEFAULT_PRECOMPUTED_EMBEDDING_ROOT,
    FeatureMode,
    default_transformer_config,
    feature_config,
)

LossReduction = Literal["mean", "sum"]
BEST_METRIC = "val_fitb_accuracy"

MAX_GRAD_NORM = 1.0
ONE_CYCLE_PCT_START = 0.3
ONE_CYCLE_DIV_FACTOR = 25.0
ONE_CYCLE_FINAL_DIV_FACTOR = 10_000.0


@dataclass(frozen=True)
class CIRTrainingConfig:
    """Validated settings for one CIR training run."""

    dataset_name: str = DEFAULT_DATASET_NAME
    subset: str = field(
        default_factory=lambda: get_dataset_source(
            DEFAULT_DATASET_NAME
        ).descriptor.default_subset
    )
    feature_mode: FeatureMode = FeatureMode.NEW_CLASSIC
    embedding_root: Path = DEFAULT_PRECOMPUTED_EMBEDDING_ROOT
    dataset_root: Path = field(
        default_factory=lambda: get_dataset_source(
            DEFAULT_DATASET_NAME
        ).descriptor.default_root
    )
    checkpoint_dir: Path = Path("checkpoints/nondisjoint/cir_new_classic")
    cache_dir: Path | None = None
    epochs: int = 200
    batch_size: int = 64
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    triplet_margin: float = DEFAULT_CIR_CONFIG.triplet_margin
    loss_reduction: LossReduction = "mean"
    retrieval_embedding_dim: int = DEFAULT_CIR_CONFIG.embedding_dim
    normalize_embeddings: bool = DEFAULT_CIR_CONFIG.normalize_embeddings
    use_category_embedding: bool = False
    seed: int = 42
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    num_workers: int = 0
    pin_memory: bool = False
    mixed_precision: bool = False
    ddp: bool = False
    device: str = "auto"
    log_every: int = 10
    pretrained_cp: Path | None = None
    resume: Path | None = None
    model: TransformerConfig | None = None

    def validate(self) -> None:
        source = get_dataset_source(self.dataset_name)
        source.descriptor.validate_subset(self.subset)
        if not isinstance(self.feature_mode, FeatureMode):
            raise TypeError("feature_mode must be a FeatureMode")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size < 2:
            raise ValueError(
                "batch_size must be at least 2 for in-batch negative mining"
            )
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if self.triplet_margin < 0.0:
            raise ValueError("triplet_margin cannot be negative")
        if self.loss_reduction not in {"mean", "sum"}:
            raise ValueError("loss_reduction must be 'mean' or 'sum'")
        if self.retrieval_embedding_dim <= 0:
            raise ValueError("retrieval_embedding_dim must be positive")
        if not isinstance(self.normalize_embeddings, bool):
            raise TypeError("normalize_embeddings must be boolean")
        if not isinstance(self.use_category_embedding, bool):
            raise TypeError("use_category_embedding must be boolean")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience <= 0
        ):
            raise ValueError("early_stopping_patience must be positive")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta cannot be negative")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if not isinstance(self.pin_memory, bool):
            raise TypeError("pin_memory must be boolean")
        if not isinstance(self.mixed_precision, bool):
            raise TypeError("mixed_precision must be boolean")
        if not isinstance(self.ddp, bool):
            raise TypeError("ddp must be boolean")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        if self.pretrained_cp is not None and self.resume is not None:
            raise ValueError("pretrained_cp and resume are mutually exclusive")
        self.model_config.validate()
        self.cir_config.validate()

    @property
    def best_metric(self) -> str:
        return BEST_METRIC

    @property
    def model_config(self) -> TransformerConfig:
        return self.model or default_transformer_config(self.feature_mode)

    @property
    def cir_config(self) -> ComplementaryItemConfig:
        return ComplementaryItemConfig(
            embedding_dim=self.retrieval_embedding_dim,
            normalize_embeddings=self.normalize_embeddings,
            triplet_margin=self.triplet_margin,
            loss_reduction=self.loss_reduction,
        )

    @property
    def effective_batch_size_per_process(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    def as_dict(
        self,
        *,
        resolved_device: str | None = None,
        world_size: int = 1,
        distributed_backend: str | None = None,
    ) -> dict[str, Any]:
        """Return checkpoint-safe configuration without authentication data."""
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        source = get_dataset_source(self.dataset_name)
        return {
            "dataset": {
                "name": source.descriptor.name,
                "id": source.descriptor.dataset_id,
                "subset": self.subset,
                "feature_mode": self.feature_mode.value,
                "embedding_root": (
                    str(self.embedding_root)
                    if self.feature_mode.uses_precomputed_embeddings
                    else None
                ),
                "dataset_root": str(self.dataset_root),
                "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            },
            "features": feature_config(self.feature_mode),
            "model": {
                "transformer": asdict(self.model_config),
                "cir": asdict(self.cir_config),
                "use_category_embedding": self.use_category_embedding,
            },
            "training": {
                "epochs": self.epochs,
                "batch_size_per_process": self.batch_size,
                "gradient_accumulation_steps": (
                    self.gradient_accumulation_steps
                ),
                "effective_batch_size_per_process": (
                    self.effective_batch_size_per_process
                ),
                "effective_batch_size_global": (
                    self.effective_batch_size_per_process * world_size
                ),
                "optimizer": "AdamW",
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "scheduler": {
                    "name": "OneCycleLR",
                    "max_lr": self.learning_rate,
                    "step": "optimizer_step",
                    "pct_start": ONE_CYCLE_PCT_START,
                    "anneal_strategy": "cos",
                    "div_factor": ONE_CYCLE_DIV_FACTOR,
                    "final_div_factor": ONE_CYCLE_FINAL_DIV_FACTOR,
                },
                "max_grad_norm": MAX_GRAD_NORM,
                "loss": "InBatchTripletMarginLoss",
                "triplet_margin": self.triplet_margin,
                "loss_reduction": self.loss_reduction,
                "best_metric": BEST_METRIC,
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_min_delta": self.early_stopping_min_delta,
                "seed": self.seed,
                "pretrained_cp_weights": (
                    str(self.pretrained_cp) if self.pretrained_cp else None
                ),
                "resume_weights": str(self.resume) if self.resume else None,
            },
            "runtime": {
                "requested_device": self.device,
                "resolved_device": resolved_device,
                "num_workers": self.num_workers,
                "pin_memory": self.pin_memory,
                "mixed_precision": self.mixed_precision,
                "ddp": self.ddp,
                "world_size": world_size,
                "distributed_backend": distributed_backend,
                "log_every": self.log_every,
            },
        }
