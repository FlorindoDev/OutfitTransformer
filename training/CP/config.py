"""Configuration for Compatibility Prediction training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from data import DEFAULT_DATASET_NAME, get_dataset_source
from model import DEFAULT_MODEL_CONFIG, TransformerConfig
from model.cp.config import DEFAULT_COMPATIBILITY_CONFIG, Reduction

BestMetric = Literal["val_auc", "val_accuracy", "val_loss"]

MAX_GRAD_NORM = 1.0
ONE_CYCLE_PCT_START = 0.3
ONE_CYCLE_DIV_FACTOR = 25.0
ONE_CYCLE_FINAL_DIV_FACTOR = 10_000.0
DEFAULT_PRECOMPUTED_EMBEDDING_ROOT = (
    Path("precomputed_embeddings")
    / DEFAULT_MODEL_CONFIG.encoders.fashion_clip_model_name.replace("/", "-")
)


class FeatureMode(str, Enum):
    """Source used to create multimodal item representations."""

    CLASSIC = "classic"
    NEW_CLASSIC = "new_classic"
    PRECOMPUTED = "precomputed"

    @property
    def uses_raw_inputs(self) -> bool:
        """Return whether images and descriptions are encoded during training."""
        return self in {
            FeatureMode.CLASSIC,
            FeatureMode.NEW_CLASSIC,
        }

    @property
    def uses_precomputed_embeddings(self) -> bool:
        """Return whether item embeddings are loaded from cache."""
        return self is FeatureMode.PRECOMPUTED

    @classmethod
    def from_serialized(cls, value: str) -> "FeatureMode":
        """Parse current and legacy checkpoint feature-mode names."""
        legacy_modes = {
            "fashion_clip_approach": cls.NEW_CLASSIC,
            "clip": cls.PRECOMPUTED,
            "openrouter": cls.PRECOMPUTED,
        }
        if value in legacy_modes:
            return legacy_modes[value]
        return cls(value)


def default_transformer_config(mode: FeatureMode) -> TransformerConfig:
    """Return architecture dimensions matching the selected feature source."""
    if not isinstance(mode, FeatureMode):
        raise TypeError("mode must be a FeatureMode")
    if mode is FeatureMode.CLASSIC:
        return DEFAULT_MODEL_CONFIG.classic_transformer
    if mode is FeatureMode.NEW_CLASSIC:
        return DEFAULT_MODEL_CONFIG.new_classic_transformer
    return DEFAULT_MODEL_CONFIG.transformer


@dataclass(frozen=True)
class CPTrainingConfig:
    """Validated settings for one CP training run."""

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
    checkpoint_dir: Path = Path(
        "checkpoints/nondisjoint/cp_new_classic"
    )
    cache_dir: Path | None = None
    epochs: int = 200
    batch_size: int = 512
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    focal_alpha: float = DEFAULT_COMPATIBILITY_CONFIG.focal_alpha
    focal_gamma: float = DEFAULT_COMPATIBILITY_CONFIG.focal_gamma
    focal_reduction: Reduction = (
        DEFAULT_COMPATIBILITY_CONFIG.focal_reduction
    )
    seed: int = 42
    best_metric: BestMetric = "val_auc"
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    num_workers: int = 0
    pin_memory: bool = False
    device: str = "auto"
    log_every: int = 10
    resume: Path | None = None
    model: TransformerConfig | None = None

    def validate(self) -> None:
        source = get_dataset_source(self.dataset_name)
        source.descriptor.validate_subset(self.subset)
        if not isinstance(self.feature_mode, FeatureMode):
            raise TypeError("feature_mode must be a FeatureMode")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if not 0.0 <= self.focal_alpha <= 1.0:
            raise ValueError("focal_alpha must be in [0, 1]")
        if self.focal_gamma < 0.0:
            raise ValueError("focal_gamma cannot be negative")
        if self.focal_reduction not in {"mean", "sum"}:
            raise ValueError("training focal_reduction must be 'mean' or 'sum'")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if self.best_metric not in {"val_auc", "val_accuracy", "val_loss"}:
            raise ValueError("unsupported best_metric")
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience <= 0
        ):
            raise ValueError("early_stopping_patience must be positive")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta cannot be negative")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        self.model_config.validate()

    @property
    def model_config(self) -> TransformerConfig:
        return self.model or default_transformer_config(self.feature_mode)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    def as_dict(self, *, resolved_device: str | None = None) -> dict[str, Any]:
        """Return checkpoint-safe configuration without authentication data."""
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
            "features": _feature_config(self.feature_mode),
            "model": asdict(self.model_config),
            "training": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "gradient_accumulation_steps": (
                    self.gradient_accumulation_steps
                ),
                "effective_batch_size": self.effective_batch_size,
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
                "loss": "FocalLoss",
                "focal_alpha": self.focal_alpha,
                "focal_gamma": self.focal_gamma,
                "focal_reduction": self.focal_reduction,
                "best_metric": self.best_metric,
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_min_delta": self.early_stopping_min_delta,
                "seed": self.seed,
                "resume_weights": str(self.resume) if self.resume else None,
            },
            "runtime": {
                "requested_device": self.device,
                "resolved_device": resolved_device,
                "num_workers": self.num_workers,
                "pin_memory": self.pin_memory,
                "log_every": self.log_every,
            },
        }


def _feature_config(mode: FeatureMode) -> dict[str, Any]:
    encoders = DEFAULT_MODEL_CONFIG.encoders
    if mode.uses_raw_inputs:
        modality_embedding_dim = default_transformer_config(
            mode
        ).modality_embedding_dim
        return {
            "mode": mode.value,
            "visual_encoder": "ResNet18VisualEncoder",
            "visual_pretrained": encoders.resnet18_pretrained,
            "visual_trainable": encoders.resnet18_trainable,
            "visual_embedding_dim": modality_embedding_dim,
            "text_encoder": "SentenceTransformerTextEncoder",
            "text_model_name": encoders.sentence_transformer_model_name,
            "text_backbone_trainable": (
                encoders.sentence_transformer_trainable
            ),
            "text_projection_trainable": True,
            "text_embedding_dim": modality_embedding_dim,
        }
    return {
        "mode": mode.value,
        "encoder": "from_embedding_manifest",
        "precomputed": True,
        "trainable": False,
    }
