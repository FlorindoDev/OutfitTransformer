"""Central, validated defaults for all model components."""

from dataclasses import dataclass, field
from typing import Literal


ActivationName = Literal["relu", "gelu", "mish"]
Reduction = Literal["none", "mean", "sum"]


@dataclass(frozen=True)
class TransformerConfig:
    """Architecture shared by common and task-specific Transformers."""

    modality_embedding_dim: int = 512
    layers: int = 6
    attention_heads: int = 16
    feedforward_dim: int = 2024
    dropout: float = 0.3
    activation: ActivationName = "mish"
    norm_first: bool = True
    layer_norm_epsilon: float = 1e-5
    max_items: int = 16
    normalization_epsilon: float = 1e-12
    embedding_initialization_std: float = 0.02

    @property
    def model_dim(self) -> int:
        return 2 * self.modality_embedding_dim

    def validate(self) -> None:
        if self.modality_embedding_dim <= 0:
            raise ValueError("modality_embedding_dim must be positive")
        if self.layers <= 0:
            raise ValueError("layers must be positive")
        if self.attention_heads <= 0:
            raise ValueError("attention_heads must be positive")
        if self.model_dim % self.attention_heads != 0:
            raise ValueError("model_dim must be divisible by attention_heads")
        if self.feedforward_dim <= 0:
            raise ValueError("feedforward_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.activation not in {"relu", "gelu", "mish"}:
            raise ValueError("activation must be 'relu', 'gelu', or 'mish'")
        if not isinstance(self.norm_first, bool):
            raise TypeError("norm_first must be boolean")
        if self.layer_norm_epsilon <= 0.0:
            raise ValueError("layer_norm_epsilon must be positive")
        if self.max_items <= 0:
            raise ValueError("max_items must be positive")
        if self.normalization_epsilon <= 0.0:
            raise ValueError("normalization_epsilon must be positive")
        if self.embedding_initialization_std <= 0.0:
            raise ValueError("embedding_initialization_std must be positive")


@dataclass(frozen=True)
class EncoderConfig:
    """Defaults for local and remote visual/text encoders."""

    sentence_transformer_model_name: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    sentence_transformer_trainable: bool = False
    fashion_clip_model_name: str = "patrickjohncyh/fashion-clip"
    fashion_clip_trainable: bool = True
    resnet18_pretrained: bool = True
    resnet18_trainable: bool = True
    openrouter_model_name: str = "google/gemini-embedding-2"
    openrouter_output_dim: int = 512
    openrouter_request_batch_size: int = 8
    openrouter_image_size: int = 224
    openrouter_timeout_seconds: float = 60.0
    openrouter_max_retries: int = 3
    openrouter_api_base: str = "https://openrouter.ai/api/v1"

    def validate(self) -> None:
        boolean_values = {
            "sentence_transformer_trainable": self.sentence_transformer_trainable,
            "fashion_clip_trainable": self.fashion_clip_trainable,
            "resnet18_pretrained": self.resnet18_pretrained,
            "resnet18_trainable": self.resnet18_trainable,
        }
        for name, value in boolean_values.items():
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be boolean")
        _validate_text(
            self.sentence_transformer_model_name,
            "sentence_transformer_model_name",
        )
        _validate_text(self.fashion_clip_model_name, "fashion_clip_model_name")
        _validate_text(self.openrouter_model_name, "openrouter_model_name")
        _validate_text(self.openrouter_api_base, "openrouter_api_base")
        if self.openrouter_output_dim <= 0:
            raise ValueError("openrouter_output_dim must be positive")
        if self.openrouter_request_batch_size <= 0:
            raise ValueError("openrouter_request_batch_size must be positive")
        if self.openrouter_image_size <= 0:
            raise ValueError("openrouter_image_size must be positive")
        if self.openrouter_timeout_seconds <= 0.0:
            raise ValueError("openrouter_timeout_seconds must be positive")
        if self.openrouter_max_retries < 0:
            raise ValueError("openrouter_max_retries cannot be negative")


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


@dataclass(frozen=True)
class ComplementaryItemConfig:
    """Defaults specific to complementary item retrieval."""

    embedding_dim: int = 128
    normalize_embeddings: bool = False
    triplet_margin: float = 2.0
    loss_reduction: Reduction = "mean"

    def validate(self) -> None:
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not isinstance(self.normalize_embeddings, bool):
            raise TypeError("normalize_embeddings must be boolean")
        if self.triplet_margin < 0.0:
            raise ValueError("triplet_margin cannot be negative")
        if self.loss_reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                "loss_reduction must be 'none', 'mean', or 'sum'"
            )


def _new_classic_transformer() -> TransformerConfig:
    return TransformerConfig(
        modality_embedding_dim=512,
        dropout=0.1,
        norm_first=False,
    )


@dataclass(frozen=True)
class ModelConfig:
    """Single entry point for shared and task-specific model defaults."""

    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    classic_transformer: TransformerConfig = field(
        default_factory=lambda: TransformerConfig(
            modality_embedding_dim=64,
            feedforward_dim=512,
            dropout=0.1,
            norm_first=False,
        )
    )
    new_classic_transformer: TransformerConfig = field(
        default_factory=_new_classic_transformer
    )
    encoders: EncoderConfig = field(default_factory=EncoderConfig)
    compatibility: CompatibilityConfig = field(
        default_factory=CompatibilityConfig
    )
    complementary_item: ComplementaryItemConfig = field(
        default_factory=ComplementaryItemConfig
    )

    def validate(self) -> None:
        self.transformer.validate()
        self.classic_transformer.validate()
        self.new_classic_transformer.validate()
        self.encoders.validate()
        self.compatibility.validate()
        self.complementary_item.validate()


def _validate_text(value: str, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} cannot be empty")


DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_MODEL_CONFIG.validate()
DEFAULT_COMPATIBILITY_CONFIG = DEFAULT_MODEL_CONFIG.compatibility
DEFAULT_CIR_CONFIG = DEFAULT_MODEL_CONFIG.complementary_item
