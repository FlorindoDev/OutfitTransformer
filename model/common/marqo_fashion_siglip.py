"""Frozen Marqo FashionSigLIP adapters sharing one multimodal backbone."""

from collections.abc import Mapping, Sequence
from typing import Literal, Protocol, cast

from torch import Tensor, nn

from .config import DEFAULT_MODEL_CONFIG
from .text_encoder import TextEncoder, _validate_descriptions
from .visual_encoder import VisualEncoder


DEFAULT_MODEL_NAME = DEFAULT_MODEL_CONFIG.encoders.marqo_fashion_siglip_model_name


class _TextTower(Protocol):
    output_dim: int
    context_length: int


class _MultimodalModel(Protocol):
    text: _TextTower


class _MarqoFeatures(Protocol):
    """Feature API provided by Marqo's dynamically loaded remote model."""

    model: _MultimodalModel

    def get_image_features(
        self, pixel_values: Tensor, *, normalize: bool = False
    ) -> Tensor: ...

    def get_text_features(
        self, input_ids: Tensor, *, normalize: bool = False
    ) -> Tensor: ...


class _TextProcessor(Protocol):
    def __call__(
        self,
        *,
        text: list[str],
        padding: Literal["max_length"],
        truncation: bool,
        max_length: int,
        return_tensors: Literal["pt"],
    ) -> Mapping[str, Tensor]: ...


def _load_backbone(model_name: str) -> nn.Module:
    from transformers.models.auto.modeling_auto import AutoModel

    # Marqo initializes OpenCLIP weights inside __init__, which cannot run on meta.
    return AutoModel.from_pretrained(
        model_name, trust_remote_code=True, low_cpu_mem_usage=False
    )


def _positive_dimension(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Marqo FashionSigLIP {name} must be a positive integer")
    return value


class MarqoFashionSigLIPVisualEncoder(VisualEncoder):
    """Encode preprocessed RGB images with the frozen Marqo image tower."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        super().__init__()
        self.backbone = _load_backbone(model_name)
        features = cast(_MarqoFeatures, self.backbone)
        # Both towers share this embedding space; the timm visual wrapper has no output_dim.
        self._output_dim = _positive_dimension(
            features.model.text.output_dim, "output_dim"
        )
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def train(self, mode: bool = True) -> "MarqoFashionSigLIPVisualEncoder":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.size(1) != 3:
            raise ValueError("images must have shape [items, 3, height, width]")
        if images.size(0) == 0:
            raise ValueError("images cannot be empty")
        features = cast(_MarqoFeatures, self.backbone)
        return features.get_image_features(images, normalize=False)


class MarqoFashionSigLIPTextEncoder(TextEncoder):
    """Encode descriptions using Marqo's text cleaning and fixed padding."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        from transformers.models.auto.processing_auto import AutoProcessor

        self.backbone = backbone if backbone is not None else _load_backbone(model_name)
        self.processor = cast(
            _TextProcessor,
            AutoProcessor.from_pretrained(model_name, trust_remote_code=True),
        )
        features = cast(_MarqoFeatures, self.backbone)
        text_tower = features.model.text
        self._output_dim = _positive_dimension(text_tower.output_dim, "text output_dim")
        self._max_length = _positive_dimension(text_tower.context_length, "context_length")
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def train(self, mode: bool = True) -> "MarqoFashionSigLIPTextEncoder":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, descriptions: Sequence[str]) -> Tensor:
        _validate_descriptions(descriptions)
        device = next(self.backbone.parameters()).device
        processed = self.processor(
            text=list(descriptions),
            padding="max_length",
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        features = cast(_MarqoFeatures, self.backbone)
        return features.get_text_features(
            processed["input_ids"].to(device), normalize=False
        )
