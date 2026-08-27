"""Interchangeable text encoders for clothing descriptions."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .openrouter import OpenRouterEmbeddingClient


class TextEncoder(nn.Module, ABC):
    """Contract required by the multimodal OutfitTransformer."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Number of features returned for each description."""

    @abstractmethod
    def forward(self, descriptions: Sequence[str]) -> Tensor:
        """Encode one feature vector per description."""


class SentenceTransformerTextEncoder(TextEncoder):
    """SentenceTransformer backbone with an optional trainable projection."""

    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        output_dim: int = 512,
        trainable_backbone: bool = False,
    ) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "SentenceTransformerTextEncoder requires "
                "the 'sentence-transformers' package"
            ) from error

        self.backbone = SentenceTransformer(model_name)
        backbone_dim = self.backbone.get_sentence_embedding_dimension()
        if backbone_dim is None:
            raise ValueError("text backbone does not expose its embedding dimension")

        self.projection = (
            nn.Identity()
            if backbone_dim == output_dim
            else nn.Linear(backbone_dim, output_dim)
        )
        self._output_dim = output_dim
        self._trainable_backbone = trainable_backbone

        if not trainable_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def train(self, mode: bool = True) -> "SentenceTransformerTextEncoder":
        super().train(mode)
        if not self._trainable_backbone:
            self.backbone.eval()
        return self

    def forward(self, descriptions: Sequence[str]) -> Tensor:
        _validate_descriptions(descriptions)
        device = next(self.backbone.parameters()).device
        features = {
            name: value.to(device)
            for name, value in self.backbone.tokenize(list(descriptions)).items()
        }

        if self._trainable_backbone:
            sentence_features = self.backbone(features)["sentence_embedding"]
        else:
            with torch.no_grad():
                sentence_features = self.backbone(features)["sentence_embedding"]
        return self.projection(sentence_features)


class FashionCLIPTextEncoder(TextEncoder):
    """FashionCLIP text tower returning projected CLIP features."""

    DEFAULT_MODEL_NAME = "patrickjohncyh/fashion-clip"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoTokenizer, CLIPTextModelWithProjection
        except ImportError as error:
            raise ImportError(
                "FashionCLIPTextEncoder requires the 'transformers' package"
            ) from error

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = CLIPTextModelWithProjection.from_pretrained(model_name)
        projection_dim = self.backbone.config.projection_dim
        if projection_dim is None or projection_dim <= 0:
            raise ValueError(
                "FashionCLIP text config must define a positive projection_dim"
            )
        self._max_length = _max_position_embeddings(self.backbone)
        self._output_dim = projection_dim
        self._trainable = trainable

        if not trainable:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def train(self, mode: bool = True) -> "FashionCLIPTextEncoder":
        super().train(mode)
        if not self._trainable:
            self.backbone.eval()
        return self

    def forward(self, descriptions: Sequence[str]) -> Tensor:
        _validate_descriptions(descriptions)
        device = next(self.backbone.parameters()).device
        tokens = self.tokenizer(
            list(descriptions),
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        ).to(device)
        return self.backbone(**tokens).text_embeds


class OpenRouterTextEncoder(TextEncoder):
    """Remote text encoder backed by OpenRouter's embedding API."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        *,
        output_dim: int = 512,
        request_batch_size: int = 8,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__()
        self.model_name = model_name.strip()
        self.client = OpenRouterEmbeddingClient(
            self.model_name,
            api_key,
            output_dim=output_dim,
            request_batch_size=request_batch_size,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        self._output_dim = output_dim
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, descriptions: Sequence[str]) -> Tensor:
        _validate_descriptions(descriptions)
        return self.client.embed_texts(
            descriptions,
            device=self.get_buffer("_device_anchor").device,
        )


def _validate_descriptions(descriptions: Sequence[str]) -> None:
    if not descriptions:
        raise ValueError("descriptions cannot be empty")
    if any(not isinstance(text, str) or not text.strip() for text in descriptions):
        raise ValueError("descriptions must contain non-empty strings")


def _max_position_embeddings(backbone: nn.Module) -> int:
    config = getattr(backbone, "config", None)
    max_length = getattr(config, "max_position_embeddings", None)
    if (
        not isinstance(max_length, int)
        or isinstance(max_length, bool)
        or max_length <= 0
    ):
        raise ValueError(
            "FashionCLIP text config must define positive "
            "max_position_embeddings"
        )
    return max_length
