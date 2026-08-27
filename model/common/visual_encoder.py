"""Interchangeable visual encoders for clothing images."""

import base64
from abc import ABC, abstractmethod
from io import BytesIO

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms.functional import to_pil_image

from .openrouter import OpenRouterEmbeddingClient


class VisualEncoder(nn.Module, ABC):
    """Contract required by the multimodal OutfitTransformer."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Number of features returned for each image."""

    @abstractmethod
    def forward(self, images: Tensor) -> Tensor:
        """Encode images shaped ``[items, channels, height, width]``."""


class ResNet18VisualEncoder(VisualEncoder):
    """ResNet-18 without its ImageNet classification head."""

    def __init__(
        self,
        *,
        pretrained: bool = True,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self._output_dim = self.backbone.fc.in_features
        self.backbone.add_module("fc", nn.Identity())
        self._trainable = trainable

        if not trainable:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def train(self, mode: bool = True) -> "ResNet18VisualEncoder":
        super().train(mode)
        if not self._trainable:
            self.backbone.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError(
                "images must have shape [items, channels, height, width]"
            )
        if images.size(0) == 0:
            raise ValueError("images cannot be empty")
        if images.size(1) != 3:
            raise ValueError("ResNet-18 expects three-channel images")
        return self.backbone(images)


class FashionCLIPVisualEncoder(VisualEncoder):
    """FashionCLIP ViT image tower returning projected CLIP features."""

    DEFAULT_MODEL_NAME = "patrickjohncyh/fashion-clip"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import CLIPVisionModelWithProjection
        except ImportError as error:
            raise ImportError(
                "FashionCLIPVisualEncoder requires the 'transformers' package"
            ) from error

        self.backbone = CLIPVisionModelWithProjection.from_pretrained(model_name)
        projection_dim = self.backbone.config.projection_dim
        if projection_dim is None or projection_dim <= 0:
            raise ValueError(
                "FashionCLIP visual config must define a positive projection_dim"
            )
        self._output_dim = projection_dim
        self._trainable = trainable

        if not trainable:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def train(self, mode: bool = True) -> "FashionCLIPVisualEncoder":
        super().train(mode)
        if not self._trainable:
            self.backbone.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError(
                "images must have shape [items, channels, height, width]"
            )
        if images.size(0) == 0:
            raise ValueError("images cannot be empty")
        if images.size(1) != 3:
            raise ValueError("FashionCLIP expects three-channel images")
        return self.backbone(pixel_values=images).image_embeds


class OpenRouterVisualEncoder(VisualEncoder):
    """Remote image encoder backed by OpenRouter's embedding API."""

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

    def forward(self, images: Tensor) -> Tensor:
        _validate_openrouter_images(images)
        cpu_images = images.detach().cpu()
        inputs = tuple(
            {
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(image)},
                    }
                ]
            }
            for image in cpu_images
        )
        return self.client.embed_multimodal(
            inputs,
            device=self.get_buffer("_device_anchor").device,
        )


def _validate_openrouter_images(images: Tensor) -> None:
    if images.ndim != 4:
        raise ValueError("images must have shape [items, channels, height, width]")
    if images.size(0) == 0:
        raise ValueError("images cannot be empty")
    if images.size(1) != 3:
        raise ValueError("OpenRouter expects three-channel images")
    if not torch.is_floating_point(images):
        raise TypeError("OpenRouter images must be floating point")
    if not bool(torch.isfinite(images).all()):
        raise ValueError("OpenRouter images must contain only finite values")
    if bool((images < 0.0).any()) or bool((images > 1.0).any()):
        raise ValueError("OpenRouter images must contain values in [0, 1]")


def _image_data_url(image: Tensor) -> str:
    buffer = BytesIO()
    to_pil_image(image).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
