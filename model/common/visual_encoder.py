"""Interchangeable visual encoders for clothing images."""

from abc import ABC, abstractmethod

from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


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
        self.backbone.fc = nn.Identity()
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
