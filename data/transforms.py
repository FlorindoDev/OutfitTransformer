"""Image preprocessing factories for the supported visual encoders."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ImageTransform = Callable[[Image.Image], Tensor]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def build_resnet18_transform(
    *,
    augment: bool = False,
    image_size: int = 224,
) -> ImageTransform:
    """Build ImageNet-compatible preprocessing for ``ResNet18VisualEncoder``."""
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    if augment:
        geometry: list[Any] = [
            transforms.RandomResizedCrop(
                image_size,
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        resize_size = round(image_size * 256 / 224)
        geometry = [
            transforms.Resize(
                resize_size,
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(image_size),
        ]

    return transforms.Compose(
        [
            *geometry,
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


class _ImageProcessor(Protocol):
    def __call__(
        self,
        *,
        images: Image.Image,
        return_tensors: str,
    ) -> Any: ...


@dataclass(frozen=True)
class _FashionCLIPTransform:
    processor: _ImageProcessor

    def __call__(self, image: Image.Image) -> Tensor:
        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"]
        if pixel_values.ndim != 4 or pixel_values.size(0) != 1:
            raise ValueError(
                "FashionCLIP processor must return pixel_values shaped [1, C, H, W]"
            )
        return pixel_values.squeeze(0)


def build_fashion_clip_transform(
    model_name: str = "patrickjohncyh/fashion-clip",
) -> ImageTransform:
    """Build preprocessing from the selected FashionCLIP checkpoint."""
    if not model_name.strip():
        raise ValueError("model_name cannot be empty")
    try:
        from transformers import AutoImageProcessor
    except ImportError as error:
        raise ImportError(
            "build_fashion_clip_transform requires the 'transformers' package"
        ) from error

    processor = AutoImageProcessor.from_pretrained(model_name)
    return _FashionCLIPTransform(processor)
