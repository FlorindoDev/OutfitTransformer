from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class BoundingBox:
    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        if self.left < 0:
            raise ValueError("left must be greater than or equal to 0")
        if self.top < 0:
            raise ValueError("top must be greater than or equal to 0")
        if self.right <= self.left:
            raise ValueError("right must be greater than left")
        if self.bottom <= self.top:
            raise ValueError("bottom must be greater than top")

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def as_pillow_box(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom


@dataclass(frozen=True)
class GarmentCropConfig:
    mask_threshold: int = 127
    margin_ratio: float = 0.10
    min_margin_pixels: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.mask_threshold <= 255:
            raise ValueError("mask_threshold must be between 0 and 255")
        if self.margin_ratio < 0:
            raise ValueError("margin_ratio must be greater than or equal to 0")
        if self.min_margin_pixels < 0:
            raise ValueError("min_margin_pixels must be greater than or equal to 0")


@dataclass(frozen=True)
class CropResult:
    image: Image.Image
    mask: Image.Image
    bounding_box: BoundingBox


class CropDependencyError(RuntimeError):
    pass


class EmptyMaskError(ValueError):
    pass


def find_foreground_bounding_box(
    mask: Image.Image,
    config: GarmentCropConfig | None = None,
) -> BoundingBox | None:
    config = config or GarmentCropConfig()
    cv2, np = _load_crop_dependencies()
    mask_array = _to_binary_mask_array(mask, config.mask_threshold, np)
    foreground_points = cv2.findNonZero(mask_array)
    if foreground_points is None:
        return None

    left, top, width, height = (
        int(value)
        for value in cv2.boundingRect(foreground_points)
    )

    return BoundingBox(
        left=left,
        top=top,
        right=left + width,
        bottom=top + height,
    )


def crop_to_bounding_box(
    image: Image.Image,
    bounding_box: BoundingBox,
) -> Image.Image:
    _validate_bounding_box_inside_image(image, bounding_box)

    return image.crop(bounding_box.as_pillow_box())


def crop_garment_with_margin(
    image: Image.Image,
    mask: Image.Image,
    config: GarmentCropConfig | None = None,
) -> CropResult:
    config = config or GarmentCropConfig()
    _validate_image_and_mask_size(image, mask)
    bounding_box = find_foreground_bounding_box(mask, config)
    if bounding_box is None:
        raise EmptyMaskError("mask has no foreground pixels")

    cropped_image = crop_to_bounding_box(image, bounding_box)
    cropped_mask = crop_to_bounding_box(mask.convert("L"), bounding_box)
    margin_x, margin_y = _calculate_margin_pixels(bounding_box, config)

    return CropResult(
        image=_add_margin(
            cropped_image,
            margin_x,
            margin_y,
            _fill_color_for_image(cropped_image),
        ),
        mask=_add_margin(cropped_mask, margin_x, margin_y, 0),
        bounding_box=bounding_box,
    )


def _load_crop_dependencies() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise CropDependencyError(
            "Cropping requires opencv-python-headless and numpy. "
            "Install project requirements before using crop_garment_with_margin()."
        ) from exc

    return cv2, np


def _to_binary_mask_array(mask: Image.Image, threshold: int, np: Any) -> Any:
    grayscale_mask = mask.convert("L")
    grayscale_mask.load()
    mask_array = np.asarray(grayscale_mask, dtype=np.uint8)

    return np.where(mask_array > threshold, 255, 0).astype(np.uint8)


def _validate_bounding_box_inside_image(
    image: Image.Image,
    bounding_box: BoundingBox,
) -> None:
    if bounding_box.right > image.width:
        raise ValueError("bounding_box.right must be inside image width")
    if bounding_box.bottom > image.height:
        raise ValueError("bounding_box.bottom must be inside image height")


def _validate_image_and_mask_size(image: Image.Image, mask: Image.Image) -> None:
    if image.size != mask.size:
        raise ValueError("image and mask must have the same size")


def _calculate_margin_pixels(
    bounding_box: BoundingBox,
    config: GarmentCropConfig,
) -> tuple[int, int]:
    margin_x = max(config.min_margin_pixels, ceil(bounding_box.width * config.margin_ratio))
    margin_y = max(config.min_margin_pixels, ceil(bounding_box.height * config.margin_ratio))

    return margin_x, margin_y


def _add_margin(
    image: Image.Image,
    margin_x: int,
    margin_y: int,
    fill_color: Any,
) -> Image.Image:
    if margin_x == 0 and margin_y == 0:
        return image.copy()

    canvas = Image.new(
        image.mode,
        (image.width + (margin_x * 2), image.height + (margin_y * 2)),
        fill_color,
    )
    canvas.paste(image, (margin_x, margin_y))

    return canvas


def _fill_color_for_image(image: Image.Image) -> Any:
    if "A" in image.getbands():
        return tuple(0 for _ in image.getbands())
    if image.mode == "RGB":
        return (255, 255, 255)

    return 0
