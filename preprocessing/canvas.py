from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from PIL import Image, ImageChops, ImageOps

from .image_loader import normalize_image

RGBColor: TypeAlias = tuple[int, int, int]
DEFAULT_BACKGROUND_COLOR: RGBColor = (255, 255, 255)
DEFAULT_CANVAS_SIZE = 512


@dataclass(frozen=True)
class CanvasConfig:
    size: int = DEFAULT_CANVAS_SIZE
    background_color: RGBColor = DEFAULT_BACKGROUND_COLOR
    allow_upscale: bool = True

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("size must be greater than 0")
        _validate_rgb_color(self.background_color)


def compose_on_background(
    image: Image.Image,
    mask: Image.Image | None = None,
    background_color: RGBColor = DEFAULT_BACKGROUND_COLOR,
) -> Image.Image:
    _validate_rgb_color(background_color)
    rgba_image = normalize_image(image, mode="RGBA")
    if mask is not None:
        _validate_image_and_mask_size(rgba_image, mask)
        rgba_image = _apply_mask_to_alpha(rgba_image, mask)

    background = Image.new(
        "RGBA",
        rgba_image.size,
        (*background_color, 255),
    )
    background.alpha_composite(rgba_image)

    return background.convert("RGB")


def center_on_square_canvas(
    image: Image.Image,
    config: CanvasConfig | None = None,
) -> Image.Image:
    config = config or CanvasConfig()
    rgb_image = normalize_image(image, mode="RGB")
    resized_image = _resize_to_fit_square(rgb_image, config)
    canvas = Image.new(
        "RGB",
        (config.size, config.size),
        config.background_color,
    )
    paste_position = _center_position(canvas.size, resized_image.size)
    canvas.paste(resized_image, paste_position)

    return canvas


def create_square_garment_image(
    image: Image.Image,
    mask: Image.Image | None = None,
    config: CanvasConfig | None = None,
) -> Image.Image:
    config = config or CanvasConfig()
    image_on_background = compose_on_background(
        image,
        mask=mask,
        background_color=config.background_color,
    )

    return center_on_square_canvas(image_on_background, config)


def _apply_mask_to_alpha(image: Image.Image, mask: Image.Image) -> Image.Image:
    alpha = image.getchannel("A")
    mask_alpha = mask.convert("L")
    masked_alpha = ImageChops.multiply(alpha, mask_alpha)
    masked_image = image.copy()
    masked_image.putalpha(masked_alpha)

    return masked_image


def _resize_to_fit_square(image: Image.Image, config: CanvasConfig) -> Image.Image:
    if config.allow_upscale:
        return ImageOps.contain(
            image,
            (config.size, config.size),
            method=Image.Resampling.LANCZOS,
        )

    scale = min(
        config.size / image.width,
        config.size / image.height,
        1.0,
    )
    resized_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    if resized_size == image.size:
        return image.copy()

    return image.resize(resized_size, Image.Resampling.LANCZOS)


def _center_position(
    canvas_size: tuple[int, int],
    image_size: tuple[int, int],
) -> tuple[int, int]:
    return (
        (canvas_size[0] - image_size[0]) // 2,
        (canvas_size[1] - image_size[1]) // 2,
    )


def _validate_image_and_mask_size(image: Image.Image, mask: Image.Image) -> None:
    if image.size != mask.size:
        raise ValueError("image and mask must have the same size")


def _validate_rgb_color(color: RGBColor) -> None:
    if len(color) != 3:
        raise ValueError("background_color must have exactly 3 channels")
    for channel in color:
        if not 0 <= channel <= 255:
            raise ValueError("background_color channels must be between 0 and 255")
