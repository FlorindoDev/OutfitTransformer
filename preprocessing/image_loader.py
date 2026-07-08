from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps

ColorMode = Literal["RGB", "RGBA"]
SUPPORTED_COLOR_MODES = {"RGB", "RGBA"}


def load_image_from_bytes(image_bytes: bytes, mode: ColorMode = "RGB") -> Image.Image:
    if not image_bytes:
        raise ValueError("image_bytes must not be empty")

    with Image.open(BytesIO(image_bytes)) as image:
        return normalize_image(image, mode)


def load_image_from_path(path: str | Path, mode: ColorMode = "RGB") -> Image.Image:
    with Image.open(path) as image:
        return normalize_image(image, mode)


def normalize_image(image: Image.Image, mode: ColorMode = "RGB") -> Image.Image:
    _validate_color_mode(mode)

    oriented_image = ImageOps.exif_transpose(image)
    normalized_image = oriented_image.convert(mode)
    normalized_image.load()

    return normalized_image


def _validate_color_mode(mode: str) -> None:
    if mode not in SUPPORTED_COLOR_MODES:
        raise ValueError("mode must be either 'RGB' or 'RGBA'")
