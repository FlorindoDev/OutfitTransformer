from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from typing import Any, Callable

from PIL import Image

from .image_loader import load_image_from_bytes, normalize_image

DEFAULT_BACKGROUND_REMOVAL_MODEL = "isnet-general-use"


class BackgroundRemovalDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackgroundRemovalConfig:
    model_name: str = DEFAULT_BACKGROUND_REMOVAL_MODEL

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")


def remove_background(
    image: Image.Image,
    config: BackgroundRemovalConfig | None = None,
) -> Image.Image:
    config = config or BackgroundRemovalConfig()
    normalized_image = normalize_image(image, mode="RGBA")
    session = _get_background_session(config.model_name)
    remove = _get_rembg_remove()

    output = remove(normalized_image, session=session)

    return _normalize_rembg_output(output)


def remove_background_from_bytes(
    image_bytes: bytes,
    config: BackgroundRemovalConfig | None = None,
) -> Image.Image:
    image = load_image_from_bytes(image_bytes, mode="RGBA")

    return remove_background(image, config)


def clear_background_session_cache() -> None:
    _get_background_session.cache_clear()


@lru_cache(maxsize=4)
def _get_background_session(model_name: str) -> Any:
    new_session = _get_rembg_new_session()

    return new_session(model_name)


def _get_rembg_new_session() -> Callable[[str], Any]:
    try:
        from rembg import new_session
    except ImportError as exc:
        raise BackgroundRemovalDependencyError(
            'rembg is required for background removal. Install it with: pip install "rembg[cpu]"'
        ) from exc

    return new_session


def _get_rembg_remove() -> Callable[..., Any]:
    try:
        from rembg import remove
    except ImportError as exc:
        raise BackgroundRemovalDependencyError(
            'rembg is required for background removal. Install it with: pip install "rembg[cpu]"'
        ) from exc

    return remove


def _normalize_rembg_output(output: Image.Image | bytes) -> Image.Image:
    if isinstance(output, Image.Image):
        return normalize_image(output, mode="RGBA")

    if isinstance(output, bytes):
        with Image.open(BytesIO(output)) as image:
            return normalize_image(image, mode="RGBA")

    raise TypeError("rembg output must be a PIL image or bytes")
