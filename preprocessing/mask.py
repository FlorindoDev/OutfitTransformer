from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from .background import BackgroundRemovalConfig, remove_background
from .image_loader import normalize_image


@dataclass(frozen=True)
class AlphaMaskConfig:
    alpha_threshold: int = 40

    def __post_init__(self) -> None:
        if not 0 <= self.alpha_threshold <= 255:
            raise ValueError("alpha_threshold must be between 0 and 255")


@dataclass(frozen=True)
class MaskCleaningConfig:
    mask_threshold: int = 127
    opening_kernel_size: int = 3
    closing_kernel_size: int = 5
    min_component_area: int = 64

    def __post_init__(self) -> None:
        if not 0 <= self.mask_threshold <= 255:
            raise ValueError("mask_threshold must be between 0 and 255")
        if self.opening_kernel_size < 0:
            raise ValueError("opening_kernel_size must be greater than or equal to 0")
        if self.closing_kernel_size < 0:
            raise ValueError("closing_kernel_size must be greater than or equal to 0")
        if self.min_component_area < 0:
            raise ValueError("min_component_area must be greater than or equal to 0")


@dataclass(frozen=True)
class MainComponentConfig:
    mask_threshold: int = 127
    min_component_area: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.mask_threshold <= 255:
            raise ValueError("mask_threshold must be between 0 and 255")
        if self.min_component_area < 0:
            raise ValueError("min_component_area must be greater than or equal to 0")


class MaskCleaningDependencyError(RuntimeError):
    pass


def extract_alpha_mask(
    image: Image.Image,
    config: AlphaMaskConfig | None = None,
) -> Image.Image:
    config = config or AlphaMaskConfig()
    rgba_image = normalize_image(image, mode="RGBA")
    alpha_channel = rgba_image.getchannel("A")
    threshold_table = [
        255 if value > config.alpha_threshold else 0
        for value in range(256)
    ]

    return alpha_channel.point(threshold_table)


def clean_binary_mask(
    mask: Image.Image,
    config: MaskCleaningConfig | None = None,
) -> Image.Image:
    config = config or MaskCleaningConfig()
    cv2, np = _load_mask_cleaning_dependencies()
    mask_array = _to_binary_mask_array(mask, config.mask_threshold, np)
    mask_array = _apply_morphology(
        mask_array,
        operation=cv2.MORPH_OPEN,
        kernel_size=config.opening_kernel_size,
        cv2=cv2,
        np=np,
    )
    mask_array = _apply_morphology(
        mask_array,
        operation=cv2.MORPH_CLOSE,
        kernel_size=config.closing_kernel_size,
        cv2=cv2,
        np=np,
    )
    mask_array = _remove_small_components(
        mask_array,
        min_component_area=config.min_component_area,
        cv2=cv2,
        np=np,
    )

    return Image.fromarray(mask_array)


def keep_main_component(
    mask: Image.Image,
    config: MainComponentConfig | None = None,
) -> Image.Image:
    config = config or MainComponentConfig()
    cv2, np = _load_mask_cleaning_dependencies()
    mask_array = _to_binary_mask_array(mask, config.mask_threshold, np)
    main_component_array = _keep_largest_component(
        mask_array,
        min_component_area=config.min_component_area,
        cv2=cv2,
        np=np,
    )

    return Image.fromarray(main_component_array)


def remove_background_and_extract_mask(
    image: Image.Image,
    background_config: BackgroundRemovalConfig | None = None,
    mask_config: AlphaMaskConfig | None = None,
) -> tuple[Image.Image, Image.Image]:
    foreground_image = remove_background(image, background_config)
    mask = extract_alpha_mask(foreground_image, mask_config)

    return foreground_image, mask


def _load_mask_cleaning_dependencies() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise MaskCleaningDependencyError(
            "Mask cleaning requires opencv-python-headless and numpy. "
            "Install project requirements before using clean_binary_mask()."
        ) from exc

    return cv2, np


def _to_binary_mask_array(mask: Image.Image, threshold: int, np: Any) -> Any:
    grayscale_mask = mask.convert("L")
    grayscale_mask.load()
    mask_array = np.asarray(grayscale_mask, dtype=np.uint8)

    return np.where(mask_array > threshold, 255, 0).astype(np.uint8)


def _apply_morphology(
    mask_array: Any,
    operation: int,
    kernel_size: int,
    cv2: Any,
    np: Any,
) -> Any:
    if kernel_size <= 1:
        return mask_array

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    return cv2.morphologyEx(mask_array, operation, kernel)


def _remove_small_components(
    mask_array: Any,
    min_component_area: int,
    cv2: Any,
    np: Any,
) -> Any:
    if min_component_area <= 0:
        return mask_array

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_array,
        connectivity=8,
    )
    cleaned_mask = np.zeros_like(mask_array)
    for component_label in range(1, component_count):
        component_area = stats[component_label, cv2.CC_STAT_AREA]
        if component_area >= min_component_area:
            cleaned_mask[labels == component_label] = 255

    return cleaned_mask


def _keep_largest_component(
    mask_array: Any,
    min_component_area: int,
    cv2: Any,
    np: Any,
) -> Any:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_array,
        connectivity=8,
    )
    if component_count <= 1:
        return np.zeros_like(mask_array)

    largest_label = max(
        range(1, component_count),
        key=lambda component_label: stats[component_label, cv2.CC_STAT_AREA],
    )
    largest_area = stats[largest_label, cv2.CC_STAT_AREA]
    if largest_area < min_component_area:
        return np.zeros_like(mask_array)

    main_component_mask = np.zeros_like(mask_array)
    main_component_mask[labels == largest_label] = 255

    return main_component_mask
