from __future__ import annotations

import argparse
from pathlib import Path

from preprocessing import (
    AlphaMaskConfig,
    BackgroundRemovalConfig,
    BackgroundRemovalDependencyError,
    CanvasConfig,
    GarmentCropConfig,
    MainComponentConfig,
    MaskCleaningConfig,
    clean_binary_mask,
    crop_garment_with_margin,
    create_square_garment_image,
    extract_alpha_mask,
    keep_main_component,
    load_image_from_path,
    remove_background,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporary preprocessing demo")
    parser.add_argument("image", type=Path, help="input image path")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output PNG path",
    )
    parser.add_argument(
        "--model",
        default="isnet-general-use",
        help="rembg model name",
    )
    parser.add_argument(
        "--only-normalize",
        action="store_true",
        help="only apply EXIF orientation and RGBA conversion",
    )
    parser.add_argument(
        "--mask-output",
        type=Path,
        default=None,
        help="optional output path for alpha mask PNG",
    )
    parser.add_argument(
        "--clean-mask-output",
        type=Path,
        default=None,
        help="optional output path for cleaned alpha mask PNG",
    )
    parser.add_argument(
        "--main-mask-output",
        type=Path,
        default=None,
        help="optional output path for main foreground component mask PNG",
    )
    parser.add_argument(
        "--crop-output",
        type=Path,
        default=None,
        help="optional output path for cropped garment PNG",
    )
    parser.add_argument(
        "--crop-mask-output",
        type=Path,
        default=None,
        help="optional output path for cropped garment mask PNG",
    )
    parser.add_argument(
        "--square-output",
        type=Path,
        default=None,
        help="optional output path for white square garment PNG",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=512,
        help="square canvas size in pixels",
    )
    parser.add_argument(
        "--no-canvas-upscale",
        action="store_true",
        help="do not upscale images smaller than the square canvas",
    )
    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=0,
        help="alpha value threshold used to build binary mask",
    )
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=127,
        help="mask value threshold used before cleaning",
    )
    parser.add_argument(
        "--opening-kernel-size",
        type=int,
        default=3,
        help="kernel size used to remove small white noise from mask",
    )
    parser.add_argument(
        "--closing-kernel-size",
        type=int,
        default=5,
        help="kernel size used to close small black holes in mask",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=64,
        help="minimum foreground component area kept in cleaned mask",
    )
    parser.add_argument(
        "--main-component-min-area",
        type=int,
        default=1,
        help="minimum area required for the selected main foreground component",
    )
    parser.add_argument(
        "--crop-margin-ratio",
        type=float,
        default=0.10,
        help="margin ratio added around the cropped garment",
    )
    parser.add_argument(
        "--crop-min-margin-pixels",
        type=int,
        default=0,
        help="minimum margin pixels added around the cropped garment",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"image not found: {args.image}")

    image = load_image_from_path(args.image, mode="RGBA")
    output = image if args.only_normalize else _remove_background(image, args.model)
    output_path = args.output or _default_output_path(args.image, args.only_normalize)
    output.save(output_path)
    mask_path = None
    clean_mask_path = None
    main_mask_path = None
    crop_path = None
    crop_mask_path = None
    square_path = None
    if (
        args.mask_output is not None
        or args.clean_mask_output is not None
        or args.main_mask_output is not None
        or args.crop_output is not None
        or args.crop_mask_output is not None
        or args.square_output is not None
    ):
        mask = _extract_mask(output, args.alpha_threshold)
        mask_path = args.mask_output
        if mask_path is not None:
            mask.save(mask_path)
        if (
            args.clean_mask_output is not None
            or args.main_mask_output is not None
            or args.crop_output is not None
            or args.crop_mask_output is not None
            or args.square_output is not None
        ):
            clean_mask = _clean_mask(
                mask,
                mask_threshold=args.mask_threshold,
                opening_kernel_size=args.opening_kernel_size,
                closing_kernel_size=args.closing_kernel_size,
                min_component_area=args.min_component_area,
            )
            if args.clean_mask_output is not None:
                clean_mask_path = args.clean_mask_output
                clean_mask.save(clean_mask_path)
            if (
                args.main_mask_output is not None
                or args.crop_output is not None
                or args.crop_mask_output is not None
                or args.square_output is not None
            ):
                main_mask = _keep_main_component(
                    clean_mask,
                    mask_threshold=args.mask_threshold,
                    min_component_area=args.main_component_min_area,
                )
                if args.main_mask_output is not None:
                    main_mask_path = args.main_mask_output
                    main_mask.save(main_mask_path)
            if (
                args.crop_output is not None
                or args.crop_mask_output is not None
                or args.square_output is not None
            ):
                crop_result = _crop_garment(
                    output,
                    main_mask,
                    mask_threshold=args.mask_threshold,
                    margin_ratio=args.crop_margin_ratio,
                    min_margin_pixels=args.crop_min_margin_pixels,
                )
                if args.crop_output is not None:
                    crop_path = args.crop_output
                    crop_result.image.save(crop_path)
                if args.crop_mask_output is not None:
                    crop_mask_path = args.crop_mask_output
                    crop_result.mask.save(crop_mask_path)
                if args.square_output is not None:
                    square_path = args.square_output
                    square_image = create_square_garment_image(
                        crop_result.image,
                        crop_result.mask,
                        CanvasConfig(
                            size=args.canvas_size,
                            allow_upscale=not args.no_canvas_upscale,
                        ),
                    )
                    square_image.save(square_path)

    print(f"input:  {args.image.resolve()}")
    print(f"output: {output_path.resolve()}")
    if mask_path is not None:
        print(f"mask:   {mask_path.resolve()}")
    if clean_mask_path is not None:
        print(f"clean:  {clean_mask_path.resolve()}")
    if main_mask_path is not None:
        print(f"main:   {main_mask_path.resolve()}")
    if crop_path is not None:
        print(f"crop:   {crop_path.resolve()}")
    if crop_mask_path is not None:
        print(f"crop-m: {crop_mask_path.resolve()}")
    if square_path is not None:
        print(f"square: {square_path.resolve()}")
    print(f"mode:   {output.mode}")
    print(f"size:   {output.size}")
    if not args.only_normalize:
        print(f"model:  {args.model}")


def _remove_background(image, model_name: str):
    try:
        config = BackgroundRemovalConfig(model_name=model_name)
        return remove_background(image, config)
    except BackgroundRemovalDependencyError as exc:
        raise SystemExit(str(exc)) from exc


def _extract_mask(image, alpha_threshold: int):
    return extract_alpha_mask(
        image,
        AlphaMaskConfig(alpha_threshold=alpha_threshold),
    )


def _clean_mask(
    mask,
    mask_threshold: int,
    opening_kernel_size: int,
    closing_kernel_size: int,
    min_component_area: int,
):
    return clean_binary_mask(
        mask,
        MaskCleaningConfig(
            mask_threshold=mask_threshold,
            opening_kernel_size=opening_kernel_size,
            closing_kernel_size=closing_kernel_size,
            min_component_area=min_component_area,
        ),
    )


def _keep_main_component(mask, mask_threshold: int, min_component_area: int):
    return keep_main_component(
        mask,
        MainComponentConfig(
            mask_threshold=mask_threshold,
            min_component_area=min_component_area,
        ),
    )


def _crop_garment(
    image,
    mask,
    mask_threshold: int,
    margin_ratio: float,
    min_margin_pixels: int,
):
    return crop_garment_with_margin(
        image,
        mask,
        GarmentCropConfig(
            mask_threshold=mask_threshold,
            margin_ratio=margin_ratio,
            min_margin_pixels=min_margin_pixels,
        ),
    )


def _default_output_path(input_path: Path, only_normalize: bool) -> Path:
    suffix = ".normalized.png" if only_normalize else ".no-bg.png"
    return input_path.with_name(f"{input_path.stem}{suffix}")


if __name__ == "__main__":
    main()
