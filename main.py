import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image

from data import (
    OutfitBatch,
    OutfitExample,
    build_image_transform,
    collate_outfits,
)
from model import (
    CompatibilityPredictor,
    OutfitEncoderConfig,
    load_cp_checkpoint,
)


def build_example_batch(image_size: int = 224) -> OutfitBatch:
    examples = [
        OutfitExample(
            outfit_id="formal-outfit",
            images=torch.rand(3, 3, image_size, image_size),
            descriptions=(
                "white cotton shirt",
                "navy tailored trousers",
                "brown leather shoes",
            ),
        )
    ]
    return collate_outfits(examples)


def build_image_batch(
    image_paths: Sequence[Path],
    descriptions: Sequence[str] | None = None,
    image_size: int = 224,
) -> OutfitBatch:
    if not image_paths:
        raise ValueError("at least one image path is required")

    resolved_descriptions = _resolve_descriptions(image_paths, descriptions)
    transform = build_image_transform(image_size)
    images = []
    for path in image_paths:
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path}")
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")))

    example = OutfitExample(
        outfit_id="custom-outfit",
        images=torch.stack(images),
        descriptions=resolved_descriptions,
    )
    return collate_outfits([example])


def _resolve_descriptions(
    image_paths: Sequence[Path],
    descriptions: Sequence[str] | None,
) -> tuple[str, ...]:
    if descriptions is None:
        return tuple(
            path.stem.replace("_", " ").replace("-", " ")
            for path in image_paths
        )
    if len(descriptions) != len(image_paths):
        raise ValueError(
            "--descriptions must contain exactly one value per image"
        )
    return tuple(descriptions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Outfit input encoder example")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--text-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceBERT model name or local directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="CP checkpoint produced by training.cp.train_cp",
    )
    parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        default=None,
        help="one or more image paths forming the outfit",
    )
    parser.add_argument(
        "--descriptions",
        nargs="+",
        default=None,
        help="one text description per image; defaults to file names",
    )
    parser.add_argument(
        "--no-pretrained-image",
        action="store_true",
        help="do not download ImageNet weights for this example",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OutfitEncoderConfig(
        text_model_name=args.text_model,
        pretrained_image_encoder=(
            not args.no_pretrained_image and args.checkpoint is None
        ),
    )
    batch = (
        build_example_batch()
        if args.images is None
        else build_image_batch(args.images, args.descriptions)
    ).to(args.device)
    model = CompatibilityPredictor(config=config).to(args.device)
    if args.checkpoint is not None:
        load_cp_checkpoint(args.checkpoint, model, args.device)
        print(f"checkpoint:            {args.checkpoint.resolve()}")
    model.eval()

    with torch.inference_mode():
        output = model(
            images=batch.images,
            descriptions=batch.descriptions,
            padding_mask=batch.padding_mask,
        )

    print(f"images:                {tuple(batch.images.shape)}")
    print(f"padding mask:           {tuple(batch.padding_mask.shape)}")
    print(f"global outfit embedding: {tuple(output.outfit_embedding.shape)}")
    print(f"compatibility score:    {output.compatibility_score.tolist()}")
if __name__ == "__main__":
    main()
