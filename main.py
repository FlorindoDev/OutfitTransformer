import argparse

import torch

from data import OutfitBatch, OutfitExample, collate_outfits
from model import OutfitEncoder, OutfitEncoderConfig


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
        ),
        OutfitExample(
            outfit_id="casual-outfit",
            images=torch.rand(2, 3, image_size, image_size),
            descriptions=(
                "black printed t-shirt",
                "light blue denim jeans",
            ),
        ),
    ]
    return collate_outfits(examples)


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
        "--no-pretrained-image",
        action="store_true",
        help="do not download ImageNet weights for this example",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OutfitEncoderConfig(
        text_model_name=args.text_model,
        pretrained_image_encoder=not args.no_pretrained_image,
    )
    batch = build_example_batch().to(args.device)
    model = OutfitEncoder(config).to(args.device).eval()

    with torch.inference_mode():
        output = model(
            images=batch.images,
            descriptions=batch.descriptions,
            padding_mask=batch.padding_mask,
        )

    print(f"images:                {tuple(batch.images.shape)}")
    print(f"padding mask:           {tuple(batch.padding_mask.shape)}")
    print(f"concatenated features:  {tuple(output.item_embeddings.shape)}")
    print(f"transformer output:     {tuple(output.contextual_embeddings.shape)}")
    print(f"global outfit embedding: {tuple(output.outfit_embedding.shape)}")


if __name__ == "__main__":
    main()
