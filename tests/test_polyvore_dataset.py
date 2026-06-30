import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from data import (
    PolyvoreCompatibilityDataset,
    collate_compatibility,
)


def image_transform(image: Image.Image) -> torch.Tensor:
    return pil_to_tensor(image).to(torch.float32) / 255.0


class PolyvoreCompatibilityDatasetTests(unittest.TestCase):
    def test_reads_official_compatibility_questions(self) -> None:
        rows = [
            {
                "set_id": "outfit-a",
                "items": [
                    {
                        "index": 1,
                        "item_id": "item-a1",
                        "title": "red shirt",
                        "image": Image.new("RGB", (2, 2), "red"),
                    },
                    {
                        "index": 2,
                        "item_id": "item-a2",
                        "title": "black trousers",
                        "image": Image.new("RGB", (2, 2), "black"),
                    },
                ],
            },
            {
                "set_id": "outfit-b",
                "items": [
                    {
                        "index": 1,
                        "item_id": "item-b1",
                        "title": "blue shoes",
                        "image": Image.new("RGB", (2, 2), "blue"),
                    },
                ],
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            questions = Path(directory) / "compatibility_train.txt"
            questions.write_text(
                "1 outfit-a_1 outfit-a_2\n"
                "0 outfit-a_1 outfit-b_1\n",
                encoding="utf-8",
            )
            dataset = PolyvoreCompatibilityDataset(
                outfit_rows=rows,
                compatibility_path=questions,
                image_transform=image_transform,
            )

            positive = dataset[0]
            negative = dataset[1]
            batch = collate_compatibility([positive, negative])

        self.assertEqual(len(dataset), 2)
        self.assertEqual(positive.label, 1.0)
        self.assertEqual(negative.label, 0.0)
        self.assertEqual(positive.descriptions, ("red shirt", "black trousers"))
        self.assertEqual(negative.descriptions, ("red shirt", "blue shoes"))
        self.assertEqual(batch.images.shape, (2, 2, 3, 2, 2))
        torch.testing.assert_close(batch.labels, torch.tensor([1.0, 0.0]))
        self.assertFalse(batch.padding_mask.any())

    def test_supports_flat_hugging_face_item_rows(self) -> None:
        rows = [
            {
                "set_id": "set-a",
                "index": 1,
                "item_id": "item-a",
                "title": "green top",
                "image": Image.new("RGB", (2, 2), "green"),
            },
            {
                "set_id": "set-b",
                "index": 2,
                "item_id": "item-b",
                "title": "brown shoes",
                "image": Image.new("RGB", (2, 2), "brown"),
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            questions = Path(directory) / "compatibility_train.txt"
            questions.write_text(
                "0 set-a_1 set-b_2\n",
                encoding="utf-8",
            )
            dataset = PolyvoreCompatibilityDataset(
                outfit_rows=rows,
                compatibility_path=questions,
                image_transform=image_transform,
            )

            example = dataset[0]

        self.assertEqual(example.label, 0.0)
        self.assertEqual(example.descriptions, ("green top", "brown shoes"))
        self.assertEqual(example.images.shape, (2, 3, 2, 2))

    def test_supports_hugging_face_image_rows_with_outfit_mapping(self) -> None:
        rows = [
            {
                "item_id": "item-a",
                "image": Image.new("RGB", (2, 2), "green"),
            },
            {
                "item_id": "item-b",
                "image": Image.new("RGB", (2, 2), "brown"),
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            questions = Path(directory) / "compatibility_train.txt"
            questions.write_text("1 set-a_1 set-a_2\n", encoding="utf-8")
            outfit_mapping = Path(directory) / "train.json"
            outfit_mapping.write_text(
                """[
                  {
                    "set_id": "set-a",
                    "items": [
                      {"index": 1, "item_id": "item-a"},
                      {"index": 2, "item_id": "item-b"}
                    ]
                  }
                ]""",
                encoding="utf-8",
            )

            dataset = PolyvoreCompatibilityDataset(
                outfit_rows=rows,
                compatibility_path=questions,
                image_transform=image_transform,
                outfit_mapping_path=outfit_mapping,
            )
            example = dataset[0]

        self.assertEqual(example.label, 1.0)
        self.assertEqual(example.descriptions, ("fashion item", "fashion item"))
        self.assertEqual(example.images.shape, (2, 3, 2, 2))

    def test_rejects_malformed_question(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            questions = Path(directory) / "compatibility_train.txt"
            questions.write_text("1 one-item-only\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "at least two items"):
                PolyvoreCompatibilityDataset(
                    outfit_rows=[],
                    compatibility_path=questions,
                )


if __name__ == "__main__":
    unittest.main()
