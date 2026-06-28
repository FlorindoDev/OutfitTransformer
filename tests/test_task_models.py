import unittest
from types import SimpleNamespace

import torch
from torch import Tensor, nn

from model import CompatibilityPredictor, ComplementaryItemRetriever


class StubOutfitEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            image_embedding_dim=2,
            text_embedding_dim=2,
            item_embedding_dim=4,
        )
        self.text_projection = nn.Linear(1, 2)

    def forward(
        self,
        images: Tensor,
        descriptions: object,
        padding_mask: Tensor,
    ) -> SimpleNamespace:
        outfit_embedding = images.masked_fill(
            padding_mask.unsqueeze(-1),
            0.0,
        ).sum(dim=1)
        return SimpleNamespace(outfit_embedding=outfit_embedding)

    def encode_items(
        self,
        images: Tensor,
        descriptions: object,
        padding_mask: Tensor,
    ) -> Tensor:
        return images

    def encode_flat_items(
        self,
        images: Tensor,
        descriptions: object,
    ) -> Tensor:
        return images

    def encode_text(self, descriptions: list[str]) -> Tensor:
        lengths = torch.tensor(
            [[len(description)] for description in descriptions],
            dtype=self.text_projection.weight.dtype,
        )
        return self.text_projection(lengths)

    def contextualize(
        self,
        embeddings: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        return embeddings.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class CompatibilityPredictorTests(unittest.TestCase):
    def test_returns_score_and_global_embedding(self) -> None:
        model = CompatibilityPredictor(encoder=StubOutfitEncoder())
        images = torch.randn(2, 3, 4)
        mask = torch.tensor(
            [[False, False, False], [False, False, True]]
        )

        output = model(images, [("a", "b", "c"), ("d", "e")], mask)

        self.assertEqual(output.logits.shape, (2,))
        self.assertEqual(output.compatibility_score.shape, (2,))
        self.assertEqual(output.outfit_embedding.shape, (2, 4))
        self.assertTrue(
            (
                (output.compatibility_score >= 0.0)
                & (output.compatibility_score <= 1.0)
            ).all()
        )


class ComplementaryItemRetrieverTests(unittest.TestCase):
    def test_prepends_target_token_and_returns_target_embedding(self) -> None:
        model = ComplementaryItemRetriever(encoder=StubOutfitEncoder())
        images = torch.randn(2, 3, 4)
        mask = torch.tensor(
            [[False, False, False], [False, False, True]]
        )

        output = model(
            images,
            [("a", "b", "c"), ("d", "e")],
            mask,
            ["shoes", "shirt"],
        )

        self.assertEqual(output.target_embedding.shape, (2, 4))
        self.assertEqual(output.target_token.shape, (2, 4))
        self.assertEqual(output.contextual_embeddings.shape, (2, 4, 4))
        self.assertEqual(output.padding_mask.shape, (2, 4))
        self.assertFalse(output.padding_mask[:, 0].any())

    def test_requires_one_target_description_per_outfit(self) -> None:
        model = ComplementaryItemRetriever(encoder=StubOutfitEncoder())

        with self.assertRaisesRegex(ValueError, "one target description"):
            model(
                torch.randn(2, 1, 4),
                [("a",), ("b",)],
                torch.zeros(2, 1, dtype=torch.bool),
                ["shoes"],
            )


if __name__ == "__main__":
    unittest.main()
