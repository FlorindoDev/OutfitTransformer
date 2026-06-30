import unittest
from unittest.mock import patch

import torch

from model.common.text_encoder import TextEncoder


class FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def get_embedding_dimension(self) -> int:
        return 3

    def requires_grad_(self, requires_grad: bool) -> "FakeSentenceTransformer":
        return self

    def eval(self) -> "FakeSentenceTransformer":
        return self

    def encode(
        self,
        descriptions: list[str],
        *,
        convert_to_tensor: bool,
        device: str,
        show_progress_bar: bool,
    ) -> torch.Tensor:
        with torch.inference_mode():
            return torch.ones((len(descriptions), 3), device=device)


class TextEncoderTests(unittest.TestCase):
    def test_projection_can_backpropagate_from_sentence_transformer_features(
        self,
    ) -> None:
        with patch(
            "model.common.text_encoder.SentenceTransformer",
            FakeSentenceTransformer,
        ):
            encoder = TextEncoder(embedding_dim=2, model_name="fake")

        output = encoder(["red shirt", "blue shoes"])
        loss = output.sum()
        loss.backward()

        self.assertEqual(output.shape, (2, 2))
        self.assertIsNotNone(encoder.projection.weight.grad)


if __name__ == "__main__":
    unittest.main()
