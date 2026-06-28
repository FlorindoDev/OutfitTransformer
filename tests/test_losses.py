import unittest

import torch
from torch.nn import functional as F

from model import BinaryFocalLoss, SetWiseRankingLoss


class BinaryFocalLossTests(unittest.TestCase):
    def test_gamma_zero_without_alpha_matches_binary_cross_entropy(self) -> None:
        logits = torch.tensor([1.2, -0.7, 0.1])
        targets = torch.tensor([1.0, 0.0, 1.0])

        actual = BinaryFocalLoss(alpha=None, gamma=0.0)(logits, targets)
        expected = F.binary_cross_entropy_with_logits(logits, targets)

        torch.testing.assert_close(actual, expected)

    def test_easy_example_has_lower_loss_than_hard_example(self) -> None:
        loss = BinaryFocalLoss(alpha=None, reduction="none")
        targets = torch.ones(2)

        values = loss(torch.tensor([5.0, 0.0]), targets)

        self.assertLess(values[0].item(), values[1].item())


class SetWiseRankingLossTests(unittest.TestCase):
    def test_returns_zero_when_margin_is_satisfied(self) -> None:
        target = torch.tensor([[0.0, 0.0]])
        positive = torch.tensor([[0.0, 0.0]])
        negatives = torch.tensor([[[3.0, 0.0], [0.0, 4.0]]])

        actual = SetWiseRankingLoss(margin=2.0)(target, positive, negatives)

        torch.testing.assert_close(actual, torch.tensor(0.0))

    def test_combines_all_and_hard_negative_terms(self) -> None:
        target = torch.tensor([[0.0]])
        positive = torch.tensor([[1.0]])
        negatives = torch.tensor([[[2.0], [0.5]]])

        actual = SetWiseRankingLoss(margin=2.0)(target, positive, negatives)

        torch.testing.assert_close(actual, torch.tensor(4.25))

    def test_rejects_empty_negative_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one negative"):
            SetWiseRankingLoss()(
                torch.zeros(1, 2),
                torch.zeros(1, 2),
                torch.zeros(1, 0, 2),
            )


if __name__ == "__main__":
    unittest.main()
