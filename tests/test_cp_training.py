import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data import CompatibilityExample, collate_compatibility
from model import BinaryFocalLoss
from training import train_cp


class TinyCompatibilityModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(1, 1)

    def forward(
        self,
        images: Tensor,
        descriptions: object,
        padding_mask: Tensor,
    ) -> SimpleNamespace:
        valid = (~padding_mask).to(images.dtype)
        item_values = images.mean(dim=(2, 3, 4))
        outfit_values = (
            (item_values * valid).sum(dim=1) / valid.sum(dim=1)
        ).unsqueeze(-1)
        return SimpleNamespace(logits=self.classifier(outfit_values).squeeze(-1))


class CPTrainingTests(unittest.TestCase):
    def test_training_updates_weights_and_saves_best_checkpoint(self) -> None:
        examples = [
            CompatibilityExample(
                outfit_id="compatible",
                images=torch.ones(2, 3, 2, 2),
                descriptions=("a", "b"),
                label=1.0,
            ),
            CompatibilityExample(
                outfit_id="incompatible",
                images=torch.zeros(2, 3, 2, 2),
                descriptions=("c", "d"),
                label=0.0,
            ),
        ]
        loader = DataLoader(
            examples,
            batch_size=2,
            collate_fn=collate_compatibility,
        )
        model = TinyCompatibilityModel()
        initial_weight = model.classifier.weight.detach().clone()
        optimizer = Adam(model.parameters(), lr=0.1)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "cp.pt"
            epoch_checkpoint_dir = Path(directory) / "epochs"
            history = train_cp(
                model=model,
                train_batches=loader,
                optimizer=optimizer,
                criterion=BinaryFocalLoss(alpha=None),
                epochs=2,
                device="cpu",
                validation_batches=loader,
                checkpoint_path=checkpoint,
                epoch_checkpoint_dir=epoch_checkpoint_dir,
            )
            saved = torch.load(checkpoint, weights_only=True)
            first_epoch = torch.load(
                epoch_checkpoint_dir / "cp_epoch_001.pt",
                weights_only=True,
            )
            second_epoch = torch.load(
                epoch_checkpoint_dir / "cp_epoch_002.pt",
                weights_only=True,
            )

        self.assertEqual(len(history.train), 2)
        self.assertEqual(len(history.validation), 2)
        self.assertEqual(history.train[0].examples, 2)
        self.assertFalse(
            torch.equal(initial_weight, model.classifier.weight.detach())
        )
        self.assertIn("model_state_dict", saved)
        self.assertIn("optimizer_state_dict", saved)
        self.assertEqual(first_epoch["epoch"], 1)
        self.assertEqual(second_epoch["epoch"], 2)
        self.assertIn("train_metrics", second_epoch)
        self.assertIn("validation_metrics", second_epoch)


if __name__ == "__main__":
    unittest.main()
