import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from data import CompatibilityExample, collate_compatibility
from model import BinaryFocalLoss
from training import train_cp

train_cp_cli = import_module("train_cp")


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
    def _loader(self) -> DataLoader:
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
        return DataLoader(
            examples,
            batch_size=2,
            collate_fn=collate_compatibility,
        )

    def test_training_updates_weights_and_saves_best_checkpoint(self) -> None:
        loader = self._loader()
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

    def test_resume_starts_from_next_epoch(self) -> None:
        loader = self._loader()
        model = TinyCompatibilityModel()
        optimizer = Adam(model.parameters(), lr=0.1)
        seen_epochs: list[int] = []

        with tempfile.TemporaryDirectory() as directory:
            epoch_checkpoint_dir = Path(directory) / "epochs"
            history = train_cp(
                model=model,
                train_batches=loader,
                optimizer=optimizer,
                criterion=BinaryFocalLoss(alpha=None),
                epochs=4,
                device="cpu",
                validation_batches=loader,
                epoch_checkpoint_dir=epoch_checkpoint_dir,
                start_epoch=3,
                initial_best_loss=0.0,
                on_epoch_end=lambda epoch, *_: seen_epochs.append(epoch),
            )
            third_epoch = torch.load(
                epoch_checkpoint_dir / "cp_epoch_003.pt",
                weights_only=True,
            )
            fourth_epoch = torch.load(
                epoch_checkpoint_dir / "cp_epoch_004.pt",
                weights_only=True,
            )

        self.assertEqual(seen_epochs, [3, 4])
        self.assertEqual(len(history.train), 2)
        self.assertEqual(len(history.validation), 2)
        self.assertEqual(third_epoch["epoch"], 3)
        self.assertEqual(fourth_epoch["epoch"], 4)

    def test_cli_loads_resume_checkpoint_state(self) -> None:
        loader = self._loader()
        model = TinyCompatibilityModel()
        optimizer = Adam(model.parameters(), lr=0.1)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "cp.pt"
            train_cp(
                model=model,
                train_batches=loader,
                optimizer=optimizer,
                criterion=BinaryFocalLoss(alpha=None),
                epochs=1,
                device="cpu",
                validation_batches=loader,
                scheduler=scheduler,
                checkpoint_path=checkpoint,
            )

            restored_model = TinyCompatibilityModel()
            restored_optimizer = Adam(restored_model.parameters(), lr=0.1)
            restored_scheduler = StepLR(
                restored_optimizer,
                step_size=1,
                gamma=0.5,
            )
            epoch, monitored_loss = train_cp_cli._load_resume_checkpoint(
                path=checkpoint,
                model=restored_model,
                optimizer=restored_optimizer,
                scheduler=restored_scheduler,
                device="cpu",
            )

        self.assertEqual(epoch, 1)
        self.assertIsInstance(monitored_loss, float)
        torch.testing.assert_close(
            restored_model.classifier.weight,
            model.classifier.weight,
        )
        self.assertEqual(
            restored_scheduler.state_dict(),
            scheduler.state_dict(),
        )


if __name__ == "__main__":
    unittest.main()
