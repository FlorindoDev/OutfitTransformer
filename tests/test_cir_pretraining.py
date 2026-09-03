"""Test CP-to-CIR checkpoint initialization."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from training.CIR.config import CIRTrainingConfig
from training.CIR.pretraining import load_cp_pretrained_weights
from training.CIR.train_cir import parse_args


class _TaskEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(2))


class _CIRBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_emb = nn.Parameter(torch.zeros(2))
        self.task_embedding = _TaskEmbedding()
        self.encoder = nn.Linear(2, 2)
        self.head = nn.Linear(2, 2, bias=False)


class _CIRModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.common = nn.Linear(2, 2)
        self.cir = _CIRBranch()


class CPPretrainingTests(unittest.TestCase):
    def test_transfers_shared_weights_and_keeps_cir_specific_weights(self) -> None:
        model = _CIRModel()
        original_embed = model.cir.embed_emb.detach().clone()
        original_encoder_weight = model.cir.encoder.weight.detach().clone()
        original_encoder_bias = model.cir.encoder.bias.detach().clone()
        original_head = model.cir.head.weight.detach().clone()
        checkpoint = {
            "model_state_dict": {
                "common.weight": torch.full((2, 2), 1.0),
                "common.bias": torch.full((2,), 2.0),
                "cp.encoder.weight": torch.full((2, 2), 3.0),
                "cp.encoder.bias": torch.full((2,), 4.0),
                "cp.task_embedding.embedding": torch.full((2,), 5.0),
                "cp.predict_emb": torch.full((2,), 6.0),
                "cp.head.classifier.weight": torch.full((1, 2), 7.0),
            }
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cp.pt"
            torch.save(checkpoint, path)
            report = load_cp_pretrained_weights(model, path)

        self.assertEqual(report.loaded_tensor_count, 3)
        self.assertEqual(
            set(report.loaded_keys),
            {
                "common.weight",
                "common.bias",
                "cir.task_embedding.embedding",
            },
        )
        state_dict = checkpoint["model_state_dict"]
        torch.testing.assert_close(
            model.common.weight,
            state_dict["common.weight"],
        )
        torch.testing.assert_close(
            model.common.bias,
            state_dict["common.bias"],
        )
        torch.testing.assert_close(
            model.cir.task_embedding.embedding,
            state_dict["cp.task_embedding.embedding"],
        )
        torch.testing.assert_close(
            model.cir.encoder.weight,
            original_encoder_weight,
        )
        torch.testing.assert_close(
            model.cir.encoder.bias,
            original_encoder_bias,
        )
        torch.testing.assert_close(model.cir.embed_emb, original_embed)
        torch.testing.assert_close(model.cir.head.weight, original_head)

    def test_rejects_checkpoint_missing_common_weights(self) -> None:
        model = _CIRModel()
        checkpoint = {
            "model_state_dict": {
                "common.weight": torch.ones(2, 2),
                "cp.encoder.weight": torch.ones(2, 2),
                "cp.task_embedding.embedding": torch.ones(2),
            }
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cp.pt"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "missing weights"):
                load_cp_pretrained_weights(model, path)

    def test_pretrained_cp_and_resume_are_mutually_exclusive(self) -> None:
        config = CIRTrainingConfig(
            pretrained_cp=Path("cp.pt"),
            resume=Path("cir.pt"),
        )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            config.validate()

    def test_cli_parses_pretrained_cp(self) -> None:
        config, _ = parse_args(
            ["--precomputed", "--pretrained-cp", "cp.pt"]
        )

        self.assertEqual(config.pretrained_cp, Path("cp.pt"))
        self.assertIsNone(config.resume)


if __name__ == "__main__":
    unittest.main()
