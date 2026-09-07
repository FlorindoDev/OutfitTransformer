"""Checkpoint restoration preserves current CIR predictions and rejects old weights."""

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from evaluation.CIR.checkpoint import load_cir_checkpoint, restore_cir_model
from evaluation.CIR.tests.fixtures import training_config, write_checkpoint
from training.CIR import CIRTrainingModel


class CIRCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(12)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "cir.pt"
        self.outfits = (torch.randn(2, 8), torch.randn(3, 8))
        self.items = (torch.randn(1, 8), torch.randn(1, 8))
        self.categories = ("tops", "shoes")

    def _source(self, *, categories=False):
        config = training_config(self.root, categories=categories)
        model = CIRTrainingModel(
            config.model_config, config.cir_config,
            feature_mode=config.feature_mode,
            use_category_embedding=categories,
        ).eval()
        write_checkpoint(self.path, config, model)
        return model, load_cir_checkpoint(self.path)

    def test_current_query_and_item_predictions_with_and_without_categories(self):
        for categories in (False, True):
            with self.subTest(categories=categories):
                source, checkpoint = self._source(categories=categories)
                restored = restore_cir_model(checkpoint).eval()
                with torch.inference_mode():
                    expected = source(self.outfits, self.items, self.categories)
                    actual = restored(self.outfits, self.items, self.categories)
                for actual_values, expected_values in zip(actual, expected):
                    torch.testing.assert_close(actual_values, expected_values, rtol=0, atol=0)
                self.assertIsNone(restored.cir.encoder.norm)

    def test_incompatible_weights_are_rejected(self):
        _, checkpoint = self._source()
        for problem in ("missing", "unexpected", "shape"):
            with self.subTest(problem=problem):
                state = dict(checkpoint.state_dict)
                if problem == "missing":
                    del state["cir.encoder.layers.0.norm1.weight"]
                elif problem == "unexpected":
                    state["cir.encoder.extra.weight"] = torch.ones(8)
                else:
                    state["cir.embed_emb"] = torch.ones(1)
                with self.assertRaises(RuntimeError):
                    restore_cir_model(replace(checkpoint, state_dict=state))

    def test_legacy_norm_is_rejected_including_partial_weights(self):
        for categories in (False, True):
            source, checkpoint = self._source(categories=categories)
            for suffixes in (("weight", "bias"), ("weight",), ("bias",)):
                with self.subTest(categories=categories, suffixes=suffixes):
                    state = dict(checkpoint.state_dict)
                    state.update({
                        f"cir.encoder.norm.{suffix}": torch.ones(source.config.model_dim)
                        for suffix in suffixes
                    })
                    with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
                        restore_cir_model(replace(checkpoint, state_dict=state))

    def test_invalid_checkpoint_metadata_is_rejected(self):
        self._source()
        original = torch.load(self.path, weights_only=True)
        for problem in ("schema", "epoch", "dataset", "category", "embedding_root"):
            with self.subTest(problem=problem):
                payload = copy.deepcopy(original)
                if problem == "schema":
                    payload["checkpoint_schema_version"] = 2
                elif problem == "epoch":
                    payload["epoch"] = True
                elif problem == "dataset":
                    payload["run_config"]["dataset"]["id"] = "wrong/dataset"
                elif problem == "category":
                    payload["run_config"]["model"]["use_category_embedding"] = "false"
                else:
                    payload["run_config"]["dataset"]["embedding_root"] = None
                torch.save(payload, self.path)
                with self.assertRaises((TypeError, ValueError)):
                    load_cir_checkpoint(self.path)
