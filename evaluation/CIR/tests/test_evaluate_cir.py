"""Offline integration checks for the CIR command, data selection and report."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from data import DataSplit
from evaluation.CIR.checkpoint import load_cir_checkpoint, restore_cir_model
from evaluation.CIR.config import CIREvaluationConfig
from evaluation.CIR.evaluate_cir import _default_output_path, main, parse_args, run
from evaluation.CIR.tests.fixtures import training_config, write_checkpoint, write_fitb_data
from training.CIR import CIRTrainingModel
from training.CIR.data import (
    RetrievalDataConfig,
    build_retrieval_loader,
    build_retrieval_loaders,
)
from training.CIR.distributed import DistributedContext
from training.CIR.trainer import _evaluate as evaluate_training_validation


class CIREvaluationCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(18)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "cir.pt"

    def _prepare(self, split=DataSplit.TEST, *, categories=False):
        config = training_config(self.root, categories=categories)
        model = CIRTrainingModel(
            config.model_config, config.cir_config,
            feature_mode=config.feature_mode, use_category_embedding=categories,
        )
        write_checkpoint(self.path, config, model)
        write_fitb_data(config, split)
        return config

    def test_full_local_pipeline_for_test_and_validation_with_categories(self):
        for split in (DataSplit.TEST, DataSplit.VALIDATION):
            for categories in (False, True):
                with self.subTest(split=split, categories=categories):
                    self._prepare(split, categories=categories)
                    output = self.root / f"{split.value}.json"
                    config = CIREvaluationConfig(
                        checkpoint=self.path, split=split, output_path=output,
                        batch_size=2, candidate_batch_size=3, device="cpu",
                    )
                    with patch(
                        "data.polyvore.download._download_file",
                        side_effect=AssertionError("offline fixtures must be sufficient"),
                    ):
                        result = run(config, token=False)
                        single = run(replace(config, batch_size=1), token=False)
                    report = json.loads(output.read_text(encoding="utf-8"))
                    self.assertEqual(result.metrics, single.metrics)
                    self.assertEqual(report["metrics"]["examples"], 3)
                    self.assertEqual(report["protocol"], "polyvore_fitb")
                    self.assertEqual(report["split"], split.value)
                    self.assertEqual(report["use_category_embedding"], categories)
                    self.assertEqual(report["tie_policy"], "pessimistic")
                    self.assertNotIn("threshold", report)
                    self.assertFalse(output.with_name(output.name + ".tmp").exists())

    def test_cli_accepts_moved_data_and_embedding_roots(self):
        config = self._prepare()
        payload = torch.load(self.path, weights_only=True)
        payload["run_config"]["dataset"]["embedding_root"] = "missing-embedding-root"
        payload["run_config"]["dataset"]["dataset_root"] = "missing-dataset-root"
        torch.save(payload, self.path)
        output = self.root / "moved.json"
        arguments = [
            "--checkpoint", str(self.path), "--split", "test",
            "--embedding-root", str(config.embedding_root),
            "--dataset-root", str(config.dataset_root),
            "--output", str(output), "--batch-size", "1",
            "--candidate-batch-size", "2", "--device", "cpu", "--no-token",
        ]
        with (
            patch("evaluation.CIR.evaluate_cir.logging.basicConfig"),
            self.assertLogs("evaluation.CIR", level="INFO"),
        ):
            self.assertEqual(main(arguments), 0)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["metrics"]["examples"], 3)
        self.assertNotIn("token", report)

    def test_positive_answer_is_resolved_from_annotation_not_array_position(self):
        config = self._prepare()
        data_config = RetrievalDataConfig.from_training_config(config)
        loader = build_retrieval_loader(data_config, split=DataSplit.TEST, token=False)
        self.assertEqual(len(loader.dataset), 3)
        cache_path = config.embedding_root / config.subset / "test" / "shard.pt"
        embeddings = torch.load(cache_path, weights_only=True)["embeddings"]
        torch.testing.assert_close(loader.dataset[0].positive_item, embeddings[1])

    def test_missing_distractor_embedding_fails_before_report_write(self):
        config = self._prepare()
        cache = config.embedding_root / config.subset / "test"
        shard = torch.load(cache / "shard.pt", weights_only=True)
        shard["item_ids"] = shard["item_ids"][:-1]
        shard["embeddings"] = shard["embeddings"][:-1].clone()
        torch.save(shard, cache / "shard.pt")
        manifest_path = cache / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["count"] = 4
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        output = self.root / "missing.json"
        with self.assertRaisesRegex(ValueError, "missing embedding"):
            run(CIREvaluationConfig(checkpoint=self.path, output_path=output), token=False)
        self.assertFalse(output.exists())

    def test_invalid_runtime_values_are_rejected(self):
        self._prepare()
        config = CIREvaluationConfig(checkpoint=self.path)
        for values in (
            {"split": DataSplit.TRAIN}, {"batch_size": 0},
            {"candidate_batch_size": 0}, {"num_workers": -1},
            {"seed": -1}, {"device": " "}, {"log_every": 0},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(config, **values).validate()
        with self.assertRaises(FileNotFoundError):
            replace(config, checkpoint=self.root / "missing.pt").validate()

    def test_training_validation_metrics_remain_consistent(self):
        config = self._prepare(categories=True)
        loader = build_retrieval_loader(
            RetrievalDataConfig.from_training_config(config),
            split=DataSplit.TEST, token=False,
        )
        model = restore_cir_model(load_cir_checkpoint(self.path))
        reference = evaluate_training_validation(
            model, loader, config, DistributedContext(device=torch.device("cpu")),
        )
        result = run(CIREvaluationConfig(
            checkpoint=self.path, output_path=self.root / "training-comparison.json",
        ), token=False)
        self.assertEqual(reference.examples, result.metrics.examples)
        self.assertEqual(reference.fitb_accuracy, result.metrics.fitb_accuracy)
        self.assertAlmostEqual(reference.mrr, result.metrics.mrr)
        self.assertEqual(reference.recall_at_2, result.metrics.recall_at_2)

    def test_training_still_rejects_single_example_batches(self):
        config = training_config(self.root)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            build_retrieval_loaders(replace(config, batch_size=1), token=False)

    def test_default_output_paths_and_cli_defaults(self):
        self._prepare()
        config, token = parse_args(["--checkpoint", str(self.path)])
        self.assertEqual(config.split, DataSplit.TEST)
        self.assertEqual(config.candidate_batch_size, 512)
        self.assertTrue(token)
        for checkpoint in ("checkpoints/run/best.pt", "checkpoints/run/epochs/cir_epoch_003.pt"):
            path = Path(checkpoint)
            self.assertEqual(
                _default_output_path(path, "polyvore", "nondisjoint", DataSplit.TEST),
                Path("results/cir/polyvore/nondisjoint/run") / f"{path.stem}_test.json",
            )
