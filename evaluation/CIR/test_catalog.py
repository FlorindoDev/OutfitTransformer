"""Regression checks for full-catalog recall on a small Polyvore split."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from data import DataSplit
from data.polyvore import POLYVORE_DATASET_ID
from model import TransformerConfig
from training.CIR.data import RetrievalDataConfig, build_retrieval_loader
from training.common.features import FeatureMode

from .catalog_data import build_catalog_loaders
from .evaluator import evaluate, evaluate_catalog


class _TestModel(nn.Module):
    def embed_queries(
        self, outfits: tuple[torch.Tensor, ...], target_categories: tuple[str, ...],
    ) -> torch.Tensor:
        return torch.stack([outfit.mean(dim=0) for outfit in outfits])

    def embed_items(self, outfits: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.cat(outfits)


class CatalogEvaluationTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        dataset_root = root / "dataset"
        self.embedding_root = root / "embeddings"
        split_dir = dataset_root / "nondisjoint"
        split_dir.mkdir(parents=True)
        self.item_vectors: dict[str, float] = {
            "query_left": -8,
            "query_right": 8,
            "positive": 2,
            "negative_1": 3,
            "negative_2": 4,
            "negative_3": 5,
            "unlisted_closer": 1,
            "unlisted_tied": -2,
        }
        outfits = [
            ("query", ("query_left", "query_right", "positive")),
            ("answers", ("negative_1", "negative_2", "negative_3")),
            ("other", ("unlisted_closer", "unlisted_tied")),
        ]
        (split_dir / "test.json").write_text(json.dumps([
            {
                "set_id": set_id,
                "items": [
                    {"item_id": item_id, "index": index}
                    for index, item_id in enumerate(item_ids, start=1)
                ],
            }
            for set_id, item_ids in outfits
        ]), encoding="utf-8")
        (split_dir / "fill_in_blank_test.json").write_text(json.dumps([{
            "question": ["query_1", "query_2"],
            "blank_position": 3,
            "answers": ["answers_1", "answers_2", "query_3", "answers_3"],
        }]), encoding="utf-8")
        self.config = RetrievalDataConfig(
            dataset_name="polyvore",
            subset="nondisjoint",
            feature_mode=FeatureMode.PRECOMPUTED,
            embedding_root=self.embedding_root,
            dataset_root=dataset_root,
            cache_dir=None,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            seed=42,
            use_category_embedding=False,
            distributed=False,
            model_config=TransformerConfig(modality_embedding_dim=1, attention_heads=1),
        )

    def test_recall_accounts_for_unlisted_items_and_ties(self):
        self._write_embeddings()
        model = _TestModel()
        device = torch.device("cpu")
        fitb = build_retrieval_loader(self.config, split=DataSplit.TEST, token=False)
        fitb_errors = []
        self.assertEqual(evaluate(model, fitb, device, errors=fitb_errors).recall_at_2, 1.0)
        self.assertEqual(fitb_errors, [])

        loaders = build_catalog_loaders(
            self.config, split=DataSplit.TEST, candidate_batch_size=3, token=False,
        )
        metrics = evaluate_catalog(model, loaders, device)
        self.assertEqual(metrics.as_dict(), {
            "examples": 1,
            "catalog_items": 8,
            "mrr": 1 / 3,
            "recall_at_1": 0.0,
            "recall_at_2": 0.0,
            "recall_at_10": 1.0,
            "recall_at_30": 1.0,
            "recall_at_50": 1.0,
            "recall_at_100": 1.0,
        })

    def test_reports_failed_metrics_and_neighbors_of_the_chosen_item(self):
        self.item_vectors.update({
            "negative_1": -2,
            "negative_3": -4,
            "unlisted_closer": -2.5,
            "unlisted_tied": 2,
        })
        self._write_embeddings()
        model = _TestModel()
        device = torch.device("cpu")
        fitb = build_retrieval_loader(self.config, split=DataSplit.TEST, token=False)
        errors = []
        metrics = evaluate(model, fitb, device, errors=errors, error_neighbors=1)
        self.assertEqual(metrics, evaluate(model, fitb, device))
        expected_query = {
            "example_id": "retrieval:0",
            "partial_item_ids": ("query_left", "query_right"),
            "target_category": "unknown",
            "positive_item_id": "positive",
            "predicted_item_id": "negative_1",
            "predicted_distance": 2.0,
        }
        self.assertEqual(errors, [{
            **expected_query,
            "positive_rank": 2,
            "predicted_distractor_index": 0,
            "failed_metrics": ["mrr", "fitb_accuracy"],
            "neighbors": [{"item_id": "negative_3", "distance": 2.0}],
        }])

        loaders = build_catalog_loaders(
            self.config, split=DataSplit.TEST, candidate_batch_size=3, token=False,
        )
        catalog_errors = []
        metrics = evaluate_catalog(model, loaders, device, errors=catalog_errors, error_neighbors=1)
        self.assertEqual(metrics, evaluate_catalog(model, loaders, device))
        self.assertEqual(catalog_errors, [{
            **expected_query,
            "positive_rank": 3,
            "failed_metrics": ["mrr", "recall_at_1", "recall_at_2"],
            "neighbors": [{"item_id": "unlisted_closer", "distance": 0.5}],
        }])

    def test_rejects_cache_missing_an_item_outside_fitb(self):
        del self.item_vectors["unlisted_closer"]
        self._write_embeddings()
        with self.assertRaisesRegex(ValueError, "incomplete test catalog"):
            build_catalog_loaders(
                self.config, split=DataSplit.TEST, candidate_batch_size=3, token=False,
            )

    def _write_embeddings(self):
        split_dir = self.embedding_root / "nondisjoint" / "test"
        split_dir.mkdir(parents=True)
        torch.save({
            "item_ids": list(self.item_vectors),
            "embeddings": torch.tensor(
                [[value, 0.0] for value in self.item_vectors.values()],
            ),
        }, split_dir / "items.pt")
        (split_dir / "manifest.json").write_text(json.dumps({
            "schema_version": 2,
            "dataset": POLYVORE_DATASET_ID,
            "subset": "nondisjoint",
            "split": "test",
            "count": len(self.item_vectors),
            "embedding_dim": 2,
            "shards": [{"file": "items.pt"}],
        }), encoding="utf-8")
