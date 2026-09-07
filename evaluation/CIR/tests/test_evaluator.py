"""Observable ranking, candidate batching and inference-mode contracts."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from evaluation.CIR.evaluator import evaluate
from model import OutfitItem


class VectorModel(nn.Module):
    """Use supplied vectors directly to make expected ranks independent of CIR."""

    def __init__(self):
        super().__init__()
        self.candidate_calls = []

    def embed_queries(self, outfits, categories):
        if self.training or torch.is_grad_enabled():
            raise AssertionError("evaluation must disable training and gradients")
        return torch.stack([outfit[0] for outfit in outfits])

    def embed_items(self, outfits):
        self.candidate_calls.append(len(outfits))
        return torch.stack([
            outfit[0].embedding if isinstance(outfit[0], OutfitItem) else outfit[0]
            for outfit in outfits
        ])


def make_batch(distances, *, raw=False):
    def item(value):
        vector = torch.tensor([float(value)])
        return OutfitItem(embedding=vector) if raw else vector

    return SimpleNamespace(
        partial_outfits=tuple(torch.zeros(1, 1) for _ in distances),
        positive_items=tuple(item(row[0]) for row in distances),
        negative_items=tuple(tuple(item(value) for value in row[1:]) for row in distances),
        target_categories=tuple("tops" for _ in distances),
    )


class CIREvaluatorTests(unittest.TestCase):
    def test_metrics_include_incomplete_batches_and_candidate_chunks(self):
        rows = [[0.2, 0.4, 0.8, 1.0], [0.7, 0.4, 1.1, 1.8], [1.5, 0.2, 0.4, 1.0]]
        for raw in (False, True):
            for chunk_size in (1, 3, 20):
                with self.subTest(raw=raw, candidate_batch_size=chunk_size):
                    model = VectorModel()
                    metrics = evaluate(
                        model, [make_batch(rows[:2], raw=raw), make_batch(rows[2:], raw=raw)],
                        torch.device("cpu"), candidate_batch_size=chunk_size,
                    )
                    self.assertEqual(metrics.examples, 3)
                    self.assertAlmostEqual(metrics.fitb_accuracy, 1 / 3)
                    self.assertAlmostEqual(metrics.mrr, 7 / 12)
                    self.assertAlmostEqual(metrics.recall_at_2, 2 / 3)
                    self.assertLessEqual(max(model.candidate_calls), chunk_size)

    def test_exact_ties_rank_positive_last(self):
        metrics = evaluate(
            VectorModel(), [make_batch([[1, 1, 1, 1]])], torch.device("cpu"),
        )
        self.assertEqual(metrics.fitb_accuracy, 0)
        self.assertEqual(metrics.mrr, 0.25)
        self.assertEqual(metrics.recall_at_2, 0)

    def test_variable_candidate_counts_preserve_groups(self):
        metrics = evaluate(
            VectorModel(), [make_batch([[2, 1], [0, 1, 2, 3, 4]])], torch.device("cpu"),
            candidate_batch_size=3,
        )
        self.assertEqual(metrics.fitb_accuracy, 0.5)
        self.assertEqual(metrics.mrr, 0.75)
        self.assertEqual(metrics.recall_at_2, 1)

    def test_invalid_candidates_and_empty_loaders_are_rejected(self):
        for rows in ([], [[0]], [[float("nan"), 1]], [[0, float("inf")]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                evaluate(VectorModel(), [make_batch(rows)], torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluate(VectorModel(), [], torch.device("cpu"))

    def test_mismatched_query_count_is_rejected(self):
        batch = make_batch([[0, 1], [1, 0]])
        batch.partial_outfits = batch.partial_outfits[:1]
        with self.assertRaisesRegex(ValueError, "shape"):
            evaluate(VectorModel(), [batch], torch.device("cpu"))
