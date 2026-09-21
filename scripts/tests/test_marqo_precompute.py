"""Local precompute and CP/CIR regression tests without model downloads."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image
from torch import nn

from data import ItemBatch, build_marqo_fashion_siglip_transform
from evaluation.CIR.checkpoint import load_cir_checkpoint, restore_cir_model
from evaluation.CP.checkpoint import load_cp_checkpoint, restore_cp_model
from model import OutfitItem
from scripts import precompute_embeddings as precompute
from training.CIR import CIRTrainingConfig, CIRTrainingModel
from training.CIR.pretraining import load_cp_pretrained_weights
from training.CP import CPTrainingConfig, CPTrainingModel
from training.common import EmbeddingCache
from training.common.features import FeatureMode


class FakeMarqoBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(768))
        self.model = SimpleNamespace(
            visual=SimpleNamespace(),
            text=SimpleNamespace(output_dim=768, context_length=64),
        )
        self.config = SimpleNamespace(_commit_hash="test-marqo-revision")

    def get_image_features(self, images, *, normalize):
        return self.weight.expand(images.size(0), -1) * 2

    def get_text_features(self, input_ids, *, normalize):
        if input_ids.size(1) != 64:
            raise ValueError("Marqo requires fixed-length text tokens")
        return self.weight.expand(input_ids.size(0), -1) * 3


def process_inputs(*, text=None, images=None, **kwargs):
    if images is not None:
        return {"pixel_values": torch.ones(1, 3, 224, 224)}
    return {"input_ids": torch.ones(len(text), kwargs["max_length"], dtype=torch.long)}


class MarqoPrecomputeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = precompute.parse_args([
            "--marqo-fashion-siglip", "--subset", "nondisjoint",
            "--output-dir", str(self.root), "--no-token",
        ])
        self.backbone = FakeMarqoBackbone()
        self.processor = Mock(side_effect=process_inputs)
        self.model_loader = self.enterContext(patch(
            "transformers.AutoModel.from_pretrained", return_value=self.backbone,
        ))
        self.enterContext(patch(
            "transformers.AutoProcessor.from_pretrained", return_value=self.processor,
        ))

    def test_backend_selection_preserves_fashionclip_and_openrouter(self):
        self.assertEqual(self.config.model_name, "Marqo/marqo-fashionSigLIP")
        self.assertEqual(self.config.target_dir, self.root / "Marqo-marqo-fashionSigLIP/nondisjoint/train")
        explicit = precompute.parse_args(["--model-name", "Marqo/marqo-fashionSigLIP"])
        self.assertTrue(explicit.uses_marqo_fashion_siglip)
        default = precompute.parse_args([])
        self.assertEqual(default.model_name, "patrickjohncyh/fashion-clip")
        self.assertFalse(default.uses_marqo_fashion_siglip)
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            remote = precompute.parse_args(["--openrouter"])
        self.assertEqual(remote.model_name, "google/gemini-embedding-2")
        self.assertFalse(remote.uses_marqo_fashion_siglip)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            precompute.parse_args(["--marqo-fashion-siglip", "--openrouter"])
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            replace(self.config, use_openrouter=True).validate()

    def test_shared_frozen_backbone_and_official_preprocessing(self):
        visual, text = precompute._build_encoders(self.config, torch.device("cpu"))
        self.model_loader.assert_called_once_with(
            "Marqo/marqo-fashionSigLIP", trust_remote_code=True, low_cpu_mem_usage=False,
        )
        self.assertIs(visual.backbone, text.backbone)
        visual.train()
        text.train()
        self.assertFalse(self.backbone.training)
        self.assertFalse(self.backbone.weight.requires_grad)
        self.assertEqual(visual.output_dim, 768)
        self.assertEqual(text.output_dim, 768)
        descriptions = ["A RED t-shirt!", "long description " * 100]
        self.assertEqual(text(descriptions).shape, (2, 768))
        self.processor.assert_called_once_with(
            text=descriptions, padding="max_length", truncation=True,
            max_length=64, return_tensors="pt",
        )
        transform = build_marqo_fashion_siglip_transform()
        self.assertEqual(transform(Image.new("RGB", (32, 16))).shape, (3, 224, 224))
        for images in (torch.ones(3, 224, 224), torch.ones(1, 1, 224, 224), torch.empty(0, 3, 224, 224)):
            with self.subTest(shape=images.shape), self.assertRaises(ValueError):
                visual(images)
        for descriptions in ([], [" "], [None]):
            with self.subTest(descriptions=descriptions), self.assertRaises(ValueError):
                text(descriptions)

    def test_precompute_limit_shards_metadata_and_cache_roundtrip(self):
        batch = ItemBatch(
            item_ids=("1", "2", "3"), categories=("tops",) * 3,
            model_items=tuple(
                OutfitItem(image=torch.ones(3, 224, 224), text="red shirt")
                for _ in range(3)
            ),
        )
        source = Mock(wraps=precompute.get_dataset_source("polyvore"))
        source.descriptor = precompute.get_dataset_source("polyvore").descriptor
        source.item_dataset.return_value = []
        config = replace(self.config, limit=2, shard_size=1, output_dtype="float16")
        with patch.object(precompute, "get_dataset_source", return_value=source), patch.object(
            precompute, "create_item_loader", return_value=[batch],
        ):
            manifest_path = precompute.run(config)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["embedding_dim"], 1536)
        self.assertEqual(manifest["count"], 2)
        self.assertEqual(len(manifest["shards"]), 2)
        self.assertEqual(manifest["encoder"]["modality_dim"], 768)
        self.assertEqual(manifest["encoder"]["visual_commit"], "test-marqo-revision")
        self.assertEqual(manifest["encoder"]["provider"], "huggingface")
        self.assertEqual(manifest["encoder"]["normalization"], "l2_per_modality")
        cache = EmbeddingCache(config.target_dir)
        self.assertEqual(set(cache), {"1", "2"})
        self.assertEqual(cache["1"].dtype, torch.float16)
        for part in cache["1"].float().split(768):
            self.assertAlmostEqual(float(part.norm()), 1.0, places=3)
        with self.assertRaisesRegex(FileExistsError, "not empty"):
            precompute.run(config)


class PrecomputedTrainingDimensionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        directory = self.root / "nondisjoint/train"
        directory.mkdir(parents=True)
        self.manifest_path = directory / "manifest.json"

    def write_dimension(self, dimension):
        self.manifest_path.write_text(json.dumps({
            "schema_version": 2, "embedding_dim": dimension,
        }), encoding="utf-8")

    def test_native_dimensions_and_serialized_checkpoint_architecture(self):
        for dimension in (1024, 1536):
            self.write_dimension(dimension)
            for config_class in (CPTrainingConfig, CIRTrainingConfig):
                with self.subTest(dimension=dimension, task=config_class.__name__):
                    config = config_class(feature_mode=FeatureMode.PRECOMPUTED, embedding_root=self.root)
                    self.assertEqual(config.model_config.model_dim, dimension)
                    self.assertEqual(config.model_config.modality_embedding_dim, dimension // 2)
                    config.validate()
                    values = config.as_dict()["model"]
                    if config_class is CIRTrainingConfig:
                        values = values["transformer"]
                    self.assertEqual(values["modality_embedding_dim"], dimension // 2)

    def test_invalid_or_missing_dimensions_fail_early(self):
        for dimension in (0, -2, True, "1536", 1535, 1538):
            self.write_dimension(dimension)
            with self.subTest(dimension=dimension), self.assertRaises(ValueError):
                CPTrainingConfig(feature_mode=FeatureMode.PRECOMPUTED, embedding_root=self.root).validate()
        self.manifest_path.unlink()
        with self.assertRaises(FileNotFoundError):
            CIRTrainingConfig(feature_mode=FeatureMode.PRECOMPUTED, embedding_root=self.root).validate()
        self.assertEqual(CPTrainingConfig(embedding_root=self.root).model_config.model_dim, 1024)

    def test_1536_feature_forward_backward_and_checkpoint_restore(self):
        self.write_dimension(1536)
        for config_class, model_class, schema, load, restore in (
            (CPTrainingConfig, CPTrainingModel, 2, load_cp_checkpoint, restore_cp_model),
            (CIRTrainingConfig, CIRTrainingModel, 1, load_cir_checkpoint, restore_cir_model),
        ):
            with self.subTest(task=model_class.__name__):
                config = config_class(feature_mode=FeatureMode.PRECOMPUTED, embedding_root=self.root)
                small = replace(config.model_config, layers=1, feedforward_dim=16, max_items=3, dropout=0.0)
                config = replace(config, model=small)
                model = model_class(config.model_config, feature_mode=config.feature_mode)
                outfits = (torch.randn(2, 1536), torch.randn(1, 1536))
                forward = model if schema == 2 else model.embed_queries
                prediction = forward(outfits)
                prediction.sum().backward()
                self.assertTrue(any(p.grad is not None for p in model.parameters()))
                model.eval()
                with torch.no_grad():
                    expected = forward(outfits)
                path = self.root / f"{model_class.__name__}.pt"
                torch.save({
                    "checkpoint_schema_version": schema, "epoch": 1,
                    "run_config": config.as_dict(), "model_state_dict": model.state_dict(),
                }, path)
                restored = restore(load(path)).eval()
                restored_forward = restored if schema == 2 else restored.embed_queries
                with torch.no_grad():
                    torch.testing.assert_close(restored_forward(outfits), expected)

    def test_cp_transfer_with_marqo_category_embeddings(self):
        self.write_dimension(1536)
        config = CIRTrainingConfig(
            feature_mode=FeatureMode.PRECOMPUTED, embedding_root=self.root,
            use_category_embedding=True,
        )
        small = replace(config.model_config, layers=1, feedforward_dim=16, max_items=3)
        cp = CPTrainingModel(small, feature_mode=config.feature_mode)
        cir = CIRTrainingModel(
            small, feature_mode=config.feature_mode, use_category_embedding=True,
        )
        path = self.root / "cp.pt"
        torch.save(cp.state_dict(), path)
        report = load_cp_pretrained_weights(cir, path)
        self.assertGreater(report.loaded_tensor_count, 0)
        torch.testing.assert_close(
            cir.cir.task_embedding.embedding, cp.cp.task_embedding.embedding,
        )
        queries = cir.embed_queries((torch.randn(2, 1536),), ("tops",))
        self.assertEqual(queries.shape, (1, 128))
        self.assertTrue(bool(torch.isfinite(queries).all()))


if __name__ == "__main__":
    unittest.main()
