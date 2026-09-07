"""Small local checkpoints and FITB data, independent of downloaded assets."""

import json
from pathlib import Path

import torch

from data import DataSplit
from model import TransformerConfig
from training.CIR import CIRTrainingConfig, CIRTrainingModel
from training.common.features import FeatureMode


def training_config(root: Path, *, categories: bool = False) -> CIRTrainingConfig:
    return CIRTrainingConfig(
        feature_mode=FeatureMode.PRECOMPUTED,
        embedding_root=root / "embeddings",
        dataset_root=root / "dataset",
        retrieval_embedding_dim=3,
        use_category_embedding=categories,
        model=TransformerConfig(
            modality_embedding_dim=4,
            layers=1,
            attention_heads=2,
            feedforward_dim=12,
            dropout=0.0,
            max_items=3,
            layer_norm_epsilon=1e-2,
        ),
    )


def write_checkpoint(
    path: Path, config: CIRTrainingConfig, model: CIRTrainingModel,
) -> None:
    torch.save(
        {
            "checkpoint_schema_version": 1,
            "epoch": 3,
            "run_config": config.as_dict(),
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def write_fitb_data(config: CIRTrainingConfig, split: DataSplit) -> None:
    stem = "valid" if split is DataSplit.VALIDATION else "test"
    annotation_dir = config.dataset_root / config.subset
    annotation_dir.mkdir(parents=True, exist_ok=True)
    ids = [str(index) for index in range(1, 6)]
    outfits = [{
        "set_id": "a",
        "items": [{"item_id": item_id, "index": item_id} for item_id in ids],
    }]
    questions = [
        {
            "question": ["a_1"],
            "answers": ["a_5", "a_3", "a_2", "a_4"],
            "blank_position": position,
        }
        for position in (2, 3, 4)
    ]
    (annotation_dir / f"{stem}.json").write_text(json.dumps(outfits), encoding="utf-8")
    (annotation_dir / f"fill_in_blank_{stem}.json").write_text(
        json.dumps(questions), encoding="utf-8",
    )
    if config.use_category_embedding:
        metadata = {item_id: {"semantic_category": "tops"} for item_id in ids}
        (config.dataset_root / "polyvore_item_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8",
        )

    cache_dir = config.embedding_root / config.subset / split.value
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"item_ids": ids, "embeddings": torch.randn(5, config.model_config.model_dim)},
        cache_dir / "shard.pt",
    )
    (cache_dir / "manifest.json").write_text(json.dumps({
        "schema_version": 2,
        "dataset": "mvasil/polyvore-outfits",
        "subset": config.subset,
        "split": split.value,
        "embedding_dim": config.model_config.model_dim,
        "count": 5,
        "model_fingerprint": "test-fixture",
        "shards": [{"file": "shard.pt"}],
    }), encoding="utf-8")
