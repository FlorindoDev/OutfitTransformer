"""Command-line entry point for Complementary Item Retrieval training."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from pathlib import Path

from data import DEFAULT_DATASET_NAME, get_dataset_source
from model import DEFAULT_CIR_CONFIG
from training.common import load_model_weights, seed_everything
from training.common.features import (
    DEFAULT_PRECOMPUTED_EMBEDDING_ROOT,
    FeatureMode,
)

from .config import CIRTrainingConfig
from .data import build_retrieval_loaders
from .distributed import close_distributed, initialize_distributed
from .model import CIRTrainingModel
from .trainer import TrainingResult, train

LOGGER = logging.getLogger("training.CIR")


def run(
    config: CIRTrainingConfig,
    *,
    token: bool | str | None = True,
) -> TrainingResult:
    """Build dependencies and execute one CIR training run."""
    config.validate()
    runtime = initialize_distributed(config)
    try:
        seed_everything(config.seed + runtime.rank)
        if runtime.is_main_process:
            LOGGER.info(
                "device=%s seed=%d world_size=%d",
                runtime.device,
                config.seed,
                runtime.world_size,
            )
        loaders = build_retrieval_loaders(config, token=token)
        model = CIRTrainingModel(
            config.model_config,
            config.cir_config,
            feature_mode=config.feature_mode,
            use_category_embedding=config.use_category_embedding,
        )
        if config.resume is not None:
            load_model_weights(model, config.resume, map_location="cpu")
            if runtime.is_main_process:
                LOGGER.info(
                    "loaded_weights=%s optimizer_scheduler_history=fresh",
                    config.resume,
                )
        return train(model, loaders, config, runtime)
    finally:
        close_distributed(runtime)


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[CIRTrainingConfig, bool | str | None]:
    default_source = get_dataset_source(DEFAULT_DATASET_NAME)
    parser = argparse.ArgumentParser(
        description="Train OutfitTransformer Complementary Item Retrieval."
    )
    feature_source = parser.add_mutually_exclusive_group()
    feature_source.add_argument(
        "--classic",
        action="store_const",
        const=FeatureMode.CLASSIC,
        dest="feature_mode",
        help="encode raw inputs with the classic 64+64 profile",
    )
    feature_source.add_argument(
        "--new-classic",
        action="store_const",
        const=FeatureMode.NEW_CLASSIC,
        dest="feature_mode",
        help="encode raw inputs with the default 512+512 profile",
    )
    feature_source.add_argument(
        "--precomputed",
        action="store_const",
        const=FeatureMode.PRECOMPUTED,
        dest="feature_mode",
        help="load item embeddings precomputed with a compatible model",
    )
    parser.set_defaults(feature_mode=FeatureMode.NEW_CLASSIC)
    parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--subset",
        default=default_source.descriptor.default_subset,
        help="dataset subset",
    )
    parser.add_argument("--embedding-root", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--triplet-margin",
        type=float,
        default=DEFAULT_CIR_CONFIG.triplet_margin,
    )
    parser.add_argument(
        "--loss-reduction",
        choices=["mean", "sum"],
        default="mean",
    )
    parser.add_argument(
        "--retrieval-embedding-dim",
        type=int,
        default=DEFAULT_CIR_CONFIG.embedding_dim,
    )
    parser.add_argument("--normalize-embeddings", action="store_true")
    parser.add_argument(
        "--category-emb",
        action="store_true",
        help="condition CIR queries on the positive item's target category",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="enable CUDA autocast and gradient scaling",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="use torchrun environment and DistributedDataParallel",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", type=Path)
    authentication = parser.add_mutually_exclusive_group()
    authentication.add_argument("--token")
    authentication.add_argument("--no-token", action="store_true")
    arguments = parser.parse_args(argv)

    source = get_dataset_source(arguments.dataset)
    subset = source.descriptor.validate_subset(arguments.subset)
    dataset_root = arguments.dataset_root or source.descriptor.default_root
    category_suffix = "_category" if arguments.category_emb else ""
    checkpoint_dir = arguments.checkpoint_dir or (
        Path("checkpoints")
        / subset
        / f"cir_{arguments.feature_mode.value}{category_suffix}"
    )
    embedding_root = (
        arguments.embedding_root or DEFAULT_PRECOMPUTED_EMBEDDING_ROOT
    )
    config = CIRTrainingConfig(
        dataset_name=source.descriptor.name,
        subset=subset,
        feature_mode=arguments.feature_mode,
        embedding_root=embedding_root,
        dataset_root=dataset_root,
        checkpoint_dir=checkpoint_dir,
        cache_dir=arguments.cache_dir,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        triplet_margin=arguments.triplet_margin,
        loss_reduction=arguments.loss_reduction,
        retrieval_embedding_dim=arguments.retrieval_embedding_dim,
        normalize_embeddings=arguments.normalize_embeddings,
        use_category_embedding=arguments.category_emb,
        seed=arguments.seed,
        early_stopping_patience=arguments.early_stopping_patience,
        early_stopping_min_delta=arguments.early_stopping_min_delta,
        num_workers=arguments.num_workers,
        pin_memory=arguments.pin_memory,
        mixed_precision=arguments.mixed_precision,
        ddp=arguments.ddp,
        device=arguments.device,
        log_every=arguments.log_every,
        resume=arguments.resume,
    )
    config.validate()
    token: bool | str | None = (
        False if arguments.no_token else arguments.token or True
    )
    return config, token


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    config, token = parse_args(argv)
    result = run(config, token=token)
    if not config.ddp or os.environ.get("RANK", "0") == "0":
        LOGGER.info(
            "training_complete epochs=%d best_%s=%.6f checkpoint=%s",
            result.epochs_completed,
            result.best_metric,
            result.best_value,
            result.best_checkpoint,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
