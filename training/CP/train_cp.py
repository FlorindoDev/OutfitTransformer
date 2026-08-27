"""Command-line entry point for Compatibility Prediction training."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from data import DEFAULT_DATASET_NAME, get_dataset_source
from model import DEFAULT_MODEL_CONFIG
from training.common import load_model_weights, resolve_device, seed_everything

from .config import CPTrainingConfig, DEFAULT_EMBEDDING_ROOT, FeatureMode
from .data import build_compatibility_loaders
from .model import CPTrainingModel
from .trainer import TrainingResult, train

LOGGER = logging.getLogger("training.CP")


def run(
    config: CPTrainingConfig,
    *,
    token: bool | str | None = True,
) -> TrainingResult:
    """Build dependencies and execute one CP training run."""
    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    LOGGER.info("device=%s seed=%d", device, config.seed)

    loaders = build_compatibility_loaders(config, token=token)
    model = CPTrainingModel(
        config.model_config,
        feature_mode=config.feature_mode,
    )
    if config.resume is not None:
        load_model_weights(model, config.resume, map_location="cpu")
        LOGGER.info(
            "loaded_weights=%s optimizer_scheduler_history=fresh",
            config.resume,
        )
    return train(model, loaders, config, device)


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[CPTrainingConfig, bool | str | None]:
    default_source = get_dataset_source(DEFAULT_DATASET_NAME)
    parser = argparse.ArgumentParser(
        description="Train OutfitTransformer Compatibility Prediction."
    )
    feature_source = parser.add_mutually_exclusive_group()
    feature_source.add_argument(
        "--classic",
        action="store_const",
        const=FeatureMode.CLASSIC,
        dest="feature_mode",
    )
    feature_source.add_argument(
        "--new-classic",
        action="store_const",
        const=FeatureMode.NEW_CLASSIC,
        dest="feature_mode",
    )
    feature_source.add_argument(
        "--clip",
        action="store_const",
        const=FeatureMode.CLIP,
        dest="feature_mode",
    )
    parser.set_defaults(feature_mode=FeatureMode.CLIP)
    parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--subset",
        default=default_source.descriptor.default_subset,
        help="dataset subset",
    )
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=DEFAULT_EMBEDDING_ROOT,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=DEFAULT_MODEL_CONFIG.compatibility.focal_alpha,
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=DEFAULT_MODEL_CONFIG.compatibility.focal_gamma,
    )
    parser.add_argument(
        "--focal-reduction",
        choices=["mean", "sum"],
        default=DEFAULT_MODEL_CONFIG.compatibility.focal_reduction,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--best-metric",
        choices=["val_auc", "val_accuracy", "val_loss"],
        default="val_auc",
    )
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
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
    checkpoint_dir = arguments.checkpoint_dir or (
        Path("checkpoints")
        / subset
        / f"cp_{arguments.feature_mode.value}"
    )
    config = CPTrainingConfig(
        dataset_name=source.descriptor.name,
        subset=subset,
        feature_mode=arguments.feature_mode,
        embedding_root=arguments.embedding_root,
        dataset_root=dataset_root,
        checkpoint_dir=checkpoint_dir,
        cache_dir=arguments.cache_dir,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        focal_alpha=arguments.focal_alpha,
        focal_gamma=arguments.focal_gamma,
        focal_reduction=arguments.focal_reduction,
        seed=arguments.seed,
        best_metric=arguments.best_metric,
        early_stopping_patience=arguments.early_stopping_patience,
        early_stopping_min_delta=arguments.early_stopping_min_delta,
        num_workers=arguments.num_workers,
        pin_memory=arguments.pin_memory,
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
