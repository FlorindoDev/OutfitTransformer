"""Command-line entry point for Compatibility Prediction evaluation."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from data.polyvore import DEFAULT_DATASET_ROOT, PolyvoreSplit
from training.CP.data import (
    CompatibilityDataConfig,
    build_compatibility_loader,
)
from training.common import resolve_device, seed_everything, write_json

from .checkpoint import load_cp_checkpoint, restore_cp_model
from .config import CPEvaluationConfig
from .evaluator import CPEvaluationResult, evaluate

LOGGER = logging.getLogger("evaluation.CP")


def run(
    config: CPEvaluationConfig,
    *,
    token: bool | str | None = True,
) -> CPEvaluationResult:
    """Restore one checkpoint, evaluate one split and save its report."""
    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    checkpoint = load_cp_checkpoint(config.checkpoint)
    embedding_root = config.embedding_root or checkpoint.embedding_root
    if embedding_root is None:
        embedding_root = Path("precomputed_embeddings")
    dataset_root = (
        config.dataset_root
        or checkpoint.dataset_root
        or DEFAULT_DATASET_ROOT
    )

    data_config = CompatibilityDataConfig(
        variant=checkpoint.variant,
        feature_mode=checkpoint.feature_mode,
        embedding_root=embedding_root,
        dataset_root=dataset_root,
        cache_dir=config.cache_dir or checkpoint.cache_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        seed=config.seed,
        model_config=checkpoint.model_config,
    )
    loader = build_compatibility_loader(
        data_config,
        split=config.split,
        shuffle=False,
        token=token,
    )
    metrics = evaluate(
        restore_cp_model(checkpoint),
        loader,
        device,
        threshold=config.threshold,
        log_every=config.log_every,
    )
    output_path = config.output_path or _default_output_path(
        checkpoint.path,
        checkpoint.variant.value,
        config.split,
    )
    result = CPEvaluationResult(
        checkpoint=checkpoint.path,
        checkpoint_epoch=checkpoint.epoch,
        output_path=output_path,
        split=config.split,
        variant=checkpoint.variant,
        feature_mode=checkpoint.feature_mode,
        threshold=config.threshold,
        metrics=metrics,
    )
    write_json(result.as_dict(), output_path)
    return result


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[CPEvaluationConfig, bool | str | None]:
    parser = argparse.ArgumentParser(
        description="Evaluate an OutfitTransformer CP checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=[PolyvoreSplit.VALIDATION.value, PolyvoreSplit.TEST.value],
        default=PolyvoreSplit.TEST.value,
    )
    parser.add_argument("--embedding-root", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=10)
    authentication = parser.add_mutually_exclusive_group()
    authentication.add_argument("--token")
    authentication.add_argument("--no-token", action="store_true")
    arguments = parser.parse_args(argv)

    config = CPEvaluationConfig(
        checkpoint=arguments.checkpoint,
        split=PolyvoreSplit(arguments.split),
        embedding_root=arguments.embedding_root,
        dataset_root=arguments.dataset_root,
        output_path=arguments.output,
        cache_dir=arguments.cache_dir,
        batch_size=arguments.batch_size,
        threshold=arguments.threshold,
        seed=arguments.seed,
        num_workers=arguments.num_workers,
        pin_memory=arguments.pin_memory,
        device=arguments.device,
        log_every=arguments.log_every,
    )
    config.validate()
    token: bool | str | None = (
        False if arguments.no_token else arguments.token or True
    )
    return config, token


def _default_output_path(
    checkpoint: Path,
    variant: str,
    split: PolyvoreSplit,
) -> Path:
    run_name = (
        checkpoint.parent.parent.name
        if checkpoint.parent.name == "epochs"
        else checkpoint.parent.name
    )
    return (
        Path("results")
        / "cp"
        / variant
        / run_name
        / f"{checkpoint.stem}_{split.value}.json"
    )


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    config, token = parse_args(argv)
    result = run(config, token=token)
    LOGGER.info(
        "evaluation_complete split=%s examples=%d accuracy=%.4f "
        "precision=%.4f recall=%.4f f1=%.4f auc=%.4f output=%s",
        result.split.value,
        result.metrics.examples,
        result.metrics.accuracy,
        result.metrics.precision,
        result.metrics.recall,
        result.metrics.f1,
        result.metrics.auc,
        result.output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
