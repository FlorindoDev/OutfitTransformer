"""Command-line entry point for Complementary Item Retrieval evaluation."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from data import DataSplit, get_dataset_source
from training.CIR.data import RetrievalDataConfig, build_retrieval_loader
from training.common import resolve_device, seed_everything, write_json

from .checkpoint import load_cir_checkpoint, restore_cir_model
from .config import CIREvaluationConfig
from .evaluator import CIREvaluationResult, evaluate

LOGGER = logging.getLogger("evaluation.CIR")


def run(
    config: CIREvaluationConfig,
    *,
    token: bool | str | None = True,
) -> CIREvaluationResult:
    """Restore one checkpoint, evaluate one FITB split and save its report."""
    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    checkpoint = load_cir_checkpoint(config.checkpoint)
    model = restore_cir_model(checkpoint)
    source = get_dataset_source(checkpoint.dataset_name)
    data_config = RetrievalDataConfig(
        dataset_name=checkpoint.dataset_name,
        subset=checkpoint.subset,
        feature_mode=checkpoint.feature_mode,
        embedding_root=(
            config.embedding_root or checkpoint.embedding_root or Path("precomputed_embeddings")
        ),
        dataset_root=(
            config.dataset_root or checkpoint.dataset_root or source.descriptor.default_root
        ),
        cache_dir=config.cache_dir or checkpoint.cache_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        seed=config.seed,
        use_category_embedding=checkpoint.use_category_embedding,
        distributed=False,
        model_config=checkpoint.model_config,
    )
    loader = build_retrieval_loader(data_config, split=config.split, token=token)
    metrics = evaluate(
        model,
        loader,
        device,
        candidate_batch_size=config.candidate_batch_size,
        log_every=config.log_every,
    )
    output_path = config.output_path or _default_output_path(
        checkpoint.path, checkpoint.dataset_name, checkpoint.subset, config.split,
    )
    result = CIREvaluationResult(
        checkpoint=checkpoint.path,
        checkpoint_epoch=checkpoint.epoch,
        output_path=output_path,
        split=config.split,
        dataset_name=checkpoint.dataset_name,
        dataset_id=checkpoint.dataset_id,
        subset=checkpoint.subset,
        feature_mode=checkpoint.feature_mode,
        use_category_embedding=checkpoint.use_category_embedding,
        metrics=metrics,
    )
    write_json(result.as_dict(), output_path)
    return result


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[CIREvaluationConfig, bool | str | None]:
    parser = argparse.ArgumentParser(
        description="Evaluate an OutfitTransformer CIR checkpoint on official FITB queries."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=[DataSplit.VALIDATION.value, DataSplit.TEST.value],
        default=DataSplit.TEST.value,
    )
    parser.add_argument("--embedding-root", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--candidate-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=10)
    authentication = parser.add_mutually_exclusive_group()
    authentication.add_argument("--token")
    authentication.add_argument("--no-token", action="store_true")
    arguments = parser.parse_args(argv)

    config = CIREvaluationConfig(
        checkpoint=arguments.checkpoint,
        split=DataSplit(arguments.split),
        embedding_root=arguments.embedding_root,
        dataset_root=arguments.dataset_root,
        output_path=arguments.output,
        cache_dir=arguments.cache_dir,
        batch_size=arguments.batch_size,
        candidate_batch_size=arguments.candidate_batch_size,
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
    dataset_name: str,
    subset: str,
    split: DataSplit,
) -> Path:
    run_name = (
        checkpoint.parent.parent.name
        if checkpoint.parent.name == "epochs"
        else checkpoint.parent.name
    )
    return (
        Path("results") / "cir" / dataset_name / subset / run_name
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
        "evaluation_complete protocol=polyvore_fitb split=%s examples=%d "
        "fitb_accuracy=%.4f mrr=%.4f recall_at_2=%.4f output=%s",
        result.split.value,
        result.metrics.examples,
        result.metrics.fitb_accuracy,
        result.metrics.mrr,
        result.metrics.recall_at_2,
        result.output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
