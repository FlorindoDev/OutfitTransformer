from __future__ import annotations

import argparse
import random
from collections.abc import Mapping, Sequence, Sized
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from data import CompatibilityBatch, create_polyvore_compatibility_loader
from model import (
    IMAGE_FINE_TUNE_MODES,
    BinaryFocalLoss,
    CompatibilityPredictor,
    OutfitEncoderConfig,
)
from .fine_tuning import (
    CPFineTuneCheckpoint,
    CPFineTuneOptimizerConfig,
    CP_OPTIMIZER_NAMES,
    build_cp_fine_tune_optimizer,
    optimizer_learning_rates,
)
from .plotting import CPHistoryPlotter
from .selection import CP_BEST_METRICS
from .trainer import train_cp
from .types import (
    CPBatchProgress,
    CPCheckpointInfo,
    CPEpochMetrics,
    CPTrainingHistory,
)


SchedulerName = Literal["none", "step", "cosine"]
LossName = Literal["focal", "bce"]
SCHEDULER_NAMES: tuple[SchedulerName, ...] = ("none", "step", "cosine")
LOSS_NAMES: tuple[LossName, ...] = ("focal", "bce")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start a new CP fine-tuning phase from model weights in any "
            "compatible CP checkpoint"
        ),
    )
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--additional-epochs", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/cp_fine_tune"),
    )
    parser.add_argument(
        "--variant",
        choices=("nondisjoint", "disjoint"),
        default=None,
        help="dataset variant; default inherits checkpoint or uses disjoint",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--image-backbone-learning-rate",
        type=float,
        default=None,
        help="LR for trainable ResNet blocks; default uses --learning-rate",
    )
    parser.add_argument(
        "--optimizer",
        choices=CP_OPTIMIZER_NAMES,
        default="adam",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument(
        "--scheduler",
        choices=SCHEDULER_NAMES,
        default="step",
    )
    parser.add_argument("--lr-step-size", type=int, default=10)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument("--loss", choices=LOSS_NAMES, default="focal")
    parser.add_argument(
        "--focal-alpha",
        type=_optional_float,
        default=0.5,
        help="positive-class alpha or 'none'",
    )
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument(
        "--best-metric",
        choices=CP_BEST_METRICS,
        default="val_auc",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=_optional_float,
        default=1.0,
        help="gradient clipping norm or 'none'",
    )
    parser.add_argument(
        "--image-fine-tune-mode",
        choices=IMAGE_FINE_TUNE_MODES,
        default="fc_and_layer4",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="override checkpoint dropout without changing weight shapes",
    )
    parser.add_argument(
        "--text-model",
        default=None,
        help="override checkpoint SentenceBERT location; architecture must match",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    _validate_output_directory(args.output_dir)
    _seed_everything(args.seed)

    source = CPFineTuneCheckpoint.load(args.source_checkpoint, device="cpu")
    variant = _resolve_variant(args.variant, source)
    model_config = source.model_config(
        image_fine_tune_mode=args.image_fine_tune_mode,
        dropout=args.dropout,
        text_model_name=args.text_model,
    )
    model = CompatibilityPredictor(config=model_config)
    source.load_weights(model)

    train_loader, validation_loader = _create_loaders(args, variant)
    criterion = _create_criterion(args)
    optimizer_config = CPFineTuneOptimizerConfig(
        name=args.optimizer,
        learning_rate=args.learning_rate,
        image_backbone_learning_rate=args.image_backbone_learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        eps=args.adam_eps,
    )
    optimizer = build_cp_fine_tune_optimizer(model, optimizer_config)
    scheduler = _create_scheduler(args, optimizer)

    start_epoch = source.epoch + 1
    final_epoch = source.epoch + args.additional_epochs
    best_path = args.output_dir / "best.pt"
    epoch_directory = args.output_dir / "epochs"
    plot_directory = args.output_dir / "plots"
    _print_startup(
        args,
        source,
        variant=variant,
        start_epoch=start_epoch,
        final_epoch=final_epoch,
        optimizer=optimizer,
    )
    _print_data_summary(train_loader, validation_loader)
    _print_parameter_summary(model)

    plotter = None if args.no_plots else CPHistoryPlotter(plot_directory)
    progress_interval = args.log_interval if args.log_interval > 0 else None
    train_cp(
        model=model,
        train_batches=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=final_epoch,
        device=args.device,
        validation_batches=validation_loader,
        scheduler=scheduler,
        max_grad_norm=args.max_grad_norm,
        checkpoint_path=best_path,
        epoch_checkpoint_dir=epoch_directory,
        start_epoch=start_epoch,
        best_metric=args.best_metric,
        initial_history=CPTrainingHistory(),
        checkpoint_metadata=_checkpoint_metadata(
            args,
            source,
            variant=variant,
            model_config=model_config,
            optimizer_config=optimizer_config,
        ),
        progress_interval=progress_interval,
        on_batch_end=_print_batch if progress_interval is not None else None,
        on_checkpoint_saved=_print_checkpoint,
        on_history_updated=(
            None if plotter is None else partial(_plot_and_report, plotter)
        ),
        on_epoch_end=partial(_print_epoch, optimizer=optimizer),
    )


def _optional_float(value: str) -> float | None:
    if value.lower() == "none":
        return None
    try:
        return float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a number or 'none'") from error


def _validate_args(args: argparse.Namespace) -> None:
    if args.additional_epochs <= 0:
        raise ValueError("--additional-epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    optimizer_config = CPFineTuneOptimizerConfig(
        name=args.optimizer,
        learning_rate=args.learning_rate,
        image_backbone_learning_rate=args.image_backbone_learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        eps=args.adam_eps,
    )
    optimizer_config.validate()
    if args.scheduler == "step" and args.lr_step_size <= 0:
        raise ValueError("--lr-step-size must be positive")
    if args.scheduler == "step" and not 0.0 < args.lr_gamma <= 1.0:
        raise ValueError("--lr-gamma must be in (0, 1]")
    if args.min_learning_rate < 0.0:
        raise ValueError("--min-learning-rate must be non-negative")
    initial_learning_rates = [args.learning_rate]
    if args.image_backbone_learning_rate is not None:
        initial_learning_rates.append(args.image_backbone_learning_rate)
    if (
        args.scheduler == "cosine"
        and args.min_learning_rate > min(initial_learning_rates)
    ):
        raise ValueError(
            "--min-learning-rate cannot exceed an initial learning rate"
        )
    if args.focal_alpha is not None and not 0.0 <= args.focal_alpha <= 1.0:
        raise ValueError("--focal-alpha must be in [0, 1] or none")
    if args.focal_gamma < 0.0:
        raise ValueError("--focal-gamma must be non-negative")
    if args.max_grad_norm is not None and args.max_grad_norm <= 0.0:
        raise ValueError("--max-grad-norm must be positive or none")
    if args.dropout is not None and not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if args.log_interval < 0:
        raise ValueError("--log-interval must be non-negative")


def _validate_output_directory(output_directory: Path) -> None:
    if output_directory.exists() and not output_directory.is_dir():
        raise NotADirectoryError(
            f"fine-tune output must be a directory: {output_directory}"
        )
    if output_directory.exists() and any(output_directory.rglob("*.pt")):
        raise FileExistsError(
            f"fine-tune output already contains checkpoints: {output_directory}"
        )


def _resolve_variant(
    requested_variant: str | None,
    source: CPFineTuneCheckpoint,
) -> str:
    if requested_variant is not None:
        return requested_variant
    if source.run_config is not None:
        dataset_config = source.run_config.get("dataset")
        if isinstance(dataset_config, Mapping):
            saved_variant = dataset_config.get("variant")
            if saved_variant in ("disjoint", "nondisjoint"):
                return saved_variant
    return "disjoint"


def _create_loaders(
    args: argparse.Namespace,
    variant: str,
) -> tuple[DataLoader[CompatibilityBatch], DataLoader[CompatibilityBatch]]:
    common_options = {
        "variant": variant,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "pin_memory": str(args.device).startswith("cuda"),
        "cache_dir": args.cache_dir,
    }
    return (
        create_polyvore_compatibility_loader(split="train", **common_options),
        create_polyvore_compatibility_loader(
            split="validation",
            **common_options,
        ),
    )


def _create_criterion(args: argparse.Namespace) -> nn.Module:
    if args.loss == "bce":
        return nn.BCEWithLogitsLoss()
    return BinaryFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)


def _create_scheduler(args: argparse.Namespace, optimizer: Optimizer) -> Any | None:
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=args.additional_epochs,
            eta_min=args.min_learning_rate,
        )
    return StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
    )


def _checkpoint_metadata(
    args: argparse.Namespace,
    source: CPFineTuneCheckpoint,
    *,
    variant: str,
    model_config: OutfitEncoderConfig,
    optimizer_config: CPFineTuneOptimizerConfig,
) -> dict[str, Any]:
    return {
        "dataset": {
            "id": "mvasil/polyvore-outfits",
            "variant": variant,
        },
        "model": asdict(model_config),
        "training": {
            "phase": "fine_tune",
            "additional_epochs": args.additional_epochs,
            "batch_size": args.batch_size,
            "optimizer": optimizer_config.name,
            "learning_rate": optimizer_config.learning_rate,
            "image_backbone_learning_rate": (
                optimizer_config.resolved_image_backbone_learning_rate
            ),
            "weight_decay": optimizer_config.weight_decay,
            "adam_beta1": optimizer_config.beta1,
            "adam_beta2": optimizer_config.beta2,
            "adam_eps": optimizer_config.eps,
            "scheduler": args.scheduler,
            "lr_step_size": args.lr_step_size,
            "lr_gamma": args.lr_gamma,
            "min_learning_rate": args.min_learning_rate,
            "loss": args.loss,
            "focal_alpha": args.focal_alpha,
            "focal_gamma": args.focal_gamma,
            "best_metric": args.best_metric,
            "max_grad_norm": args.max_grad_norm,
            "seed": args.seed,
        },
        "fine_tuning": {
            "source_checkpoint": str(source.path.resolve()),
            "source_epoch": source.epoch,
            "source_schema_version": source.schema_version,
            "optimizer_state_reused": False,
            "scheduler_state_reused": False,
            "history_reused": False,
            "rng_state_reused": False,
        },
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _print_startup(
    args: argparse.Namespace,
    source: CPFineTuneCheckpoint,
    *,
    variant: str,
    start_epoch: int,
    final_epoch: int,
    optimizer: Optimizer,
) -> None:
    print("training=compatibility_prediction_fine_tune")
    print(f"source_checkpoint={source.path.resolve()} source_epoch={source.epoch}")
    print(f"epochs={start_epoch}-{final_epoch} dataset_variant={variant}")
    print(f"device={args.device} seed={args.seed} batch_size={args.batch_size}")
    print(
        f"resnet_fine_tune_mode={args.image_fine_tune_mode} "
        f"optimizer={args.optimizer} scheduler={args.scheduler} loss={args.loss}"
    )
    print(
        "learning_rates="
        + ",".join(
            f"{name}:{value:.8g}"
            for name, value in optimizer_learning_rates(optimizer).items()
        )
    )
    print(f"best_metric={args.best_metric} output_dir={args.output_dir.resolve()}")
    print("source_optimizer_scheduler_history_rng=discarded")


def _print_data_summary(
    train_loader: DataLoader[CompatibilityBatch],
    validation_loader: DataLoader[CompatibilityBatch],
) -> None:
    if not isinstance(train_loader.dataset, Sized) or not isinstance(
        validation_loader.dataset,
        Sized,
    ):
        raise TypeError("CP datasets must provide their number of examples")
    print(
        f"train_examples={len(train_loader.dataset)} "
        f"validation_examples={len(validation_loader.dataset)}"
    )


def _print_parameter_summary(model: CompatibilityPredictor) -> None:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"model_parameters={total} trainable_parameters={trainable}")


def _print_epoch(
    epoch: int,
    train_metrics: CPEpochMetrics,
    validation_metrics: CPEpochMetrics | None,
    *,
    optimizer: Optimizer,
) -> None:
    message = (
        f"epoch={epoch} train_loss={train_metrics.loss:.6f} "
        f"train_accuracy={train_metrics.accuracy:.4f}"
    )
    if train_metrics.auc is not None:
        message += f" train_auc={train_metrics.auc:.4f}"
    if validation_metrics is not None:
        message += (
            f" val_loss={validation_metrics.loss:.6f}"
            f" val_accuracy={validation_metrics.accuracy:.4f}"
            f" val_auc={validation_metrics.auc:.4f}"
        )
    message += " lr=" + ",".join(
        f"{name}:{value:.8g}"
        for name, value in optimizer_learning_rates(optimizer).items()
    )
    print(message)


def _print_batch(progress: CPBatchProgress) -> None:
    batches = progress.batches if progress.batches is not None else "?"
    print(
        f"epoch={progress.epoch} phase={progress.phase} "
        f"batch={progress.batch}/{batches} "
        f"running_loss={progress.running_loss:.6f} "
        f"running_accuracy={progress.running_accuracy:.4f}"
    )


def _print_checkpoint(info: CPCheckpointInfo) -> None:
    print(
        f"checkpoint={info.kind} epoch={info.epoch} "
        f"path={info.path.resolve()} metric={info.selection_metric} "
        f"value={info.selection_value:.6f} "
        f"best={info.best_selection_value:.6f}"
    )


def _plot_and_report(
    plotter: CPHistoryPlotter,
    history: CPTrainingHistory,
) -> None:
    paths = plotter(history)
    print("plots_saved=" + ",".join(str(path.resolve()) for path in paths))


if __name__ == "__main__":
    main()
