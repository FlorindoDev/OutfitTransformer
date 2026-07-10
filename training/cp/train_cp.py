from __future__ import annotations

import argparse
import random
from collections.abc import Sequence, Sized
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from data import CompatibilityBatch, create_polyvore_compatibility_loader
from model import (
    IMAGE_FINE_TUNE_MODES,
    BinaryFocalLoss,
    CompatibilityPredictor,
    OutfitEncoderConfig,
)
from .checkpointing import CPResumeState, load_cp_training_checkpoint
from .plotting import CPHistoryPlotter
from .selection import CP_BEST_METRICS, CPSelectionCriterion
from .trainer import train_cp
from .types import (
    CPBatchProgress,
    CPCheckpointInfo,
    CPEpochMetrics,
    CPTrainingHistory,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train OutfitTransformer for compatibility prediction",
    )
    parser.add_argument(
        "--variant",
        choices=("nondisjoint", "disjoint"),
        default="disjoint",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-step-size", type=int, default=10)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--focal-alpha", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="print batch progress every N batches; use 0 to disable",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/cp_best.pt"),
    )
    parser.add_argument(
        "--best-metric",
        choices=CP_BEST_METRICS,
        default="val_loss",
        help="validation metric used to select the best checkpoint",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/cp_epochs"),
        help="directory for one checkpoint per epoch",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="resume training from a saved CP checkpoint",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("checkpoints/cp_plots"),
        help="directory for cumulative charts saved after every epoch",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="disable cumulative training charts",
    )
    parser.add_argument(
        "--text-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--no-pretrained-image",
        action="store_true",
        help=(
            "initialize ResNet-18 without ImageNet weights; frozen blocks "
            "remain frozen according to --image-fine-tune-mode"
        ),
    )
    parser.add_argument(
        "--image-fine-tune-mode",
        choices=IMAGE_FINE_TUNE_MODES,
        default="fc_only",
        help="train the image FC, layer4 and FC, or the full ResNet",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    _seed_everything(args.seed)
    _print_startup(args)

    train_loader, validation_loader = _create_loaders(args)
    _print_data_summary(train_loader, validation_loader)

    model_config = OutfitEncoderConfig(
        text_model_name=args.text_model,
        pretrained_image_encoder=not args.no_pretrained_image,
        image_fine_tune_mode=args.image_fine_tune_mode,
    )
    model = CompatibilityPredictor(config=model_config)
    _print_parameter_summary(model)
    criterion = BinaryFocalLoss(
        alpha=args.focal_alpha,
        gamma=args.focal_gamma,
    )
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
    )

    resume_state = _load_resume_if_requested(
        args,
        model,
        optimizer,
        scheduler,
    )
    start_epoch = 1 if resume_state is None else resume_state.epoch + 1
    selection_criterion = CPSelectionCriterion(args.best_metric)
    initial_best_value = _initial_best_value(
        selection_criterion,
        resume_state,
    )
    initial_history = (
        CPTrainingHistory() if resume_state is None else resume_state.history
    )
    progress_interval = args.log_interval if args.log_interval > 0 else None
    history_callback = _create_history_callback(args)

    train_cp(
        model=model,
        train_batches=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=args.epochs,
        device=args.device,
        validation_batches=validation_loader,
        scheduler=scheduler,
        max_grad_norm=args.max_grad_norm,
        checkpoint_path=args.checkpoint,
        epoch_checkpoint_dir=args.checkpoint_dir,
        start_epoch=start_epoch,
        best_metric=args.best_metric,
        initial_best_value=initial_best_value,
        initial_history=initial_history,
        checkpoint_metadata=_checkpoint_metadata(
            args,
            model_config,
            optimizer,
            scheduler,
            resume_state,
        ),
        progress_interval=progress_interval,
        on_batch_end=_print_batch if progress_interval is not None else None,
        on_checkpoint_saved=_print_checkpoint,
        on_history_updated=history_callback,
        on_epoch_end=partial(_print_epoch, optimizer=optimizer),
    )


def _create_loaders(
    args: argparse.Namespace,
) -> tuple[DataLoader[CompatibilityBatch], DataLoader[CompatibilityBatch]]:
    common_options = {
        "variant": args.variant,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "pin_memory": str(args.device).startswith("cuda"),
        "cache_dir": args.cache_dir,
    }
    train_loader = create_polyvore_compatibility_loader(
        split="train",
        **common_options,
    )
    validation_loader = create_polyvore_compatibility_loader(
        split="validation",
        **common_options,
    )
    return train_loader, validation_loader


def _load_resume_if_requested(
    args: argparse.Namespace,
    model: CompatibilityPredictor,
    optimizer: Optimizer,
    scheduler: StepLR,
) -> CPResumeState | None:
    if args.resume is None:
        return None
    state = load_cp_training_checkpoint(
        path=args.resume,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
    )
    if state.epoch >= args.epochs:
        raise ValueError("--epochs must be greater than the resumed checkpoint epoch")
    _print_resume(args.resume, state)
    _warn_resume_configuration(state, args, optimizer, scheduler)
    return state


def _create_history_callback(args: argparse.Namespace) -> Any | None:
    if args.no_plots:
        return None
    plotter = CPHistoryPlotter(args.plot_dir)
    return partial(_plot_and_report, plotter)


def _plot_and_report(
    plotter: CPHistoryPlotter,
    history: CPTrainingHistory,
) -> None:
    paths = plotter(history)
    print(
        "plots_saved="
        + ",".join(str(path.resolve()) for path in paths)
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative")
    if args.lr_step_size <= 0:
        raise ValueError("--lr-step-size must be positive")
    if not 0.0 < args.lr_gamma <= 1.0:
        raise ValueError("--lr-gamma must be in (0, 1]")
    if not 0.0 <= args.focal_alpha <= 1.0:
        raise ValueError("--focal-alpha must be in [0, 1]")
    if args.focal_gamma < 0.0:
        raise ValueError("--focal-gamma must be non-negative")
    if args.max_grad_norm <= 0.0:
        raise ValueError("--max-grad-norm must be positive")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if args.log_interval < 0:
        raise ValueError("--log-interval must be non-negative")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _print_epoch(
    epoch: int,
    train_metrics: CPEpochMetrics,
    validation_metrics: CPEpochMetrics | None,
    *,
    optimizer: Optimizer,
) -> None:
    message = (
        f"epoch={epoch} "
        f"train_loss={train_metrics.loss:.6f} "
        f"train_accuracy={train_metrics.accuracy:.4f} "
        f"train_examples={train_metrics.examples}"
    )
    if train_metrics.auc is not None:
        message += f" train_auc={train_metrics.auc:.4f}"
    message += f" lr={_current_lr(optimizer):.8f}"
    if validation_metrics is not None:
        message += (
            f" val_loss={validation_metrics.loss:.6f}"
            f" val_accuracy={validation_metrics.accuracy:.4f}"
            f" val_auc={validation_metrics.auc:.4f}"
            f" val_examples={validation_metrics.examples}"
        )
    print(message)


def _print_startup(args: argparse.Namespace) -> None:
    print("training=compatibility_prediction")
    print(f"dataset=mvasil/polyvore-outfits variant={args.variant}")
    print(f"dataset_cache={_dataset_cache_location(args.cache_dir)}")
    print(f"hub_cache={_hub_cache_location(args.cache_dir)}")
    print(f"device={args.device} seed={args.seed}")
    print(
        "resnet_constructor_pretrained="
        f"{not args.no_pretrained_image} "
        f"resnet_fine_tune_mode={args.image_fine_tune_mode}"
    )
    print(f"checkpoint_best={args.checkpoint.resolve()}")
    print(f"checkpoint_best_metric={args.best_metric}")
    print(f"checkpoint_epochs={args.checkpoint_dir.resolve()}")
    if args.no_plots:
        print("plots=disabled")
    else:
        print(f"plot_directory={args.plot_dir.resolve()}")
    if args.resume is not None:
        print(f"resume_checkpoint={args.resume.resolve()}")
    if args.log_interval == 0:
        print("batch_logs=disabled")
    else:
        print(f"batch_logs=every_{args.log_interval}_batches")
    if args.no_pretrained_image and args.resume is None:
        print(
            "warning=random ResNet blocks remain frozen; "
            "--no-pretrained-image is not full training from scratch"
        )


def _print_data_summary(
    train_loader: DataLoader[CompatibilityBatch],
    validation_loader: DataLoader[CompatibilityBatch],
) -> None:
    train_dataset = train_loader.dataset
    validation_dataset = validation_loader.dataset
    if not isinstance(train_dataset, Sized) or not isinstance(
        validation_dataset,
        Sized,
    ):
        raise TypeError("CP datasets must provide their number of examples")
    print(
        f"train_examples={len(train_dataset)} "
        f"validation_examples={len(validation_dataset)}"
    )
    print(
        f"train_batches={len(train_loader)} "
        f"validation_batches={len(validation_loader)} "
        f"batch_size={train_loader.batch_size}"
    )


def _print_parameter_summary(model: CompatibilityPredictor) -> None:
    parameters = tuple(model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in parameters
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in parameters)
    image_parameters = tuple(model.encoder.image_encoder.parameters())
    trainable_image = sum(
        parameter.numel()
        for parameter in image_parameters
        if parameter.requires_grad
    )
    total_image = sum(parameter.numel() for parameter in image_parameters)
    print(f"model_parameters={total} trainable_parameters={trainable}")
    print(
        f"resnet_parameters={total_image} "
        f"trainable_resnet_parameters={trainable_image}"
    )


def _print_batch(progress: CPBatchProgress) -> None:
    batches = progress.batches if progress.batches is not None else "?"
    print(
        f"epoch={progress.epoch} "
        f"phase={progress.phase} "
        f"batch={progress.batch}/{batches} "
        f"loss={progress.loss:.6f} "
        f"running_loss={progress.running_loss:.6f} "
        f"running_accuracy={progress.running_accuracy:.4f} "
        f"examples={progress.examples}"
    )


def _print_checkpoint(info: CPCheckpointInfo) -> None:
    print(
        f"checkpoint={info.kind} "
        f"epoch={info.epoch} "
        f"path={info.path.resolve()} "
        f"selection_metric={info.selection_metric} "
        f"selection_source={info.selection_source} "
        f"selection_value={info.selection_value:.6f} "
        f"best_selection_value={info.best_selection_value:.6f}"
    )


def _print_resume(path: Path, state: CPResumeState) -> None:
    history_status = "complete" if state.history_complete else "legacy_partial"
    rng_status = "restored" if state.rng_restored else "legacy_unavailable"
    print(
        f"resume_loaded={path.resolve()} "
        f"resume_epoch={state.epoch} "
        f"next_epoch={state.epoch + 1} "
        f"selection_metric={state.selection_metric} "
        f"best_selection_value={state.best_selection_value:.6f} "
        f"history={history_status} "
        f"rng={rng_status}"
    )
    if not state.history_complete:
        print(
            "warning=legacy checkpoint lacks full history and historical best; "
            "changing --best-metric cannot reconstruct earlier epochs"
        )


def _initial_best_value(
    criterion: CPSelectionCriterion,
    state: CPResumeState | None,
) -> float:
    if state is None:
        return criterion.initial_best_value
    if state.selection_metric == criterion.metric:
        return state.best_selection_value

    if not state.history_complete:
        print(
            f"warning=best metric changed from {state.selection_metric} "
            f"to {criterion.metric}; only checkpoint epoch history is available"
        )
    else:
        print(
            f"resume_best_metric_changed={state.selection_metric}"
            f"->{criterion.metric}; historical best recomputed"
        )
    return criterion.best_value(state.history)


def _warn_resume_configuration(
    state: CPResumeState,
    args: argparse.Namespace,
    optimizer: Optimizer,
    scheduler: StepLR,
) -> None:
    if state.run_config is None:
        print(
            "warning=legacy checkpoint has no ResNet training policy; "
            f"resume uses {args.image_fine_tune_mode}"
        )
    else:
        _validate_and_warn_model_configuration(state.run_config, args)
        _warn_loss_and_data_configuration(state.run_config, args)
    _warn_optimizer_configuration(args, optimizer, scheduler)


def _validate_and_warn_model_configuration(
    run_config: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    model_config = run_config.get("model")
    if not isinstance(model_config, dict):
        return
    previous_text_model = model_config.get("text_model_name")
    if previous_text_model is not None and previous_text_model != args.text_model:
        raise ValueError(
            "--text-model must match the checkpoint text_model_name when resuming"
        )
    previous_mode = model_config.get("image_fine_tune_mode")
    if previous_mode is not None and previous_mode != args.image_fine_tune_mode:
        print(
            f"warning=ResNet fine-tune mode changed from {previous_mode} "
            f"to {args.image_fine_tune_mode}"
        )


def _warn_loss_and_data_configuration(
    run_config: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    dataset_config = run_config.get("dataset")
    if isinstance(dataset_config, dict):
        previous_variant = dataset_config.get("variant")
        if previous_variant is not None and previous_variant != args.variant:
            print(
                f"warning=dataset variant changed from {previous_variant} "
                f"to {args.variant}"
            )

    training_config = run_config.get("training")
    if not isinstance(training_config, dict):
        return
    comparisons = (
        ("batch_size", args.batch_size),
        ("focal_alpha", args.focal_alpha),
        ("focal_gamma", args.focal_gamma),
        ("max_grad_norm", args.max_grad_norm),
    )
    for name, current_value in comparisons:
        previous_value = training_config.get(name)
        if previous_value is not None and previous_value != current_value:
            print(
                f"warning={name} changed from {previous_value} "
                f"to {current_value}"
            )


def _warn_optimizer_configuration(
    args: argparse.Namespace,
    optimizer: Optimizer,
    scheduler: StepLR,
) -> None:
    print("resume_policy=checkpoint_optimizer_and_scheduler_state")
    optimizer_group = optimizer.param_groups[0]
    effective_values = (
        ("learning_rate", args.learning_rate, float(optimizer_group["lr"])),
        (
            "weight_decay",
            args.weight_decay,
            float(optimizer_group["weight_decay"]),
        ),
        ("lr_step_size", args.lr_step_size, scheduler.step_size),
        ("lr_gamma", args.lr_gamma, scheduler.gamma),
    )
    for name, requested_value, effective_value in effective_values:
        if requested_value != effective_value:
            print(
                f"warning={name} CLI value {requested_value} ignored on resume; "
                f"checkpoint value {effective_value} used"
            )


def _checkpoint_metadata(
    args: argparse.Namespace,
    model_config: OutfitEncoderConfig,
    optimizer: Optimizer,
    scheduler: StepLR,
    resume_state: CPResumeState | None,
) -> dict[str, Any]:
    effective_model_config = _effective_model_config(
        model_config,
        resume_state,
    )
    effective_seed = _effective_seed(args.seed, resume_state)
    optimizer_group = optimizer.param_groups[0]
    return {
        "dataset": {
            "id": "mvasil/polyvore-outfits",
            "variant": args.variant,
        },
        "model": effective_model_config,
        "training": {
            "batch_size": args.batch_size,
            "learning_rate": float(optimizer_group["lr"]),
            "weight_decay": float(optimizer_group["weight_decay"]),
            "lr_step_size": scheduler.step_size,
            "lr_gamma": scheduler.gamma,
            "focal_alpha": args.focal_alpha,
            "focal_gamma": args.focal_gamma,
            "best_metric": args.best_metric,
            "max_grad_norm": args.max_grad_norm,
            "seed": effective_seed,
            "rng_continued_from_checkpoint": (
                resume_state.rng_restored if resume_state is not None else False
            ),
        },
    }


def _effective_model_config(
    model_config: OutfitEncoderConfig,
    resume_state: CPResumeState | None,
) -> dict[str, Any]:
    current_config = asdict(model_config)
    if resume_state is None or resume_state.run_config is None:
        return current_config
    previous_config = resume_state.run_config.get("model")
    if not isinstance(previous_config, dict):
        return current_config
    effective_config = dict(current_config)
    effective_config.update(previous_config)
    effective_config["image_fine_tune_mode"] = model_config.image_fine_tune_mode
    return effective_config


def _effective_seed(
    current_seed: int,
    resume_state: CPResumeState | None,
) -> int:
    if (
        resume_state is None
        or not resume_state.rng_restored
        or resume_state.run_config is None
    ):
        return current_seed
    training_config = resume_state.run_config.get("training")
    if not isinstance(training_config, dict):
        return current_seed
    previous_seed = training_config.get("seed")
    return previous_seed if isinstance(previous_seed, int) else current_seed


def _current_lr(optimizer: Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _dataset_cache_location(cache_dir: Path | None) -> Path | str:
    if cache_dir is not None:
        return cache_dir.resolve()
    try:
        from datasets import config as datasets_config
    except ImportError:
        return "Hugging Face default datasets cache"
    return Path(datasets_config.HF_DATASETS_CACHE)


def _hub_cache_location(cache_dir: Path | None) -> Path | str:
    if cache_dir is not None:
        return cache_dir.resolve()
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return "Hugging Face default hub cache"
    return Path(HF_HUB_CACHE)


if __name__ == "__main__":
    main()
