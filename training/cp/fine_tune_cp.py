from __future__ import annotations

import argparse
import random
from collections.abc import Mapping, Sequence, Sized
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from data import CompatibilityBatch, create_polyvore_compatibility_loader
from model import (
    IMAGE_FINE_TUNE_MODES,
    BinaryFocalLoss,
    CompatibilityPredictor,
    OutfitEncoderConfig,
)
from .checkpointing import CPResumeState, load_cp_training_checkpoint
from .early_stopping import (
    CPEarlyStoppingStatus,
    create_early_stopping_config,
)
from .fine_tuning import (
    CPFineTuneCheckpoint,
    CPFineTuneOptimizerConfig,
    CP_OPTIMIZER_NAMES,
    build_cp_fine_tune_optimizer,
    optimizer_learning_rates,
)
from .optimization import (
    CPSchedulerParameters,
    SCHEDULER_NAMES,
    create_cp_scheduler,
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


LossName = Literal["focal", "bce"]
LOSS_NAMES: tuple[LossName, ...] = ("focal", "bce")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start a new CP fine-tuning phase or resume an interrupted "
            "fine-tuning run exactly"
        ),
    )
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--source-checkpoint",
        type=Path,
        help="model checkpoint used to start a new fine-tuning phase",
    )
    checkpoint_group.add_argument(
        "--resume",
        type=Path,
        help="fine-tuning checkpoint whose full training state must be restored",
    )
    parser.add_argument(
        "--additional-epochs",
        type=int,
        default=None,
        help=(
            "epochs in a new phase; default 10. On resume the value saved in "
            "the checkpoint is authoritative"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "output directory; default checkpoints/cp_fine_tune for a new "
            "phase, or the resumed checkpoint run directory"
        ),
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
        "--transformer-learning-rate",
        type=float,
        default=None,
        help="Transformer LR; default uses --learning-rate",
    )
    parser.add_argument(
        "--resnet-learning-rate",
        "--image-backbone-learning-rate",
        dest="image_backbone_learning_rate",
        type=float,
        default=None,
        help="trainable ResNet feature-block LR; default uses --learning-rate",
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
    parser.add_argument(
        "--transformer-scheduler",
        choices=SCHEDULER_NAMES,
        default=None,
        help="Transformer scheduler; default uses --scheduler",
    )
    parser.add_argument(
        "--resnet-scheduler",
        choices=SCHEDULER_NAMES,
        default=None,
        help="ResNet feature-block scheduler; default uses --scheduler",
    )
    parser.add_argument("--lr-step-size", type=int, default=10)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument(
        "--transformer-lr-step-size",
        type=int,
        default=None,
        help="Transformer StepLR period; default uses --lr-step-size",
    )
    parser.add_argument(
        "--transformer-lr-gamma",
        type=float,
        default=None,
        help="Transformer StepLR factor; default uses --lr-gamma",
    )
    parser.add_argument(
        "--transformer-min-learning-rate",
        type=float,
        default=None,
        help="Transformer cosine minimum LR; default uses --min-learning-rate",
    )
    parser.add_argument(
        "--resnet-lr-step-size",
        type=int,
        default=None,
        help="ResNet StepLR period; default uses --lr-step-size",
    )
    parser.add_argument(
        "--resnet-lr-gamma",
        type=float,
        default=None,
        help="ResNet StepLR factor; default uses --lr-gamma",
    )
    parser.add_argument(
        "--resnet-min-learning-rate",
        type=float,
        default=None,
        help="ResNet cosine minimum LR; default uses --min-learning-rate",
    )
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
        "--early-stopping-patience",
        type=int,
        default=None,
        help=(
            "stop after N validation epochs without sufficient improvement; "
            "disabled when omitted"
        ),
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="minimum best-metric improvement that resets patience",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=_optional_float,
        default=None,
        help="gradient clipping norm; disabled by default, or pass 'none'",
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
    checkpoint_path = _selected_checkpoint_path(args)
    source = CPFineTuneCheckpoint.load(checkpoint_path, device="cpu")
    if args.resume is None:
        if args.additional_epochs is None:
            args.additional_epochs = 10
    else:
        _apply_resume_configuration(args, source)
    args.output_dir = _resolve_output_directory(args, checkpoint_path)
    _validate_args(args)
    _validate_output_directory(args.output_dir, is_resume=args.resume is not None)
    _seed_everything(args.seed)

    variant = _resolve_variant(args.variant, source)
    model_config = source.model_config(
        image_fine_tune_mode=args.image_fine_tune_mode,
        dropout=args.dropout,
        text_model_name=args.text_model,
    )
    model = CompatibilityPredictor(config=model_config)
    model.to(args.device)
    if args.resume is None:
        source.load_weights(model)

    train_loader, validation_loader = _create_loaders(args, variant)
    criterion = _create_criterion(args)
    optimizer_config = CPFineTuneOptimizerConfig(
        name=args.optimizer,
        learning_rate=args.learning_rate,
        transformer_learning_rate=_separate_transformer_learning_rate(args),
        image_backbone_learning_rate=args.image_backbone_learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        eps=args.adam_eps,
    )
    optimizer = build_cp_fine_tune_optimizer(model, optimizer_config)
    scheduler = _create_scheduler(args, optimizer)

    resume_state = _load_resume_state(args, model, optimizer, scheduler)
    start_epoch = source.epoch + 1
    final_epoch = _resolve_final_epoch(args, source)
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
        resume_state=resume_state,
    )
    _print_data_summary(train_loader, validation_loader)
    _print_parameter_summary(model)

    plotter = None if args.no_plots else CPHistoryPlotter(plot_directory)
    progress_interval = args.log_interval if args.log_interval > 0 else None
    early_stopping = create_early_stopping_config(
        metric=args.best_metric,
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
    )
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
        initial_best_value=(
            None
            if resume_state is None
            else resume_state.best_selection_value
        ),
        initial_history=(
            CPTrainingHistory()
            if resume_state is None
            else resume_state.history
        ),
        checkpoint_metadata=_checkpoint_metadata(
            args,
            source,
            variant=variant,
            model_config=model_config,
            optimizer_config=optimizer_config,
            resume_state=resume_state,
        ),
        progress_interval=progress_interval,
        early_stopping=early_stopping,
        on_batch_end=_print_batch if progress_interval is not None else None,
        on_checkpoint_saved=_print_checkpoint,
        on_history_updated=(
            None if plotter is None else partial(_plot_and_report, plotter)
        ),
        on_epoch_end=partial(_print_epoch, optimizer=optimizer),
        on_early_stopping=_print_early_stopping,
    )


def _optional_float(value: str) -> float | None:
    if value.lower() == "none":
        return None
    try:
        return float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a number or 'none'") from error


def _selected_checkpoint_path(args: argparse.Namespace) -> Path:
    checkpoint_path = args.resume or args.source_checkpoint
    if checkpoint_path is None:
        raise ValueError("either --source-checkpoint or --resume is required")
    return checkpoint_path


def _separate_transformer_learning_rate(
    args: argparse.Namespace,
) -> float | None:
    if args.transformer_learning_rate is not None:
        return args.transformer_learning_rate
    if _has_transformer_scheduler_override(args):
        return args.learning_rate
    return None


def _scheduler_parameters(
    args: argparse.Namespace,
) -> tuple[
    CPSchedulerParameters,
    CPSchedulerParameters | None,
    CPSchedulerParameters | None,
]:
    parameters = CPSchedulerParameters(
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
        min_learning_rate=args.min_learning_rate,
    )
    return (
        parameters,
        parameters.with_optional_overrides(
            step_size=args.transformer_lr_step_size,
            gamma=args.transformer_lr_gamma,
            min_learning_rate=args.transformer_min_learning_rate,
        ),
        parameters.with_optional_overrides(
            step_size=args.resnet_lr_step_size,
            gamma=args.resnet_lr_gamma,
            min_learning_rate=args.resnet_min_learning_rate,
        ),
    )


def _validate_scheduler_configuration(args: argparse.Namespace) -> None:
    parameters, transformer_parameters, resnet_parameters = (
        _scheduler_parameters(args)
    )
    transformer_parameters = transformer_parameters or parameters
    resnet_parameters = resnet_parameters or parameters
    configurations = (
        ("base", args.scheduler, parameters, args.learning_rate),
        (
            "Transformer",
            args.transformer_scheduler or args.scheduler,
            transformer_parameters,
            args.transformer_learning_rate or args.learning_rate,
        ),
        (
            "ResNet",
            args.resnet_scheduler or args.scheduler,
            resnet_parameters,
            args.image_backbone_learning_rate or args.learning_rate,
        ),
    )
    for group_name, scheduler_name, group_parameters, learning_rate in configurations:
        group_parameters.validate()
        if (
            scheduler_name == "cosine"
            and group_parameters.min_learning_rate > learning_rate
        ):
            raise ValueError(
                f"{group_name} minimum learning rate cannot exceed its initial "
                "learning rate"
            )


def _has_transformer_scheduler_override(args: argparse.Namespace) -> bool:
    return args.transformer_scheduler is not None or any(
        value is not None
        for value in (
            args.transformer_lr_step_size,
            args.transformer_lr_gamma,
            args.transformer_min_learning_rate,
        )
    )


def _has_resnet_scheduler_override(args: argparse.Namespace) -> bool:
    return args.resnet_scheduler is not None or any(
        value is not None
        for value in (
            args.resnet_lr_step_size,
            args.resnet_lr_gamma,
            args.resnet_min_learning_rate,
        )
    )


def _require_resume_section(
    run_config: Mapping[str, Any],
    section_name: str,
) -> Mapping[str, Any]:
    section = run_config.get(section_name)
    if not isinstance(section, Mapping):
        raise ValueError(
            "exact fine-tuning resume requires complete training metadata: "
            f"missing or invalid '{section_name}' section"
        )
    return section


def _apply_resume_configuration(
    args: argparse.Namespace,
    checkpoint: CPFineTuneCheckpoint,
) -> None:
    run_config = checkpoint.run_config
    if run_config is None:
        raise ValueError("exact fine-tuning resume requires checkpoint run_config")
    training = _require_resume_section(run_config, "training")
    model = _require_resume_section(run_config, "model")
    dataset = _require_resume_section(run_config, "dataset")
    _require_resume_section(run_config, "fine_tuning")
    if training.get("phase") != "fine_tune":
        raise ValueError("--resume requires a fine-tuning checkpoint")

    saved_arguments = {
        "additional_epochs": training.get("additional_epochs"),
        "batch_size": training.get("batch_size"),
        "optimizer": training.get("optimizer"),
        "learning_rate": training.get("learning_rate"),
        "transformer_learning_rate": training.get(
            "transformer_learning_rate"
        ),
        "image_backbone_learning_rate": training.get(
            "resnet_learning_rate",
            training.get("image_backbone_learning_rate"),
        ),
        "weight_decay": training.get("weight_decay"),
        "adam_beta1": training.get("adam_beta1"),
        "adam_beta2": training.get("adam_beta2"),
        "adam_eps": training.get("adam_eps"),
        "scheduler": training.get("scheduler"),
        "transformer_scheduler": training.get("transformer_scheduler"),
        "resnet_scheduler": training.get("resnet_scheduler"),
        "lr_step_size": training.get("lr_step_size"),
        "lr_gamma": training.get("lr_gamma"),
        "min_learning_rate": training.get("min_learning_rate"),
        "transformer_lr_step_size": training.get(
            "transformer_lr_step_size"
        ),
        "transformer_lr_gamma": training.get("transformer_lr_gamma"),
        "transformer_min_learning_rate": training.get(
            "transformer_min_learning_rate"
        ),
        "resnet_lr_step_size": training.get("resnet_lr_step_size"),
        "resnet_lr_gamma": training.get("resnet_lr_gamma"),
        "resnet_min_learning_rate": training.get(
            "resnet_min_learning_rate"
        ),
        "loss": training.get("loss"),
        "focal_alpha": training.get("focal_alpha"),
        "focal_gamma": training.get("focal_gamma"),
        "best_metric": training.get("best_metric"),
        "early_stopping_patience": training.get("early_stopping_patience"),
        "early_stopping_min_delta": training.get(
            "early_stopping_min_delta"
        ),
        "max_grad_norm": training.get("max_grad_norm"),
        "seed": training.get("seed"),
        "image_fine_tune_mode": model.get("image_fine_tune_mode"),
        "dropout": model.get("dropout"),
        "text_model": model.get("text_model_name"),
        "variant": dataset.get("variant"),
    }
    missing = sorted(
        name
        for name, value in saved_arguments.items()
        if value is None
        and name
        not in {
            "early_stopping_patience",
            "focal_alpha",
            "max_grad_norm",
            "transformer_learning_rate",
            "transformer_scheduler",
            "resnet_scheduler",
            "transformer_lr_step_size",
            "transformer_lr_gamma",
            "transformer_min_learning_rate",
            "resnet_lr_step_size",
            "resnet_lr_gamma",
            "resnet_min_learning_rate",
        }
    )
    if missing:
        raise ValueError(
            "fine-tuning checkpoint lacks resume settings: "
            + ", ".join(missing)
        )
    for name, value in saved_arguments.items():
        setattr(args, name, value)


def _resolve_output_directory(
    args: argparse.Namespace,
    checkpoint_path: Path,
) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    if args.resume is None:
        return Path("checkpoints/cp_fine_tune")
    if checkpoint_path.parent.name == "epochs":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def _validate_args(args: argparse.Namespace) -> None:
    if args.additional_epochs is None or args.additional_epochs <= 0:
        raise ValueError("--additional-epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    optimizer_config = CPFineTuneOptimizerConfig(
        name=args.optimizer,
        learning_rate=args.learning_rate,
        transformer_learning_rate=_separate_transformer_learning_rate(args),
        image_backbone_learning_rate=args.image_backbone_learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        eps=args.adam_eps,
    )
    optimizer_config.validate()
    _validate_scheduler_configuration(args)
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
    if (
        _has_resnet_scheduler_override(args)
        and args.image_fine_tune_mode == "fc_only"
    ):
        raise ValueError(
            "ResNet LR/scheduler overrides require trainable feature blocks"
        )
    create_early_stopping_config(
        metric=args.best_metric,
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
    )


def _validate_output_directory(
    output_directory: Path,
    *,
    is_resume: bool,
) -> None:
    if output_directory.exists() and not output_directory.is_dir():
        raise NotADirectoryError(
            f"fine-tune output must be a directory: {output_directory}"
        )
    if (
        not is_resume
        and output_directory.exists()
        and any(output_directory.rglob("*.pt"))
    ):
        raise FileExistsError(
            f"fine-tune output already contains checkpoints: {output_directory}"
        )


def _load_resume_state(
    args: argparse.Namespace,
    model: CompatibilityPredictor,
    optimizer: Optimizer,
    scheduler: Any | None,
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
    if not state.history_complete:
        raise ValueError("exact fine-tuning resume requires complete history")
    if not state.rng_restored:
        raise ValueError("exact fine-tuning resume requires saved RNG state")
    if state.selection_metric != args.best_metric:
        raise ValueError(
            "fine-tuning resume selection metric differs from checkpoint"
        )
    return state


def _resolve_final_epoch(
    args: argparse.Namespace,
    checkpoint: CPFineTuneCheckpoint,
) -> int:
    if args.resume is None:
        return checkpoint.epoch + args.additional_epochs
    run_config = checkpoint.run_config
    if run_config is None:
        raise ValueError("fine-tuning resume checkpoint lacks run_config")
    fine_tuning = _require_resume_section(run_config, "fine_tuning")
    source_epoch = fine_tuning.get("source_epoch")
    if not isinstance(source_epoch, int) or isinstance(source_epoch, bool):
        raise ValueError("fine-tuning resume checkpoint lacks source_epoch")
    final_epoch = source_epoch + args.additional_epochs
    if final_epoch < checkpoint.epoch + 1:
        raise ValueError(
            "fine-tuning run already reached its configured final epoch"
        )
    return final_epoch


def _resolve_variant(
    requested_variant: str | None,
    source: CPFineTuneCheckpoint,
) -> str:
    if requested_variant is not None:
        return requested_variant
    run_config = source.run_config
    if run_config is not None:
        dataset_config = run_config.get("dataset")
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
    parameters, transformer_parameters, resnet_parameters = (
        _scheduler_parameters(args)
    )
    return create_cp_scheduler(
        optimizer,
        scheduler=args.scheduler,
        transformer_scheduler=args.transformer_scheduler,
        resnet_scheduler=args.resnet_scheduler,
        total_epochs=args.additional_epochs,
        parameters=parameters,
        transformer_parameters=transformer_parameters,
        resnet_parameters=resnet_parameters,
    )


def _checkpoint_metadata(
    args: argparse.Namespace,
    source: CPFineTuneCheckpoint,
    *,
    variant: str,
    model_config: OutfitEncoderConfig,
    optimizer_config: CPFineTuneOptimizerConfig,
    resume_state: CPResumeState | None,
) -> dict[str, Any]:
    if resume_state is not None:
        run_config = resume_state.run_config
        if run_config is None:
            raise ValueError("fine-tuning resume checkpoint lacks run_config")
        metadata = deepcopy(run_config)
        fine_tuning = metadata.get("fine_tuning")
        if not isinstance(fine_tuning, dict):
            raise ValueError("fine-tuning resume metadata is invalid")
        fine_tuning.update(
            {
                "resume_checkpoint": str(args.resume.resolve()),
                "resume_epoch": resume_state.epoch,
                "optimizer_state_reused": True,
                "scheduler_state_reused": True,
                "history_reused": True,
                "rng_state_reused": True,
            }
        )
        return metadata
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
            "transformer_learning_rate": (
                optimizer_config.transformer_learning_rate
            ),
            "effective_transformer_learning_rate": (
                optimizer_config.resolved_transformer_learning_rate
            ),
            "resnet_learning_rate": (
                optimizer_config.resolved_image_backbone_learning_rate
            ),
            "image_backbone_learning_rate": (
                optimizer_config.resolved_image_backbone_learning_rate
            ),
            "weight_decay": optimizer_config.weight_decay,
            "adam_beta1": optimizer_config.beta1,
            "adam_beta2": optimizer_config.beta2,
            "adam_eps": optimizer_config.eps,
            "scheduler": args.scheduler,
            "transformer_scheduler": args.transformer_scheduler,
            "resnet_scheduler": args.resnet_scheduler,
            "lr_step_size": args.lr_step_size,
            "lr_gamma": args.lr_gamma,
            "min_learning_rate": args.min_learning_rate,
            "transformer_lr_step_size": args.transformer_lr_step_size,
            "transformer_lr_gamma": args.transformer_lr_gamma,
            "transformer_min_learning_rate": (
                args.transformer_min_learning_rate
            ),
            "resnet_lr_step_size": args.resnet_lr_step_size,
            "resnet_lr_gamma": args.resnet_lr_gamma,
            "resnet_min_learning_rate": args.resnet_min_learning_rate,
            "loss": args.loss,
            "focal_alpha": args.focal_alpha,
            "focal_gamma": args.focal_gamma,
            "best_metric": args.best_metric,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
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
    resume_state: CPResumeState | None,
) -> None:
    print("training=compatibility_prediction_fine_tune")
    if resume_state is None:
        print(
            f"source_checkpoint={source.path.resolve()} "
            f"source_epoch={source.epoch}"
        )
    else:
        print(
            f"resume_checkpoint={source.path.resolve()} "
            f"resume_epoch={resume_state.epoch}"
        )
    print(f"epochs={start_epoch}-{final_epoch} dataset_variant={variant}")
    print(f"device={args.device} seed={args.seed} batch_size={args.batch_size}")
    print(
        f"resnet_fine_tune_mode={args.image_fine_tune_mode} "
        f"optimizer={args.optimizer} scheduler={args.scheduler} "
        f"transformer_scheduler={args.transformer_scheduler or args.scheduler} "
        f"resnet_scheduler={args.resnet_scheduler or args.scheduler} "
        f"loss={args.loss}"
    )
    print(
        "learning_rates="
        + ",".join(
            f"{name}:{value:.8g}"
            for name, value in optimizer_learning_rates(optimizer).items()
        )
    )
    print(f"best_metric={args.best_metric} output_dir={args.output_dir.resolve()}")
    if args.early_stopping_patience is None:
        print("early_stopping=disabled")
    else:
        print(
            f"early_stopping=enabled metric={args.best_metric} "
            f"patience={args.early_stopping_patience} "
            f"min_delta={args.early_stopping_min_delta}"
        )
    if resume_state is None:
        print("source_optimizer_scheduler_history_rng=discarded")
    else:
        print("resume_optimizer_scheduler_history_rng=restored")


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


def _print_early_stopping(status: CPEarlyStoppingStatus) -> None:
    print(
        f"early_stopping=triggered epoch={status.epoch} "
        f"metric={status.metric} value={status.value:.6f} "
        f"best={status.best_value:.6f} "
        f"epochs_without_improvement={status.epochs_without_improvement} "
        f"patience={status.patience}"
    )


def _plot_and_report(
    plotter: CPHistoryPlotter,
    history: CPTrainingHistory,
) -> None:
    paths = plotter(history)
    print("plots_saved=" + ",".join(str(path.resolve()) for path in paths))


if __name__ == "__main__":
    main()
