"""Evaluate a CP checkpoint on the Polyvore test split."""

import argparse
from collections.abc import Mapping, Sequence, Sized
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeGuard

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import collate_compatibility, load_polyvore_compatibility_dataset
from data.polyvore_loader import PolyvoreVariant
from model import (
    BinaryFocalLoss,
    CompatibilityPredictor,
    OutfitEncoderConfig,
    load_cp_checkpoint_weights,
    read_cp_checkpoint,
)
from training import CPBatchProgress, run_cp_epoch


_VARIANTS: tuple[PolyvoreVariant, ...] = ("nondisjoint", "disjoint")
_LOSSES = ("focal", "bce")


@dataclass(frozen=True)
class CPEvaluationSettings:
    variant: PolyvoreVariant
    model: OutfitEncoderConfig
    loss: str
    focal_alpha: float | None
    focal_gamma: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a CP checkpoint on the Polyvore test split",
    )
    parser.add_argument(
        "--variant",
        choices=_VARIANTS,
        default=None,
        help="legacy fallback; modern checkpoints provide the variant",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
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
        "--text-model",
        default=None,
        help="legacy fallback; modern checkpoints provide the text model",
    )
    parser.add_argument(
        "--loss",
        choices=_LOSSES,
        default=None,
        help="legacy fallback; modern checkpoints provide the loss",
    )
    parser.add_argument("--focal-alpha", type=float, default=None)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="print progress every N batches; use 0 to disable",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    checkpoint = read_cp_checkpoint(args.checkpoint, device="cpu")
    settings = _evaluation_settings(checkpoint, args)

    dataset = load_polyvore_compatibility_dataset(
        variant=settings.variant,
        split="test",
        cache_dir=args.cache_dir,
    )
    batches = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=str(args.device).startswith("cuda"),
        collate_fn=collate_compatibility,
    )

    model = CompatibilityPredictor(config=settings.model)
    load_cp_checkpoint_weights(checkpoint, model)
    model.to(args.device)
    criterion = _create_criterion(settings)
    progress_interval = args.log_interval if args.log_interval > 0 else None

    _print_startup(args, settings, dataset, batches, checkpoint)
    del checkpoint
    metrics = run_cp_epoch(
        model,
        batches,
        criterion,
        args.device,
        phase="test",
        progress_interval=progress_interval,
        on_batch_end=_print_batch if progress_interval is not None else None,
        calculate_auc=True,
    )
    if metrics.auc is None:
        raise RuntimeError("CP evaluation did not calculate ROC AUC")
    print(
        f"test_loss={metrics.loss:.6f} "
        f"test_accuracy={metrics.accuracy:.4f} "
        f"test_auc={metrics.auc:.4f} "
        f"test_examples={metrics.examples}"
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if args.focal_alpha is not None and not 0.0 <= args.focal_alpha <= 1.0:
        raise ValueError("--focal-alpha must be in [0, 1]")
    if args.focal_gamma is not None and args.focal_gamma < 0.0:
        raise ValueError("--focal-gamma must be non-negative")
    if args.log_interval < 0:
        raise ValueError("--log-interval must be non-negative")


def _evaluation_settings(
    checkpoint: Mapping[str, Any],
    args: argparse.Namespace,
) -> CPEvaluationSettings:
    run_config = _mapping(checkpoint.get("run_config"), "run_config")
    dataset_config = _mapping(run_config.get("dataset"), "run_config.dataset")
    model_config = _mapping(run_config.get("model"), "run_config.model")
    training_config = _mapping(
        run_config.get("training"),
        "run_config.training",
    )

    variant = _authoritative_value(
        dataset_config,
        "variant",
        args.variant,
        "disjoint",
    )
    if not _is_polyvore_variant(variant):
        raise ValueError(f"checkpoint dataset variant must be one of {_VARIANTS}")

    model = _model_config(model_config, args.text_model)
    loss = _authoritative_value(
        training_config,
        "loss",
        args.loss,
        "focal",
    )
    if loss not in _LOSSES:
        raise ValueError(f"checkpoint loss must be one of {_LOSSES}")

    focal_alpha = _authoritative_value(
        training_config,
        "focal_alpha",
        args.focal_alpha,
        0.5,
    )
    if focal_alpha is not None and (
        isinstance(focal_alpha, bool)
        or not isinstance(focal_alpha, int | float)
        or not 0.0 <= focal_alpha <= 1.0
    ):
        raise ValueError("checkpoint focal_alpha must be in [0, 1] or None")

    focal_gamma = _authoritative_value(
        training_config,
        "focal_gamma",
        args.focal_gamma,
        2.0,
    )
    if (
        isinstance(focal_gamma, bool)
        or not isinstance(focal_gamma, int | float)
        or focal_gamma < 0.0
    ):
        raise ValueError("checkpoint focal_gamma must be non-negative")

    return CPEvaluationSettings(
        variant=variant,
        model=model,
        loss=loss,
        focal_alpha=None if focal_alpha is None else float(focal_alpha),
        focal_gamma=float(focal_gamma),
    )


def _model_config(
    checkpoint_config: Mapping[str, Any],
    text_model_fallback: str | None,
) -> OutfitEncoderConfig:
    supported_names = {field.name for field in fields(OutfitEncoderConfig)}
    unsupported_names = set(checkpoint_config) - supported_names
    if unsupported_names:
        names = ", ".join(sorted(unsupported_names))
        raise ValueError(f"unsupported checkpoint model settings: {names}")

    values = dict(checkpoint_config)
    saved_text_model = values.get("text_model_name")
    if (
        saved_text_model is not None
        and text_model_fallback is not None
        and saved_text_model != text_model_fallback
    ):
        raise ValueError(
            "--text-model conflicts with checkpoint text_model_name; "
            "checkpoint configuration is authoritative"
        )
    if saved_text_model is None and text_model_fallback is not None:
        values["text_model_name"] = text_model_fallback

    # Strict checkpoint loading replaces every ResNet tensor before evaluation.
    values["pretrained_image_encoder"] = False
    try:
        config = OutfitEncoderConfig(**values)
        config.validate()
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"invalid checkpoint model configuration: {error}"
        ) from error
    return config


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint {name} must be a dictionary")
    return value


def _is_polyvore_variant(value: object) -> TypeGuard[PolyvoreVariant]:
    return value in _VARIANTS


def _authoritative_value(
    checkpoint_config: Mapping[str, Any],
    name: str,
    fallback: Any,
    legacy_default: Any,
) -> Any:
    if name not in checkpoint_config:
        return legacy_default if fallback is None else fallback

    checkpoint_value = checkpoint_config[name]
    if fallback is not None and fallback != checkpoint_value:
        raise ValueError(
            f"--{name.replace('_', '-')} conflicts with checkpoint {name}; "
            "checkpoint configuration is authoritative"
        )
    return checkpoint_value


def _create_criterion(settings: CPEvaluationSettings) -> nn.Module:
    if settings.loss == "bce":
        return nn.BCEWithLogitsLoss()
    return BinaryFocalLoss(
        alpha=settings.focal_alpha,
        gamma=settings.focal_gamma,
    )


def _print_startup(
    args: argparse.Namespace,
    settings: CPEvaluationSettings,
    dataset: Sized,
    batches: DataLoader,
    checkpoint: Mapping[str, Any],
) -> None:
    print("evaluation=compatibility_prediction split=test")
    print(f"variant={settings.variant} device={args.device}")
    print(f"checkpoint={args.checkpoint.resolve()}")
    print("checkpoint_model_state=strict_all_tensors")
    print(
        f"loss={settings.loss} focal_alpha={settings.focal_alpha} "
        f"focal_gamma={settings.focal_gamma}"
    )
    if isinstance(checkpoint.get("epoch"), int):
        print(f"checkpoint_epoch={checkpoint['epoch']}")
    print(f"test_examples={len(dataset)} test_batches={len(batches)}")


def _print_batch(progress: CPBatchProgress) -> None:
    batches = progress.batches if progress.batches is not None else "?"
    print(
        f"phase=test "
        f"batch={progress.batch}/{batches} "
        f"running_loss={progress.running_loss:.6f} "
        f"running_accuracy={progress.running_accuracy:.4f} "
        f"examples={progress.examples}"
    )


if __name__ == "__main__":
    main()
