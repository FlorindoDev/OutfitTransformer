import argparse
import random
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from data import collate_compatibility, load_polyvore_compatibility_dataset
from model import BinaryFocalLoss, CompatibilityPredictor, OutfitEncoderConfig
from training import CPBatchProgress, CPCheckpointInfo, CPEpochMetrics, train_cp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train OutfitTransformer for compatibility prediction",
    )
    parser.add_argument(
        "--variant",
        choices=("nondisjoint", "disjoint"),
        default="disjoint",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-step-size", type=int, default=10)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--max-grad-norm", type=float, default=None)
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
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/cp_epochs"),
        help="directory for one checkpoint per epoch",
    )
    parser.add_argument(
        "--text-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--no-pretrained-image",
        action="store_true",
        help="do not initialize ResNet-18 with ImageNet weights",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    _seed_everything(args.seed)

    _print_startup(args)
    train_dataset = load_polyvore_compatibility_dataset(
        variant=args.variant,
        split="train",
        cache_dir=args.cache_dir,
    )
    validation_dataset = load_polyvore_compatibility_dataset(
        variant=args.variant,
        split="validation",
        cache_dir=args.cache_dir,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": str(args.device).startswith("cuda"),
        "collate_fn": collate_compatibility,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_options,
    )
    _print_data_summary(
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader,
    )

    config = OutfitEncoderConfig(
        text_model_name=args.text_model,
        pretrained_image_encoder=not args.no_pretrained_image,
    )
    model = CompatibilityPredictor(config=config)
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
    progress_interval = args.log_interval if args.log_interval > 0 else None

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
        progress_interval=progress_interval,
        on_batch_end=_print_batch if progress_interval is not None else None,
        on_checkpoint_saved=_print_checkpoint,
        on_epoch_end=lambda epoch, train_metrics, validation_metrics: _print_epoch(
            epoch,
            train_metrics,
            validation_metrics,
            optimizer,
        ),
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.lr_step_size <= 0:
        raise ValueError("--lr-step-size must be positive")
    if not 0.0 < args.lr_gamma <= 1.0:
        raise ValueError("--lr-gamma must be in (0, 1]")
    if not 0.0 <= args.focal_alpha <= 1.0:
        raise ValueError("--focal-alpha must be in [0, 1]")
    if args.focal_gamma < 0.0:
        raise ValueError("--focal-gamma must be non-negative")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if args.log_interval < 0:
        raise ValueError("--log-interval must be non-negative")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _print_epoch(
    epoch: int,
    train_metrics: CPEpochMetrics,
    validation_metrics: CPEpochMetrics | None,
    optimizer: Adam,
) -> None:
    message = (
        f"epoch={epoch} "
        f"train_loss={train_metrics.loss:.6f} "
        f"train_accuracy={train_metrics.accuracy:.4f} "
        f"train_examples={train_metrics.examples} "
        f"lr={_current_lr(optimizer):.8f}"
    )
    if validation_metrics is not None:
        message += (
            f" val_loss={validation_metrics.loss:.6f}"
            f" val_accuracy={validation_metrics.accuracy:.4f}"
            f" val_examples={validation_metrics.examples}"
        )
    print(message)


def _print_startup(args: argparse.Namespace) -> None:
    print("training=compatibility_prediction")
    print(f"dataset=mvasil/polyvore-outfits variant={args.variant}")
    print(f"dataset_cache={_dataset_cache_location(args.cache_dir)}")
    print(f"hub_cache={_hub_cache_location(args.cache_dir)}")
    print(f"device={args.device} seed={args.seed}")
    print(f"checkpoint_best={args.checkpoint.resolve()}")
    print(f"checkpoint_epochs={args.checkpoint_dir.resolve()}")
    if args.log_interval == 0:
        print("batch_logs=disabled")
    else:
        print(f"batch_logs=every_{args.log_interval}_batches")


def _print_data_summary(
    train_dataset: object,
    validation_dataset: object,
    train_loader: DataLoader,
    validation_loader: DataLoader,
) -> None:
    print(
        f"train_examples={len(train_dataset)} "
        f"validation_examples={len(validation_dataset)}"
    )
    print(
        f"train_batches={len(train_loader)} "
        f"validation_batches={len(validation_loader)} "
        f"batch_size={train_loader.batch_size}"
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
        f"monitored_loss={info.monitored_loss:.6f}"
    )


def _current_lr(optimizer: Adam) -> float:
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
