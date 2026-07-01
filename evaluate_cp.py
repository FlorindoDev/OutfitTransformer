import argparse
from collections.abc import Sized
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data import collate_compatibility, load_polyvore_compatibility_dataset
from model import (
    BinaryFocalLoss,
    CompatibilityPredictor,
    OutfitEncoderConfig,
    load_cp_checkpoint,
)
from training import CPBatchProgress, run_cp_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a CP checkpoint on the Polyvore test split",
    )
    parser.add_argument(
        "--variant",
        choices=("nondisjoint", "disjoint"),
        default="disjoint",
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
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--focal-alpha", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="print progress every N batches; use 0 to disable",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)

    dataset = load_polyvore_compatibility_dataset(
        variant=args.variant,
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

    config = OutfitEncoderConfig(
        text_model_name=args.text_model,
        pretrained_image_encoder=False,
    )
    model = CompatibilityPredictor(config=config).to(args.device)
    checkpoint = load_cp_checkpoint(
        args.checkpoint,
        model,
        args.device,
    )
    criterion = BinaryFocalLoss(
        alpha=args.focal_alpha,
        gamma=args.focal_gamma,
    )
    progress_interval = args.log_interval if args.log_interval > 0 else None

    _print_startup(args, dataset, batches, checkpoint)
    metrics = run_cp_epoch(
        model,
        batches,
        criterion,
        args.device,
        phase="test",
        progress_interval=progress_interval,
        on_batch_end=_print_batch if progress_interval is not None else None,
    )
    print(
        f"test_loss={metrics.loss:.6f} "
        f"test_accuracy={metrics.accuracy:.4f} "
        f"test_examples={metrics.examples}"
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if not 0.0 <= args.focal_alpha <= 1.0:
        raise ValueError("--focal-alpha must be in [0, 1]")
    if args.focal_gamma < 0.0:
        raise ValueError("--focal-gamma must be non-negative")
    if args.log_interval < 0:
        raise ValueError("--log-interval must be non-negative")


def _print_startup(
    args: argparse.Namespace,
    dataset: Sized,
    batches: DataLoader,
    checkpoint: dict[str, Any],
) -> None:
    print("evaluation=compatibility_prediction split=test")
    print(f"variant={args.variant} device={args.device}")
    print(f"checkpoint={args.checkpoint.resolve()}")
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
