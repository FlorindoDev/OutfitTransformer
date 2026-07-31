"""Run the CP training experiment series."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CPExperimentStage:
    number: int
    name: str
    output_directory: Path
    command: tuple[str, ...]

    @property
    def best_checkpoint(self) -> Path:
        return self.output_directory / "best.pt"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the paper CP baseline and the three-stage progressive "
            "ResNet fine-tuning sequence"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("checkpoints/cp_training_series"),
    )
    parser.add_argument(
        "--variant",
        choices=("nondisjoint", "disjoint"),
        default="disjoint",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--paper-epochs", type=int, default=30)
    parser.add_argument("--fc-only-max-epochs", type=int, default=12)
    parser.add_argument("--fc-only-patience", type=int, default=3)
    parser.add_argument("--layer4-max-epochs", type=int, default=30)
    parser.add_argument("--layer4-patience", type=int, default=4)
    parser.add_argument("--full-epochs", type=int, default=4)
    parser.add_argument("--full-patience", type=int, default=2)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--task-learning-rate", type=float, default=1e-5)
    parser.add_argument("--layer4-learning-rate", type=float, default=1e-6)
    parser.add_argument("--full-task-learning-rate", type=float, default=3e-6)
    parser.add_argument("--full-backbone-learning-rate", type=float, default=3e-7)
    parser.add_argument("--progressive-weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--text-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--start-stage",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
        help="resume the series from a completed stage boundary",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without starting training",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    stages = _build_stages(args)
    selected_stages = tuple(
        stage for stage in stages if stage.number >= args.start_stage
    )
    if not args.dry_run:
        _validate_artifacts(stages, selected_stages, args.start_stage)

    for stage in selected_stages:
        print(f"stage={stage.number} name={stage.name}")
        print(f"command={subprocess.list2cmdline(stage.command)}")
        if not args.dry_run:
            subprocess.run(
                stage.command,
                cwd=PROJECT_ROOT,
                check=True,
            )


def _build_stages(args: argparse.Namespace) -> tuple[CPExperimentStage, ...]:
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    paper = _training_stage(
        number=1,
        name="paper_end_to_end",
        output_directory=output_root / "01_paper_end_to_end",
        epochs=args.paper_epochs,
        image_fine_tune_mode="full",
        learning_rate=1e-5,
        weight_decay=0.0,
        args=args,
    )
    fc_only = _training_stage(
        number=2,
        name="fc_only_base",
        output_directory=output_root / "02_fc_only_base",
        epochs=args.fc_only_max_epochs,
        image_fine_tune_mode="fc_only",
        learning_rate=args.task_learning_rate,
        weight_decay=args.progressive_weight_decay,
        args=args,
        early_stopping_patience=args.fc_only_patience,
    )
    layer4 = _fine_tuning_stage(
        number=3,
        name="layer4_plateau",
        output_directory=output_root / "03_layer4_plateau",
        source_checkpoint=fc_only.best_checkpoint,
        additional_epochs=args.layer4_max_epochs,
        image_fine_tune_mode="fc_and_layer4",
        learning_rate=args.task_learning_rate,
        backbone_learning_rate=args.layer4_learning_rate,
        early_stopping_patience=args.layer4_patience,
        args=args,
    )
    full = _fine_tuning_stage(
        number=4,
        name="full_low_lr",
        output_directory=output_root / "04_full_low_lr",
        source_checkpoint=layer4.best_checkpoint,
        additional_epochs=args.full_epochs,
        image_fine_tune_mode="full",
        learning_rate=args.full_task_learning_rate,
        backbone_learning_rate=args.full_backbone_learning_rate,
        early_stopping_patience=args.full_patience,
        args=args,
    )
    return paper, fc_only, layer4, full


def _training_stage(
    *,
    number: int,
    name: str,
    output_directory: Path,
    epochs: int,
    image_fine_tune_mode: str,
    learning_rate: float,
    weight_decay: float,
    args: argparse.Namespace,
    early_stopping_patience: int | None = None,
) -> CPExperimentStage:
    command = [
        sys.executable,
        "-m",
        "training.cp.train_cp",
        "--variant",
        args.variant,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(learning_rate),
        "--weight-decay",
        str(weight_decay),
        "--lr-step-size",
        "10",
        "--lr-gamma",
        "0.5",
        "--best-metric",
        "val_auc",
        "--image-fine-tune-mode",
        image_fine_tune_mode,
        "--checkpoint",
        str(output_directory / "best.pt"),
        "--checkpoint-dir",
        str(output_directory / "epochs"),
        "--plot-dir",
        str(output_directory / "plots"),
    ]
    _append_shared_arguments(command, args)
    _append_early_stopping(
        command,
        early_stopping_patience,
        args.early_stopping_min_delta,
    )
    return CPExperimentStage(
        number=number,
        name=name,
        output_directory=output_directory,
        command=tuple(command),
    )


def _fine_tuning_stage(
    *,
    number: int,
    name: str,
    output_directory: Path,
    source_checkpoint: Path,
    additional_epochs: int,
    image_fine_tune_mode: str,
    learning_rate: float,
    backbone_learning_rate: float,
    early_stopping_patience: int,
    args: argparse.Namespace,
) -> CPExperimentStage:
    command = [
        sys.executable,
        "-m",
        "training.cp.fine_tune_cp",
        "--source-checkpoint",
        str(source_checkpoint),
        "--additional-epochs",
        str(additional_epochs),
        "--output-dir",
        str(output_directory),
        "--variant",
        args.variant,
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(learning_rate),
        "--image-backbone-learning-rate",
        str(backbone_learning_rate),
        "--optimizer",
        "adam",
        "--weight-decay",
        str(args.progressive_weight_decay),
        "--scheduler",
        "cosine",
        "--loss",
        "focal",
        "--best-metric",
        "val_auc",
        "--image-fine-tune-mode",
        image_fine_tune_mode,
    ]
    _append_shared_arguments(command, args, include_text_model=False)
    _append_early_stopping(
        command,
        early_stopping_patience,
        args.early_stopping_min_delta,
    )
    return CPExperimentStage(
        number=number,
        name=name,
        output_directory=output_directory,
        command=tuple(command),
    )


def _append_shared_arguments(
    command: list[str],
    args: argparse.Namespace,
    *,
    include_text_model: bool = True,
) -> None:
    command.extend(
        [
            "--workers",
            str(args.workers),
            "--seed",
            str(args.seed),
            "--log-interval",
            str(args.log_interval),
        ]
    )
    if include_text_model:
        command.extend(["--text-model", args.text_model])
    if args.device is not None:
        command.extend(["--device", args.device])
    if args.cache_dir is not None:
        command.extend(["--cache-dir", str(args.cache_dir)])
    if args.no_plots:
        command.append("--no-plots")


def _append_early_stopping(
    command: list[str],
    patience: int | None,
    min_delta: float,
) -> None:
    if patience is None:
        return
    command.extend(
        [
            "--early-stopping-patience",
            str(patience),
            "--early-stopping-min-delta",
            str(min_delta),
        ]
    )


def _validate_args(args: argparse.Namespace) -> None:
    positive_integers = {
        "batch size": args.batch_size,
        "paper epochs": args.paper_epochs,
        "FC-only max epochs": args.fc_only_max_epochs,
        "FC-only patience": args.fc_only_patience,
        "layer4 max epochs": args.layer4_max_epochs,
        "layer4 patience": args.layer4_patience,
        "full epochs": args.full_epochs,
        "full patience": args.full_patience,
    }
    for name, value in positive_integers.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    positive_rates = {
        "task learning rate": args.task_learning_rate,
        "layer4 learning rate": args.layer4_learning_rate,
        "full task learning rate": args.full_task_learning_rate,
        "full backbone learning rate": args.full_backbone_learning_rate,
    }
    for name, value in positive_rates.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if args.progressive_weight_decay < 0.0:
        raise ValueError("progressive weight decay must be non-negative")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError("early stopping min delta must be non-negative")
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    if args.log_interval < 0:
        raise ValueError("log interval must be non-negative")


def _validate_artifacts(
    all_stages: tuple[CPExperimentStage, ...],
    selected_stages: tuple[CPExperimentStage, ...],
    start_stage: int,
) -> None:
    if start_stage in (3, 4):
        prerequisite = all_stages[start_stage - 2].best_checkpoint
        if not prerequisite.is_file():
            raise FileNotFoundError(
                f"stage {start_stage} requires checkpoint: {prerequisite}"
            )

    for stage in selected_stages:
        if (
            stage.output_directory.exists()
            and any(stage.output_directory.rglob("*.pt"))
        ):
            raise FileExistsError(
                f"stage output already contains checkpoints: "
                f"{stage.output_directory}"
            )


if __name__ == "__main__":
    main()
