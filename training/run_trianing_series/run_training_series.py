"""Run warm-up and four dependent ResNet fine-tuning phases."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TrainingPhase:
    number: int
    name: str
    image_fine_tune_mode: str
    epochs: int
    learning_rate: float
    resnet_fc_learning_rate: float
    resnet_learning_rate: float | None
    scheduler: str
    resnet_fc_scheduler: str
    min_learning_rate: float
    transformer_min_learning_rate: float | None
    resnet_min_learning_rate: float | None
    optimizer: str
    reuse_optimizer_state: bool
    early_stopping_patience: int

    @property
    def directory_name(self) -> str:
        return f"{self.number:02d}_{self.name}"


TRAINING_PHASES: tuple[TrainingPhase, ...] = (
    TrainingPhase(
        number=1,
        name="warmup",
        image_fine_tune_mode="fc_only",
        epochs=3,
        learning_rate=3e-6,
        resnet_fc_learning_rate=3e-6,
        resnet_learning_rate=None,
        scheduler="none",
        resnet_fc_scheduler="none",
        min_learning_rate=0.0,
        transformer_min_learning_rate=None,
        resnet_min_learning_rate=None,
        optimizer="adam",
        reuse_optimizer_state=False,
        early_stopping_patience=4,
    ),
    TrainingPhase(
        number=2,
        name="fc_only",
        image_fine_tune_mode="fc_only",
        epochs=12,
        learning_rate=1e-5,
        resnet_fc_learning_rate=3e-5,
        resnet_learning_rate=None,
        scheduler="cosine",
        resnet_fc_scheduler="cosine",
        min_learning_rate=3e-6,
        transformer_min_learning_rate=1e-6,
        resnet_min_learning_rate=None,
        optimizer="adamw",
        reuse_optimizer_state=False,
        early_stopping_patience=4,
    ),
    TrainingPhase(
        number=3,
        name="fc_and_layer4",
        image_fine_tune_mode="fc_and_layer4",
        epochs=10,
        learning_rate=5e-6,
        resnet_fc_learning_rate=1e-5,
        resnet_learning_rate=1e-6,
        scheduler="cosine",
        resnet_fc_scheduler="cosine",
        min_learning_rate=1e-6,
        transformer_min_learning_rate=5e-7,
        resnet_min_learning_rate=1e-7,
        optimizer="adamw",
        reuse_optimizer_state=True,
        early_stopping_patience=4,
    ),
    TrainingPhase(
        number=4,
        name="full_backbone",
        image_fine_tune_mode="full",
        epochs=12,
        learning_rate=2e-6,
        resnet_fc_learning_rate=5e-6,
        resnet_learning_rate=5e-7,
        scheduler="cosine",
        resnet_fc_scheduler="cosine",
        min_learning_rate=5e-7,
        transformer_min_learning_rate=2e-7,
        resnet_min_learning_rate=5e-8,
        optimizer="adamw",
        reuse_optimizer_state=True,
        early_stopping_patience=4,
    ),
    TrainingPhase(
        number=5,
        name="full_refine",
        image_fine_tune_mode="full",
        epochs=6,
        learning_rate=2e-6,
        resnet_fc_learning_rate=5e-6,
        resnet_learning_rate=5e-7,
        scheduler="cosine",
        resnet_fc_scheduler="cosine",
        min_learning_rate=5e-7,
        transformer_min_learning_rate=2e-7,
        resnet_min_learning_rate=5e-8,
        optimizer="adamw",
        reuse_optimizer_state=False,
        early_stopping_patience=4,
    ),
)
PHASE_NUMBERS = tuple(phase.number for phase in TRAINING_PHASES)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run warm-up and four dependent CP training phases",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("checkpoints/training_series"),
    )
    parser.add_argument(
        "--variant",
        choices=("nondisjoint", "disjoint"),
        default="nondisjoint",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--text-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--phases",
        nargs="+",
        type=int,
        choices=PHASE_NUMBERS,
        default=None,
        help="phases to run; omitted phases must already have best.pt",
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
    _validate_phase_definitions(TRAINING_PHASES)
    output_root = _resolve_output_root(args.output_root)
    phases = _select_phases(args.phases)
    if not args.dry_run:
        _validate_outputs_and_dependencies(phases, output_root)

    print("series=training_series")
    print(f"dependent_phases={len(phases)} output_root={output_root}")
    for phase in phases:
        source_checkpoint = _source_checkpoint(phase, output_root)
        if not args.dry_run and source_checkpoint is not None:
            _require_source_checkpoint(source_checkpoint)
        command = _build_phase_command(phase, args, output_root)
        print(
            f"phase={phase.number} name={phase.name} "
            f"image_mode={phase.image_fine_tune_mode}"
        )
        if source_checkpoint is not None:
            print(f"depends_on={source_checkpoint}")
        print(f"command={subprocess.list2cmdline(command)}")
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _build_phase_command(
    phase: TrainingPhase,
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[str, ...]:
    if phase.number == 1:
        command = _initial_training_command(phase, args, output_root)
    else:
        command = _fine_tuning_command(phase, args, output_root)
    return tuple(_append_runtime_options(command, args))


def _initial_training_command(
    phase: TrainingPhase,
    args: argparse.Namespace,
    output_root: Path,
) -> list[str]:
    output_directory = output_root / phase.directory_name
    return [
        sys.executable,
        "-m",
        "training.cp.train_cp",
        "--variant",
        args.variant,
        "--epochs",
        str(phase.epochs),
        *_common_training_arguments(phase, args),
        "--post-norm",
        "--checkpoint",
        str(output_directory / "best.pt"),
        "--checkpoint-dir",
        str(output_directory / "epochs"),
        "--plot-dir",
        str(output_directory / "plots"),
    ]


def _fine_tuning_command(
    phase: TrainingPhase,
    args: argparse.Namespace,
    output_root: Path,
) -> list[str]:
    source_checkpoint = _source_checkpoint(phase, output_root)
    if source_checkpoint is None:
        raise ValueError("fine-tuning phase requires a source checkpoint")
    command = [
        sys.executable,
        "-m",
        "training.cp.fine_tune_cp",
        "--source-checkpoint",
        str(source_checkpoint),
        "--output-dir",
        str(output_root / phase.directory_name),
        "--variant",
        args.variant,
        "--additional-epochs",
        str(phase.epochs),
        "--optimizer",
        phase.optimizer,
        "--transformer-learning-rate",
        str(phase.learning_rate),
        "--transformer-scheduler",
        phase.scheduler,
        *_common_training_arguments(phase, args),
    ]
    if phase.reuse_optimizer_state:
        command.extend(("--optimizer-state-checkpoint", str(source_checkpoint)))
    return command


def _common_training_arguments(
    phase: TrainingPhase,
    args: argparse.Namespace,
) -> list[str]:
    arguments = [
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(phase.learning_rate),
        "--resnet-fc-learning-rate",
        str(phase.resnet_fc_learning_rate),
        "--weight-decay",
        "0.0",
        "--scheduler",
        phase.scheduler,
        "--min-learning-rate",
        str(phase.min_learning_rate),
        "--resnet-fc-scheduler",
        phase.resnet_fc_scheduler,
        "--focal-alpha",
        "0.25",
        "--focal-gamma",
        "2.0",
        "--dropout",
        "0.0",
        "--seed",
        "42",
        "--best-metric",
        "val_auc",
        "--early-stopping-patience",
        str(phase.early_stopping_patience),
        "--early-stopping-min-delta",
        "0.0001",
        "--image-fine-tune-mode",
        phase.image_fine_tune_mode,
        "--text-model",
        args.text_model,
        "--workers",
        str(args.workers),
        "--log-interval",
        str(args.log_interval),
    ]
    if phase.transformer_min_learning_rate is not None:
        arguments.extend(
            (
                "--transformer-min-learning-rate",
                str(phase.transformer_min_learning_rate),
            )
        )
    if phase.resnet_learning_rate is not None:
        arguments.extend(
            (
                "--resnet-learning-rate",
                str(phase.resnet_learning_rate),
                "--resnet-scheduler",
                "cosine",
                "--resnet-min-learning-rate",
                str(phase.resnet_min_learning_rate),
            )
        )
    return arguments


def _append_runtime_options(
    command: list[str],
    args: argparse.Namespace,
) -> list[str]:
    if args.device is not None:
        command.extend(("--device", args.device))
    if args.cache_dir is not None:
        command.extend(("--cache-dir", str(args.cache_dir)))
    if args.no_plots:
        command.append("--no-plots")
    return command


def _source_checkpoint(
    phase: TrainingPhase,
    output_root: Path,
) -> Path | None:
    if phase.number == 1:
        return None
    previous = TRAINING_PHASES[phase.number - 2]
    return output_root / previous.directory_name / "best.pt"


def _select_phases(
    requested_numbers: Sequence[int] | None,
) -> tuple[TrainingPhase, ...]:
    if requested_numbers is None:
        return TRAINING_PHASES
    requested = set(requested_numbers)
    return tuple(phase for phase in TRAINING_PHASES if phase.number in requested)


def _resolve_output_root(output_root: Path) -> Path:
    return output_root if output_root.is_absolute() else PROJECT_ROOT / output_root


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if args.log_interval < 0:
        raise ValueError("--log-interval must be non-negative")


def _validate_phase_definitions(phases: Sequence[TrainingPhase]) -> None:
    if not phases:
        raise ValueError("training series requires at least one phase")
    numbers = [phase.number for phase in phases]
    if numbers != list(range(1, len(phases) + 1)):
        raise ValueError("phase numbers must be unique and contiguous")
    for phase in phases:
        if phase.epochs <= 0:
            raise ValueError(f"invalid epochs in phase {phase.number}")
        if min(phase.learning_rate, phase.resnet_fc_learning_rate) <= 0.0:
            raise ValueError(f"invalid learning rate in phase {phase.number}")
        if (
            phase.resnet_learning_rate is not None
            and phase.resnet_learning_rate <= 0.0
        ):
            raise ValueError(
                f"invalid ResNet learning rate in phase {phase.number}"
            )
        if phase.min_learning_rate < 0.0:
            raise ValueError(
                f"invalid minimum learning rate in phase {phase.number}"
            )
        if (
            phase.transformer_min_learning_rate is not None
            and phase.transformer_min_learning_rate < 0.0
        ):
            raise ValueError(
                f"invalid Transformer minimum LR in phase {phase.number}"
            )
        if (
            phase.resnet_min_learning_rate is not None
            and phase.resnet_min_learning_rate < 0.0
        ):
            raise ValueError(
                f"invalid ResNet minimum LR in phase {phase.number}"
            )
        feature_blocks_trainable = phase.image_fine_tune_mode != "fc_only"
        if feature_blocks_trainable and phase.resnet_learning_rate is None:
            raise ValueError(
                f"phase {phase.number} requires a ResNet feature-block LR"
            )
        if not feature_blocks_trainable and phase.resnet_learning_rate is not None:
            raise ValueError(
                f"phase {phase.number} cannot use a ResNet feature-block LR"
            )
        if (
            not feature_blocks_trainable
            and phase.resnet_min_learning_rate is not None
        ):
            raise ValueError(
                f"phase {phase.number} cannot use a ResNet minimum LR"
            )
        if feature_blocks_trainable and phase.resnet_min_learning_rate is None:
            raise ValueError(
                f"phase {phase.number} requires a ResNet minimum LR"
            )
        _validate_cosine_minimum(
            phase=phase,
            group_name="task/base",
            scheduler=phase.scheduler,
            learning_rate=phase.learning_rate,
            min_learning_rate=phase.min_learning_rate,
        )
        _validate_cosine_minimum(
            phase=phase,
            group_name="Transformer",
            scheduler=phase.scheduler,
            learning_rate=phase.learning_rate,
            min_learning_rate=(
                phase.transformer_min_learning_rate
                if phase.transformer_min_learning_rate is not None
                else phase.min_learning_rate
            ),
        )
        _validate_cosine_minimum(
            phase=phase,
            group_name="ResNet FC",
            scheduler=phase.resnet_fc_scheduler,
            learning_rate=phase.resnet_fc_learning_rate,
            min_learning_rate=phase.min_learning_rate,
        )
        if phase.resnet_learning_rate is not None:
            _validate_cosine_minimum(
                phase=phase,
                group_name="ResNet feature blocks",
                scheduler="cosine",
                learning_rate=phase.resnet_learning_rate,
                min_learning_rate=phase.resnet_min_learning_rate,
            )


def _validate_cosine_minimum(
    *,
    phase: TrainingPhase,
    group_name: str,
    scheduler: str,
    learning_rate: float,
    min_learning_rate: float | None,
) -> None:
    if (
        scheduler == "cosine"
        and min_learning_rate is not None
        and min_learning_rate > learning_rate
    ):
        raise ValueError(
            f"phase {phase.number} {group_name} minimum LR cannot exceed "
            "its initial LR"
        )


def _validate_outputs_and_dependencies(
    phases: Sequence[TrainingPhase],
    output_root: Path,
) -> None:
    selected_numbers = {phase.number for phase in phases}
    for phase in phases:
        output_directory = output_root / phase.directory_name
        if output_directory.exists() and any(output_directory.rglob("*.pt")):
            raise FileExistsError(
                f"phase output already contains checkpoints: {output_directory}"
            )
        source_checkpoint = _source_checkpoint(phase, output_root)
        if (
            source_checkpoint is not None
            and phase.number - 1 not in selected_numbers
        ):
            _require_source_checkpoint(source_checkpoint)


def _require_source_checkpoint(source_checkpoint: Path) -> None:
    if not source_checkpoint.is_file():
        raise FileNotFoundError(
            f"previous phase checkpoint not found: {source_checkpoint}"
        )


if __name__ == "__main__":
    main()
