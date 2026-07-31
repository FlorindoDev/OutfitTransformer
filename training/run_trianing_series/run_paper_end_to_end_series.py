"""Run controlled end-to-end CP experiments around the paper setup."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PaperEndToEndStage:
    number: int
    name: str
    changed_parameter: str
    seed: int = 42
    dropout: float = 0.1
    norm_first: bool = False
    weight_decay: float = 0.0
    max_grad_norm: float | None = None
    focal_alpha: float = 0.25

    @property
    def directory_name(self) -> str:
        return f"{self.number:02d}_{self.name}"

    @property
    def normalization_flag(self) -> str:
        return "--pre-norm" if self.norm_first else "--post-norm"


PAPER_END_TO_END_STAGES: tuple[PaperEndToEndStage, ...] = (
    PaperEndToEndStage(
        number=1,
        name="paper_standard_defaults",
        changed_parameter="baseline",
    ),
    PaperEndToEndStage(
        number=2,
        name="seed_7",
        changed_parameter="seed",
        seed=7,
    ),
    PaperEndToEndStage(
        number=3,
        name="seed_123",
        changed_parameter="seed",
        seed=123,
    ),
    PaperEndToEndStage(
        number=4,
        name="dropout_0",
        changed_parameter="dropout",
        dropout=0.0,
    ),
    PaperEndToEndStage(
        number=5,
        name="dropout_02",
        changed_parameter="dropout",
        dropout=0.2,
    ),
    PaperEndToEndStage(
        number=6,
        name="pre_norm",
        changed_parameter="norm_first",
        norm_first=True,
    ),
    PaperEndToEndStage(
        number=7,
        name="weight_decay_1e4",
        changed_parameter="weight_decay",
        weight_decay=1e-4,
    ),
    PaperEndToEndStage(
        number=8,
        name="grad_clip_1",
        changed_parameter="max_grad_norm",
        max_grad_norm=1.0,
    ),
    PaperEndToEndStage(
        number=9,
        name="focal_alpha_05",
        changed_parameter="focal_alpha",
        focal_alpha=0.5,
    ),
)
STAGE_NUMBERS = tuple(stage.number for stage in PAPER_END_TO_END_STAGES)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run independent end-to-end CP experiments that preserve every "
            "training parameter reported by the OutfitTransformer paper"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("checkpoints/cp_paper_end_to_end_series"),
    )
    parser.add_argument(
        "--variant",
        choices=("nondisjoint", "disjoint"),
        default="disjoint",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--text-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--stages",
        nargs="+",
        type=int,
        choices=STAGE_NUMBERS,
        default=None,
        help="stage numbers to run; default runs all stages",
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
    _validate_stage_definitions(PAPER_END_TO_END_STAGES)
    output_root = _resolve_output_root(args.output_root)
    stages = _select_stages(args.stages)
    if not args.dry_run:
        _validate_output_directories(stages, output_root)

    print("series=paper_end_to_end_ablation")
    print(f"independent_stages={len(stages)} output_root={output_root}")
    for stage in stages:
        command = _build_stage_command(stage, args, output_root)
        print(
            f"stage={stage.number} name={stage.name} "
            f"changed_parameter={stage.changed_parameter}"
        )
        print(f"command={subprocess.list2cmdline(command)}")
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _build_stage_command(
    stage: PaperEndToEndStage,
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[str, ...]:
    output_directory = output_root / stage.directory_name
    max_grad_norm = (
        "none" if stage.max_grad_norm is None else str(stage.max_grad_norm)
    )
    command = [
        sys.executable,
        "-m",
        "training.cp.train_cp",
        "--variant",
        args.variant,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        "1e-5",
        "--weight-decay",
        str(stage.weight_decay),
        "--lr-step-size",
        "10",
        "--lr-gamma",
        "0.5",
        "--focal-alpha",
        str(stage.focal_alpha),
        "--focal-gamma",
        "2.0",
        "--dropout",
        str(stage.dropout),
        stage.normalization_flag,
        "--max-grad-norm",
        max_grad_norm,
        "--seed",
        str(stage.seed),
        "--best-metric",
        "val_auc",
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--early-stopping-min-delta",
        str(args.early_stopping_min_delta),
        "--image-fine-tune-mode",
        "full",
        "--text-model",
        args.text_model,
        "--workers",
        str(args.workers),
        "--log-interval",
        str(args.log_interval),
        "--checkpoint",
        str(output_directory / "best.pt"),
        "--checkpoint-dir",
        str(output_directory / "epochs"),
        "--plot-dir",
        str(output_directory / "plots"),
    ]
    if args.device is not None:
        command.extend(("--device", args.device))
    if args.cache_dir is not None:
        command.extend(("--cache-dir", str(args.cache_dir)))
    if args.no_plots:
        command.append("--no-plots")
    return tuple(command)


def _select_stages(
    requested_numbers: Sequence[int] | None,
) -> tuple[PaperEndToEndStage, ...]:
    if requested_numbers is None:
        return PAPER_END_TO_END_STAGES
    requested = set(requested_numbers)
    selected = tuple(
        stage for stage in PAPER_END_TO_END_STAGES if stage.number in requested
    )
    if len(selected) != len(requested):
        unknown = sorted(requested.difference(STAGE_NUMBERS))
        raise ValueError(f"unknown stage numbers: {unknown}")
    return selected


def _resolve_output_root(output_root: Path) -> Path:
    return output_root if output_root.is_absolute() else PROJECT_ROOT / output_root


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.early_stopping_patience <= 0:
        raise ValueError("--early-stopping-patience must be positive")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError("--early-stopping-min-delta must be non-negative")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if args.log_interval < 0:
        raise ValueError("--log-interval must be non-negative")


def _validate_stage_definitions(
    stages: Sequence[PaperEndToEndStage],
) -> None:
    if not stages:
        raise ValueError("paper end-to-end series requires at least one stage")
    numbers = [stage.number for stage in stages]
    names = [stage.name for stage in stages]
    if len(set(numbers)) != len(numbers):
        raise ValueError("paper end-to-end stage numbers must be unique")
    if len(set(names)) != len(names):
        raise ValueError("paper end-to-end stage names must be unique")
    if numbers != list(range(1, len(stages) + 1)):
        raise ValueError("paper end-to-end stage numbers must be contiguous")
    for stage in stages:
        if stage.seed < 0:
            raise ValueError(f"invalid seed in stage {stage.number}")
        if not 0.0 <= stage.dropout < 1.0:
            raise ValueError(f"invalid dropout in stage {stage.number}")
        if stage.weight_decay < 0.0:
            raise ValueError(f"invalid weight decay in stage {stage.number}")
        if stage.max_grad_norm is not None and stage.max_grad_norm <= 0.0:
            raise ValueError(f"invalid gradient clipping in stage {stage.number}")
        if not 0.0 <= stage.focal_alpha <= 1.0:
            raise ValueError(f"invalid focal alpha in stage {stage.number}")

    baseline = stages[0]
    experimental_fields = tuple(
        field.name
        for field in fields(PaperEndToEndStage)
        if field.name not in {"number", "name", "changed_parameter"}
    )
    for stage in stages[1:]:
        changed = tuple(
            name
            for name in experimental_fields
            if getattr(stage, name) != getattr(baseline, name)
        )
        if changed != (stage.changed_parameter,):
            raise ValueError(
                f"stage {stage.number} must change only "
                f"{stage.changed_parameter}; changed={changed}"
            )


def _validate_output_directories(
    stages: Sequence[PaperEndToEndStage],
    output_root: Path,
) -> None:
    for stage in stages:
        output_directory = output_root / stage.directory_name
        if output_directory.exists() and any(output_directory.rglob("*.pt")):
            raise FileExistsError(
                f"stage output already contains checkpoints: {output_directory}"
            )


if __name__ == "__main__":
    main()
