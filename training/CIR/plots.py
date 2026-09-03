"""Cumulative plots matching the CP checkpoint layout for CIR metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402

SCORE_TICKS = tuple(index / 10 for index in range(1, 11))


def save_cumulative_plots(
    history: dict[str, Any],
    output_dir: str | Path,
    epoch: int,
) -> None:
    """Save cumulative CIR loss and validation-ranking charts."""
    selected_dir = Path(output_dir)
    selected_dir.mkdir(parents=True, exist_ok=True)
    epochs = history["epochs"]
    train = history["train"]
    validation = history["validation"]

    _save_series(
        epochs,
        (
            ("Train loss", [metrics["loss"] for metrics in train]),
            (
                "Validation loss",
                [metrics["loss"] for metrics in validation],
            ),
        ),
        title="CIR loss",
        ylabel="Loss",
        path=selected_dir / f"cir_loss_epoch_{epoch:03d}.png",
    )
    _save_validation_metric(
        epochs,
        validation,
        key="fitb_accuracy",
        label="Validation FITB accuracy",
        title="CIR validation FITB accuracy",
        path=selected_dir / f"cir_fitb_accuracy_epoch_{epoch:03d}.png",
    )
    _save_validation_metric(
        epochs,
        validation,
        key="mrr",
        label="Validation MRR",
        title="CIR validation MRR",
        path=selected_dir / f"cir_mrr_epoch_{epoch:03d}.png",
    )
    _save_validation_metric(
        epochs,
        validation,
        key="recall_at_2",
        label="Validation Recall@2",
        title="CIR validation Recall@2",
        path=selected_dir / f"cir_recall_at_2_epoch_{epoch:03d}.png",
    )
    _save_series(
        epochs,
        (
            (
                "FITB accuracy",
                [metrics["fitb_accuracy"] for metrics in validation],
            ),
            ("MRR", [metrics["mrr"] for metrics in validation]),
            (
                "Recall@2",
                [metrics["recall_at_2"] for metrics in validation],
            ),
        ),
        title="CIR validation ranking metrics",
        ylabel="Score",
        path=(
            selected_dir
            / f"cir_validation_metrics_epoch_{epoch:03d}.png"
        ),
        normalized_score_axis=True,
    )


def _save_validation_metric(
    epochs: list[int],
    validation: list[dict[str, Any]],
    *,
    key: str,
    label: str,
    title: str,
    path: Path,
) -> None:
    _save_series(
        epochs,
        ((label, [metrics[key] for metrics in validation]),),
        title=title,
        ylabel="Score",
        path=path,
        normalized_score_axis=True,
    )


def _save_series(
    epochs: list[int],
    series: tuple[tuple[str, list[float]], ...],
    *,
    title: str,
    ylabel: str,
    path: Path,
    normalized_score_axis: bool = False,
) -> None:
    figure, axes = plt.subplots(figsize=(10, 6))
    for label, values in series:
        axes.plot(epochs, values, marker="o", label=label)
    axes.set_title(title)
    axes.set_xlabel("Epoch")
    axes.set_ylabel(ylabel)
    axes.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    if normalized_score_axis:
        axes.set_ylim(0.0, 1.0)
        axes.set_yticks(SCORE_TICKS)
    axes.grid(alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
