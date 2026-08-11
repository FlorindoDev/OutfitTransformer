"""Cumulative plots matching the existing CP checkpoint layout."""

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
    """Save the four cumulative CP charts for the current epoch."""
    selected_dir = Path(output_dir)
    selected_dir.mkdir(parents=True, exist_ok=True)
    epochs = history["epochs"]
    train = history["train"]
    validation = history["validation"]

    _save_two_series(
        epochs,
        [metrics["loss"] for metrics in train],
        [metrics["loss"] for metrics in validation],
        first_label="Train loss",
        second_label="Validation loss",
        title="CP loss",
        ylabel="Loss",
        path=selected_dir / f"cp_loss_epoch_{epoch:03d}.png",
    )
    _save_two_series(
        epochs,
        [metrics["accuracy"] for metrics in train],
        [metrics["accuracy"] for metrics in validation],
        first_label="Train accuracy",
        second_label="Validation accuracy",
        title="CP accuracy",
        ylabel="Accuracy",
        path=selected_dir / f"cp_accuracy_epoch_{epoch:03d}.png",
        normalized_score_axis=True,
    )
    _save_two_series(
        epochs,
        [metrics["auc"] for metrics in train],
        [metrics["auc"] for metrics in validation],
        first_label="Train ROC AUC",
        second_label="Validation ROC AUC",
        title="CP ROC AUC",
        ylabel="ROC AUC",
        path=selected_dir / f"cp_auc_epoch_{epoch:03d}.png",
        normalized_score_axis=True,
    )
    _save_two_series(
        epochs,
        [metrics["accuracy"] for metrics in validation],
        [metrics["auc"] for metrics in validation],
        first_label="Validation accuracy",
        second_label="Validation ROC AUC",
        title="CP validation accuracy and ROC AUC",
        ylabel="Score",
        path=(
            selected_dir
            / f"cp_validation_accuracy_auc_epoch_{epoch:03d}.png"
        ),
        normalized_score_axis=True,
    )


def _save_two_series(
    epochs: list[int],
    first_values: list[float],
    second_values: list[float],
    *,
    first_label: str,
    second_label: str,
    title: str,
    ylabel: str,
    path: Path,
    normalized_score_axis: bool = False,
) -> None:
    figure, axes = plt.subplots(figsize=(10, 6))
    axes.plot(epochs, first_values, marker="o", label=first_label)
    axes.plot(epochs, second_values, marker="o", label=second_label)
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
