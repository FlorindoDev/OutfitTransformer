from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .types import CPTrainingHistory


class CPHistoryPlotter:
    """Save cumulative CP training charts after each completed epoch."""

    def __init__(self, output_directory: str | Path, *, dpi: int = 120) -> None:
        if dpi <= 0:
            raise ValueError("dpi must be positive")
        self._output_directory = Path(output_directory)
        self._dpi = dpi

    @property
    def output_directory(self) -> Path:
        return self._output_directory

    def __call__(self, history: CPTrainingHistory) -> tuple[Path, ...]:
        epoch = history.last_epoch
        if epoch is None:
            raise ValueError("cannot plot an empty CP training history")
        self._output_directory.mkdir(parents=True, exist_ok=True)

        saved = [
            self._plot_loss(history, epoch),
            self._plot_accuracy(history, epoch),
        ]
        validation_plot = self._plot_validation_accuracy_auc(history, epoch)
        if validation_plot is not None:
            saved.append(validation_plot)
        return tuple(saved)

    def _plot_loss(self, history: CPTrainingHistory, epoch: int) -> Path:
        series: list[tuple[Sequence[int], Sequence[float], str]] = [
            (history.epochs, [item.loss for item in history.train], "Train loss")
        ]
        if history.validation:
            series.append(
                (
                    history.validation_epochs,
                    [item.loss for item in history.validation],
                    "Validation loss",
                )
            )
        return self._save_chart(
            epoch=epoch,
            name="loss",
            title="CP loss",
            y_label="Loss",
            series=series,
            y_limits=None,
        )

    def _plot_accuracy(self, history: CPTrainingHistory, epoch: int) -> Path:
        series: list[tuple[Sequence[int], Sequence[float], str]] = [
            (
                history.epochs,
                [item.accuracy for item in history.train],
                "Train accuracy",
            )
        ]
        if history.validation:
            series.append(
                (
                    history.validation_epochs,
                    [item.accuracy for item in history.validation],
                    "Validation accuracy",
                )
            )
        return self._save_chart(
            epoch=epoch,
            name="accuracy",
            title="CP accuracy",
            y_label="Accuracy",
            series=series,
            y_limits=(0.0, 1.0),
        )

    def _plot_validation_accuracy_auc(
        self,
        history: CPTrainingHistory,
        epoch: int,
    ) -> Path | None:
        auc_points = [
            (validation_epoch, metrics)
            for validation_epoch, metrics in zip(
                history.validation_epochs,
                history.validation,
                strict=True,
            )
            if metrics.auc is not None
        ]
        if not auc_points:
            return None
        auc_epochs = [point[0] for point in auc_points]
        auc_values = [point[1].auc for point in auc_points]
        return self._save_chart(
            epoch=epoch,
            name="validation_accuracy_auc",
            title="CP validation accuracy and ROC AUC",
            y_label="Score",
            series=[
                (
                    history.validation_epochs,
                    [item.accuracy for item in history.validation],
                    "Validation accuracy",
                ),
                (auc_epochs, auc_values, "Validation ROC AUC"),
            ],
            y_limits=(0.0, 1.0),
        )

    def _save_chart(
        self,
        *,
        epoch: int,
        name: str,
        title: str,
        y_label: str,
        series: Sequence[tuple[Sequence[int], Sequence[float], str]],
        y_limits: tuple[float, float] | None,
    ) -> Path:
        pyplot = _load_pyplot()
        figure, axis = pyplot.subplots(figsize=(8, 5))
        try:
            for x_values, y_values, label in series:
                axis.plot(x_values, y_values, marker="o", label=label)
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.set_ylabel(y_label)
            axis.set_xticks(
                sorted({value for values, _, _ in series for value in values})
            )
            if y_limits is not None:
                axis.set_ylim(*y_limits)
            axis.grid(True, alpha=0.3)
            axis.legend()
            figure.tight_layout()

            path = self._output_directory / f"cp_{name}_epoch_{epoch:03d}.png"
            _atomic_save_figure(figure, path, self._dpi)
            return path
        finally:
            pyplot.close(figure)


def _load_pyplot() -> Any:
    try:
        os.environ.setdefault(
            "MPLCONFIGDIR",
            str(Path(tempfile.gettempdir()) / "outfit-transformer-matplotlib"),
        )
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot
    except ImportError as error:
        raise ImportError(
            "CP plotting requires matplotlib; install project requirements"
        ) from error
    return pyplot


def _atomic_save_figure(figure: Any, path: Path, dpi: int) -> None:
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        figure.savefig(temporary_path, format="png", dpi=dpi)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
