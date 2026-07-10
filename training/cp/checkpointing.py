from __future__ import annotations

import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from .types import (
    CPCheckpointInfo,
    CPEpochMetrics,
    CPTrainingHistory,
)


CHECKPOINT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class CPResumeState:
    epoch: int
    best_monitored_loss: float
    history: CPTrainingHistory
    history_complete: bool
    rng_restored: bool
    run_config: dict[str, Any] | None


class CPCheckpointManager:
    """Persist epoch and best checkpoints while owning best-loss state."""

    def __init__(
        self,
        *,
        best_path: str | Path | None,
        epoch_directory: str | Path | None,
        total_epochs: int,
        initial_best_loss: float = float("inf"),
        run_config: Mapping[str, Any] | None = None,
    ) -> None:
        if total_epochs <= 0:
            raise ValueError("total_epochs must be positive")
        self._best_path = Path(best_path) if best_path is not None else None
        self._epoch_directory = (
            Path(epoch_directory) if epoch_directory is not None else None
        )
        self._total_epochs = total_epochs
        self._best_loss = float(initial_best_loss)
        self._run_config = (
            _normalize_run_config(run_config) if run_config is not None else None
        )

    @property
    def best_loss(self) -> float:
        return self._best_loss

    def save(
        self,
        *,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Any | None,
        monitored_loss: float,
        train_metrics: CPEpochMetrics,
        validation_metrics: CPEpochMetrics | None,
        history: CPTrainingHistory,
    ) -> tuple[CPCheckpointInfo, ...]:
        is_best = monitored_loss < self._best_loss
        next_best_loss = min(self._best_loss, monitored_loss)
        checkpoint = _build_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            monitored_loss=monitored_loss,
            best_monitored_loss=next_best_loss,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            history=history,
            run_config=self._run_config,
        )

        saved: list[CPCheckpointInfo] = []
        if self._epoch_directory is not None:
            epoch_path = _epoch_checkpoint_path(
                self._epoch_directory,
                epoch,
                self._total_epochs,
            )
            _atomic_torch_save(checkpoint, epoch_path)
            saved.append(
                CPCheckpointInfo(
                    epoch=epoch,
                    kind="epoch",
                    path=epoch_path,
                    monitored_loss=monitored_loss,
                )
            )

        if self._best_path is not None and is_best:
            _atomic_torch_save(checkpoint, self._best_path)
            saved.append(
                CPCheckpointInfo(
                    epoch=epoch,
                    kind="best",
                    path=self._best_path,
                    monitored_loss=monitored_loss,
                )
            )

        self._best_loss = next_best_loss
        return tuple(saved)


def load_cp_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any | None,
    device: torch.device | str,
) -> CPResumeState:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("resume checkpoint must be a dictionary")
    _validate_checkpoint_schema(checkpoint)

    _load_required_state(checkpoint, "model_state_dict", model.load_state_dict)
    _load_required_state(
        checkpoint,
        "optimizer_state_dict",
        optimizer.load_state_dict,
    )
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = _checkpoint_int(checkpoint, "epoch")
    monitored_loss = _checkpoint_float(checkpoint, "monitored_loss")
    best_loss_value = checkpoint.get("best_monitored_loss", monitored_loss)
    if not isinstance(best_loss_value, int | float):
        raise ValueError("checkpoint best_monitored_loss must be numeric")

    history, history_complete = _load_history(checkpoint, epoch)
    run_config_value = checkpoint.get("run_config")
    run_config = (
        dict(run_config_value) if isinstance(run_config_value, Mapping) else None
    )
    rng_restored = _restore_rng_state(checkpoint.get("rng_state"))
    return CPResumeState(
        epoch=epoch,
        best_monitored_loss=float(best_loss_value),
        history=history,
        history_complete=history_complete,
        rng_restored=rng_restored,
        run_config=run_config,
    )


def _build_checkpoint(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any | None,
    monitored_loss: float,
    best_monitored_loss: float,
    train_metrics: CPEpochMetrics,
    validation_metrics: CPEpochMetrics | None,
    history: CPTrainingHistory,
    run_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    checkpoint: dict[str, Any] = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "monitored_loss": monitored_loss,
        "best_monitored_loss": best_monitored_loss,
        "train_metrics": train_metrics.to_payload(),
        "validation_metrics": (
            validation_metrics.to_payload()
            if validation_metrics is not None
            else None
        ),
        "training_history": history.to_payload(),
        "rng_state": _capture_rng_state(),
    }
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if run_config is not None:
        checkpoint["run_config"] = dict(run_config)
    return checkpoint


def _load_history(
    checkpoint: Mapping[str, Any],
    epoch: int,
) -> tuple[CPTrainingHistory, bool]:
    payload = checkpoint.get("training_history")
    if isinstance(payload, Mapping):
        history = CPTrainingHistory.from_payload(payload)
        if history.last_epoch != epoch:
            raise ValueError("checkpoint history does not end at checkpoint epoch")
        return history, True

    train_metrics = _optional_metrics(checkpoint.get("train_metrics"))
    validation_metrics = _optional_metrics(
        checkpoint.get("validation_metrics")
    )
    history = CPTrainingHistory()
    if train_metrics is not None:
        history = history.append(epoch, train_metrics, validation_metrics)
    return history, False


def _validate_checkpoint_schema(checkpoint: Mapping[str, Any]) -> None:
    version = checkpoint.get("checkpoint_schema_version")
    if version is None:
        return
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError("checkpoint_schema_version must be an integer")
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema version {version}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}"
        )


def _optional_metrics(value: Any) -> CPEpochMetrics | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint metrics must be a dictionary or None")
    return CPEpochMetrics.from_payload(value)


def _load_required_state(
    checkpoint: Mapping[str, Any],
    key: str,
    load_state: Any,
) -> None:
    state = checkpoint.get(key)
    if state is None:
        raise ValueError(f"resume checkpoint missing {key}")
    load_state(state)


def _checkpoint_int(checkpoint: Mapping[str, Any], key: str) -> int:
    value = checkpoint.get(key)
    if not isinstance(value, int):
        raise ValueError(f"resume checkpoint missing integer {key}")
    return value


def _checkpoint_float(checkpoint: Mapping[str, Any], key: str) -> float:
    value = checkpoint.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"resume checkpoint missing numeric {key}")
    return float(value)


def _epoch_checkpoint_path(directory: Path, epoch: int, epochs: int) -> Path:
    width = max(3, len(str(epochs)))
    return directory / f"cp_epoch_{epoch:0{width}d}.pt"


def _atomic_torch_save(checkpoint: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(dict(checkpoint), temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _normalize_run_config(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalize_config_value(config)
    if not isinstance(normalized, dict):
        raise TypeError("run_config must be a string-keyed mapping")
    return normalized


def _normalize_config_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise TypeError("run_config mapping keys must be strings")
            normalized[key] = _normalize_config_value(nested_value)
        return normalized
    if isinstance(value, list | tuple):
        return [_normalize_config_value(item) for item in value]
    raise TypeError(
        "run_config values must be primitives, paths, sequences, or mappings"
    )


def _capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": _tuples_to_lists(random.getstate()),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
    }


def _restore_rng_state(value: Any) -> bool:
    if value is None:
        return False
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint rng_state must be a dictionary")

    python_state = value.get("python")
    torch_cpu_state = value.get("torch_cpu")
    numpy_state = value.get("numpy")
    if python_state is None or not isinstance(torch_cpu_state, torch.Tensor):
        raise ValueError("checkpoint rng_state is incomplete")
    if not isinstance(numpy_state, Mapping):
        raise ValueError("checkpoint NumPy RNG state is missing")

    random.setstate(_lists_to_tuples(python_state))
    torch.set_rng_state(torch_cpu_state.cpu())
    _restore_cuda_rng_states(value.get("torch_cuda"))
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    return True


def _restore_cuda_rng_states(value: Any) -> None:
    if not torch.cuda.is_available() or value is None:
        return
    if not isinstance(value, list | tuple) or not all(
        isinstance(state, torch.Tensor) for state in value
    ):
        raise ValueError("checkpoint CUDA RNG state must contain tensors")
    for device, state in enumerate(value[: torch.cuda.device_count()]):
        torch.cuda.set_rng_state(state.cpu(), device=device)


def _tuples_to_lists(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_tuples_to_lists(item) for item in value]
    return value


def _lists_to_tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_lists_to_tuples(item) for item in value)
    return value
