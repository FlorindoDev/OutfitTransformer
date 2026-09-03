"""Atomic checkpoint I/O shared by training jobs."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch import nn


def save_checkpoint(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically save one PyTorch checkpoint."""
    selected_path = Path(path)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = selected_path.with_name(f"{selected_path.name}.tmp")
    try:
        torch.save(dict(payload), temporary_path)
        temporary_path.replace(selected_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return selected_path


def copy_checkpoint(source: str | Path, destination: str | Path) -> Path:
    """Atomically copy an epoch checkpoint to the best-model path."""
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {source_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_name(f"{destination_path.name}.tmp")
    try:
        shutil.copyfile(source_path, temporary_path)
        temporary_path.replace(destination_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination_path


def load_model_weights(
    model: nn.Module,
    path: str | Path,
    *,
    map_location: torch.device | str = "cpu",
) -> None:
    """Load only model weights from a checkpoint or bare state dictionary."""
    state_dict = load_checkpoint_state_dict(path, map_location=map_location)
    model.load_state_dict(state_dict, strict=True)


def load_checkpoint_state_dict(
    path: str | Path,
    *,
    map_location: torch.device | str = "cpu",
) -> dict[str, Tensor]:
    """Load and validate a model state dictionary from one checkpoint."""
    selected_path = Path(path)
    if not selected_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {selected_path}")
    payload = torch.load(
        selected_path,
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must contain a mapping")
    state_dict = payload.get("model_state_dict", payload)
    if not isinstance(state_dict, Mapping):
        raise TypeError("model_state_dict must be a mapping")

    validated: dict[str, Tensor] = {}
    for raw_name, value in state_dict.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise TypeError("model_state_dict keys must be non-empty strings")
        if not isinstance(value, Tensor):
            raise TypeError(
                f"model_state_dict value for {raw_name!r} must be a tensor"
            )
        validated[raw_name] = value
    if not validated:
        raise ValueError("model_state_dict cannot be empty")
    return validated


def write_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically write human-readable JSON metadata."""
    selected_path = Path(path)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = selected_path.with_name(f"{selected_path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(selected_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return selected_path
