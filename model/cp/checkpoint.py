from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn


def load_cp_checkpoint(
    path: str | Path,
    model: nn.Module,
    device: torch.device | str,
) -> dict[str, Any]:
    checkpoint = read_cp_checkpoint(path, device)
    load_cp_checkpoint_weights(checkpoint, model)
    return checkpoint


def read_cp_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Read and validate a CP checkpoint without constructing a model."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"CP checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("CP checkpoint must be a dictionary")

    _model_state(checkpoint)
    return checkpoint


def load_cp_checkpoint_weights(
    checkpoint: Mapping[str, Any],
    model: nn.Module,
) -> None:
    """Load every model tensor, rejecting missing or unexpected parameters."""
    model.load_state_dict(_model_state(checkpoint), strict=True)


def _model_state(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise ValueError("CP checkpoint missing model_state_dict")
    return model_state
