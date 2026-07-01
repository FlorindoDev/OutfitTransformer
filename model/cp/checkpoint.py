from pathlib import Path
from typing import Any

import torch
from torch import nn


def load_cp_checkpoint(
    path: str | Path,
    model: nn.Module,
    device: torch.device | str,
) -> dict[str, Any]:
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

    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError("CP checkpoint missing model_state_dict")
    model.load_state_dict(model_state)
    return checkpoint
