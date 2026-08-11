"""Reproducibility and device selection shared by training jobs."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch and select deterministic cuDNN paths."""
    if seed < 0:
        raise ValueError("seed cannot be negative")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(value: str) -> torch.device:
    """Resolve ``auto`` or validate one explicit PyTorch device."""
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")
    if device.type == "mps" and (
        not hasattr(torch.backends, "mps")
        or not torch.backends.mps.is_available()
    ):
        raise ValueError("MPS device requested but MPS is unavailable")
    return device

