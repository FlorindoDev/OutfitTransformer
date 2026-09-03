"""Optional distributed runtime for CIR training launched with torchrun."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch import distributed as dist

from training.common import resolve_device

from .config import CIRTrainingConfig


@dataclass(frozen=True)
class DistributedContext:
    """Resolved device and process identity for one CIR worker."""

    device: torch.device
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str | None = None

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def initialize_distributed(
    config: CIRTrainingConfig,
) -> DistributedContext:
    """Initialize a torchrun environment or resolve a single-process device."""
    if not config.ddp:
        return DistributedContext(device=resolve_device(config.device))

    rank = _required_environment_int("RANK")
    world_size = _required_environment_int("WORLD_SIZE")
    local_rank = _required_environment_int("LOCAL_RANK")
    if world_size < 2:
        raise ValueError("--ddp requires WORLD_SIZE of at least 2")

    if torch.cuda.is_available() and config.device != "cpu":
        device_count = torch.cuda.device_count()
        if not 0 <= local_rank < device_count:
            raise ValueError(
                f"LOCAL_RANK {local_rank} exceeds {device_count} CUDA devices"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        if config.device not in {"auto", "cpu"}:
            raise ValueError("CPU DDP accepts only --device auto or cpu")
        device = torch.device("cpu")
        backend = "gloo"

    if dist.is_initialized():
        raise RuntimeError("distributed process group is already initialized")
    dist.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(
        device=device,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        backend=backend,
    )


def close_distributed(context: DistributedContext) -> None:
    """Close a process group created for this run."""
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def _required_environment_int(name: str) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        raise RuntimeError(
            f"--ddp requires {name}; launch with torchrun"
        )
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value
