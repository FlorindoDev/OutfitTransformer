"""Initialize weights shared by CP and CIR from a CP checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from training.common import load_checkpoint_state_dict

COMMON_PREFIX = "common."
CP_TASK_EMBEDDING = "cp.task_embedding.embedding"
CIR_TASK_EMBEDDING = "cir.task_embedding.embedding"


@dataclass(frozen=True)
class CPPretrainingReport:
    """Summary of tensors transferred from CP into CIR."""

    checkpoint: Path
    loaded_keys: tuple[str, ...]

    @property
    def loaded_tensor_count(self) -> int:
        return len(self.loaded_keys)


def load_cp_pretrained_weights(
    model: nn.Module,
    path: str | Path,
    *,
    map_location: torch.device | str = "cpu",
) -> CPPretrainingReport:
    """Transfer weights structurally shared by CP and CIR.

    The CIR encoder, retrieval embedding, category embedding and head keep
    their fresh initialization.
    """
    selected_path = Path(path)
    cp_state = load_checkpoint_state_dict(
        selected_path,
        map_location=map_location,
    )
    cir_state = model.state_dict()
    transferred = _build_transfer_state(cp_state, cir_state)
    incompatible = model.load_state_dict(transferred, strict=False)
    if incompatible.unexpected_keys:
        names = ", ".join(sorted(incompatible.unexpected_keys))
        raise RuntimeError(f"unexpected transferred CP keys: {names}")
    return CPPretrainingReport(
        checkpoint=selected_path,
        loaded_keys=tuple(sorted(transferred)),
    )


def _build_transfer_state(
    cp_state: dict[str, Tensor],
    cir_state: dict[str, Tensor],
) -> dict[str, Tensor]:
    required_targets = {
        name
        for name in cir_state
        if name.startswith(COMMON_PREFIX) or name == CIR_TASK_EMBEDDING
    }
    if not required_targets:
        raise ValueError(
            "CIR model does not expose weights shared with CP"
        )

    transferred: dict[str, Tensor] = {}
    missing_sources: list[str] = []
    incompatible_shapes: list[str] = []
    for target_name in sorted(required_targets):
        source_name = _source_name(target_name)
        source_value = _find_source_tensor(cp_state, source_name)
        if source_value is None:
            missing_sources.append(source_name)
            continue
        target_value = cir_state[target_name]
        if source_value.shape != target_value.shape:
            incompatible_shapes.append(
                f"{source_name}: CP {tuple(source_value.shape)} != "
                f"CIR {tuple(target_value.shape)}"
            )
            continue
        transferred[target_name] = source_value

    if missing_sources:
        names = ", ".join(missing_sources)
        raise ValueError(
            "CP checkpoint is incompatible with the CIR shared weights; "
            "missing weights: "
            f"{names}"
        )
    if incompatible_shapes:
        details = "; ".join(incompatible_shapes)
        raise ValueError(
            "CP checkpoint and CIR shared weights use incompatible "
            "architectures: "
            f"{details}"
        )
    return transferred


def _source_name(target_name: str) -> str:
    if target_name.startswith(COMMON_PREFIX):
        return target_name
    if target_name == CIR_TASK_EMBEDDING:
        return CP_TASK_EMBEDDING
    raise ValueError(f"unsupported shared CIR weight: {target_name}")


def _find_source_tensor(
    cp_state: dict[str, Tensor],
    source_name: str,
) -> Tensor | None:
    value = cp_state.get(source_name)
    if value is not None:
        return value
    return cp_state.get(f"module.{source_name}")
