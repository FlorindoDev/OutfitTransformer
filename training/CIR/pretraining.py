"""Initialize weights shared by CP and CIR from a CP checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from training.common import load_checkpoint_state_dict

COMMON_PREFIX = "common."
CP_ENCODER_PREFIX = "cp.encoder."
CIR_ENCODER_PREFIX = "cir.encoder."
CP_TASK_EMBEDDING = "cp.task_embedding.embedding"
CIR_TASK_EMBEDDING = "cir.task_embedding.embedding"
REMOVED_CP_WEIGHTS = frozenset({"cp.encoder.norm.weight", "cp.encoder.norm.bias"})


@dataclass(frozen=True)
class CPPretrainingReport:
    """Summary of tensors transferred from CP into CIR."""

    checkpoint: Path
    loaded_keys: tuple[str, ...]
    ignored_keys: tuple[str, ...] = ()

    @property
    def loaded_tensor_count(self) -> int:
        return len(self.loaded_keys)


def load_cp_pretrained_weights(
    model: nn.Module,
    path: str | Path,
    *,
    map_location: torch.device | str = "cpu",
) -> CPPretrainingReport:
    """Transfer common encoders, Transformer layers and task token from CP.

    Only the retrieval embedding, category embedding and retrieval head keep
    their fresh initialization. Legacy final encoder LayerNorm weights are
    ignored because that layer no longer exists in CIR.
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
        ignored_keys=tuple(
            sorted(
                name
                for name in cp_state
                if name.removeprefix("module.") in REMOVED_CP_WEIGHTS
            )
        ),
    )


def _build_transfer_state(
    cp_state: dict[str, Tensor],
    cir_state: dict[str, Tensor],
) -> dict[str, Tensor]:
    required_targets = {
        name
        for name in cir_state
        if name.startswith((COMMON_PREFIX, CIR_ENCODER_PREFIX))
        or name == CIR_TASK_EMBEDDING
    }
    if not required_targets:
        raise ValueError(
            "CIR model does not expose weights shared with CP"
        )
    _validate_shared_architecture(cp_state, required_targets)

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


def _validate_shared_architecture(
    cp_state: dict[str, Tensor],
    required_targets: set[str],
) -> None:
    expected_names = {_source_name(name) for name in required_targets}
    source_names = {
        name.removeprefix("module.")
        for name in cp_state
        if name.removeprefix("module.").startswith(
            (COMMON_PREFIX, CP_ENCODER_PREFIX)
        )
    }
    unexpected_names = sorted(source_names - expected_names - REMOVED_CP_WEIGHTS)
    if unexpected_names:
        names = ", ".join(unexpected_names)
        raise ValueError(
            "CP checkpoint shared architecture does not match the CIR model; "
            f"unexpected weights: {names}"
        )


def _source_name(target_name: str) -> str:
    if target_name.startswith(COMMON_PREFIX):
        return target_name
    if target_name.startswith(CIR_ENCODER_PREFIX):
        return CP_ENCODER_PREFIX + target_name.removeprefix(CIR_ENCODER_PREFIX)
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
