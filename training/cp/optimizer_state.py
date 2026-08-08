from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Adam, AdamW, Optimizer


@dataclass(frozen=True)
class CPOptimizerStateTransfer:
    """Result of copying compatible Adam moments into a new optimizer."""

    source_path: Path
    restored_parameters: int
    source_parameters_with_state: int
    target_parameters: int

    @property
    def new_target_parameters(self) -> int:
        return self.target_parameters - self.restored_parameters


def optimizer_parameter_names(
    model: nn.Module,
    optimizer: Optimizer,
) -> tuple[tuple[str, ...], ...]:
    """Return model parameter names aligned with optimizer parameter groups."""
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    groups: list[tuple[str, ...]] = []
    for group in optimizer.param_groups:
        names: list[str] = []
        for parameter in group["params"]:
            name = names_by_id.get(id(parameter))
            if name is None:
                raise ValueError("optimizer contains a parameter outside the model")
            names.append(name)
        groups.append(tuple(names))
    return tuple(groups)


def transfer_adam_state_from_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
) -> CPOptimizerStateTransfer:
    """Copy Adam/AdamW moments by parameter name without replacing new settings."""
    if not isinstance(optimizer, (Adam, AdamW)):
        raise TypeError("optimizer state transfer supports only Adam and AdamW")

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"optimizer state checkpoint not found: {checkpoint_path}"
        )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("optimizer state checkpoint must be a dictionary")
    optimizer_state = payload.get("optimizer_state_dict")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("checkpoint lacks optimizer_state_dict")

    source_states = _source_states_by_name(payload, optimizer_state)
    target_names = optimizer_parameter_names(model, optimizer)
    target_parameters = _target_parameters_by_name(
        model,
        optimizer,
        target_names,
    )

    restored = 0
    for name, parameter in target_parameters.items():
        source_state = source_states.get(name)
        if source_state is None or not _is_compatible_adam_state(
            source_state,
            parameter,
        ):
            continue
        optimizer.state[parameter] = _copy_adam_state(source_state, parameter)
        restored += 1

    if restored == 0:
        raise ValueError(
            "optimizer checkpoint has no compatible Adam state for the new phase"
        )
    return CPOptimizerStateTransfer(
        source_path=checkpoint_path.resolve(),
        restored_parameters=restored,
        source_parameters_with_state=len(source_states),
        target_parameters=len(target_parameters),
    )


def _source_states_by_name(
    payload: Mapping[str, Any],
    optimizer_state: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    state = optimizer_state.get("state")
    groups = optimizer_state.get("param_groups")
    if not isinstance(state, Mapping) or not isinstance(groups, Sequence):
        raise ValueError("checkpoint optimizer state has invalid structure")

    names = _checkpoint_parameter_names(payload, optimizer_state)
    states_by_name: dict[str, Mapping[str, Any]] = {}
    for group, group_names in zip(groups, names, strict=True):
        if not isinstance(group, Mapping):
            raise ValueError("checkpoint optimizer parameter group is invalid")
        parameter_ids = group.get("params")
        if not isinstance(parameter_ids, Sequence):
            raise ValueError("checkpoint optimizer group lacks parameters")
        for parameter_id, name in zip(parameter_ids, group_names, strict=True):
            parameter_state = state.get(parameter_id)
            if isinstance(parameter_state, Mapping) and parameter_state:
                states_by_name[name] = parameter_state
    return states_by_name


def _checkpoint_parameter_names(
    payload: Mapping[str, Any],
    optimizer_state: Mapping[str, Any],
) -> tuple[tuple[str, ...], ...]:
    saved_names = payload.get("optimizer_parameter_names")
    if saved_names is None:
        raise ValueError(
            "optimizer state transfer requires a checkpoint with "
            "optimizer_parameter_names"
        )
    return _validate_saved_parameter_names(saved_names, optimizer_state)


def _validate_saved_parameter_names(
    value: object,
    optimizer_state: Mapping[str, Any],
) -> tuple[tuple[str, ...], ...]:
    groups = optimizer_state.get("param_groups")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("optimizer_parameter_names must be a sequence")
    if not isinstance(groups, Sequence) or len(value) != len(groups):
        raise ValueError("optimizer parameter-name groups do not align")

    normalized: list[tuple[str, ...]] = []
    for saved_group, optimizer_group in zip(value, groups, strict=True):
        if not isinstance(saved_group, Sequence) or isinstance(
            saved_group,
            (str, bytes),
        ):
            raise ValueError("optimizer parameter names group is invalid")
        if not isinstance(optimizer_group, Mapping):
            raise ValueError("checkpoint optimizer parameter group is invalid")
        names = tuple(str(name) for name in saved_group)
        parameter_ids = optimizer_group.get("params")
        if not isinstance(parameter_ids, Sequence) or len(names) != len(parameter_ids):
            raise ValueError("optimizer parameter names do not align with parameters")
        normalized.append(names)
    return tuple(normalized)


def _target_parameters_by_name(
    model: nn.Module,
    optimizer: Optimizer,
    group_names: tuple[tuple[str, ...], ...],
) -> dict[str, nn.Parameter]:
    named_parameters = dict(model.named_parameters())
    targets: dict[str, nn.Parameter] = {}
    for group, names in zip(optimizer.param_groups, group_names, strict=True):
        for parameter, name in zip(group["params"], names, strict=True):
            if named_parameters.get(name) is not parameter:
                raise ValueError("optimizer parameter-name mapping is inconsistent")
            targets[name] = parameter
    return targets


def _is_compatible_adam_state(
    state: Mapping[str, Any],
    parameter: nn.Parameter,
) -> bool:
    for name in ("exp_avg", "exp_avg_sq"):
        value = state.get(name)
        if not isinstance(value, Tensor) or value.shape != parameter.shape:
            return False
    return True


def _copy_adam_state(
    state: Mapping[str, Any],
    parameter: nn.Parameter,
) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for name, value in state.items():
        if isinstance(value, Tensor):
            tensor = value.detach().clone()
            if name != "step":
                tensor = tensor.to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            copied[name] = tensor
        else:
            copied[name] = deepcopy(value)
    return copied


__all__ = [
    "CPOptimizerStateTransfer",
    "optimizer_parameter_names",
    "transfer_adam_state_from_checkpoint",
]
