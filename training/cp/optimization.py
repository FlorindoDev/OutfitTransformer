from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


SchedulerName = Literal["none", "step", "cosine"]
SCHEDULER_NAMES: tuple[SchedulerName, ...] = ("none", "step", "cosine")

_TRANSFORMER_GROUP = "transformer"
_RESNET_GROUPS = frozenset(("image_backbone", "resnet"))


@dataclass(frozen=True)
class CPSchedulerParameters:
    """Hyperparameters shared by StepLR and cosine LR policies."""

    step_size: int = 10
    gamma: float = 0.5
    min_learning_rate: float = 0.0

    def with_optional_overrides(
        self,
        *,
        step_size: int | None,
        gamma: float | None,
        min_learning_rate: float | None,
    ) -> CPSchedulerParameters | None:
        if step_size is None and gamma is None and min_learning_rate is None:
            return None
        return CPSchedulerParameters(
            step_size=self.step_size if step_size is None else step_size,
            gamma=self.gamma if gamma is None else gamma,
            min_learning_rate=(
                self.min_learning_rate
                if min_learning_rate is None
                else min_learning_rate
            ),
        )

    def validate(self) -> None:
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if self.min_learning_rate < 0.0:
            raise ValueError("min_learning_rate must be non-negative")


class CPGroupLRScheduler:
    """Apply one checkpointable LR policy to each optimizer parameter group."""

    def __init__(
        self,
        optimizer: Optimizer,
        scheduler_names: Mapping[str, SchedulerName],
        *,
        parameters: CPSchedulerParameters,
        group_parameters: Mapping[str, CPSchedulerParameters] | None = None,
        total_epochs: int,
    ) -> None:
        self.optimizer = optimizer
        self.scheduler_names = dict(scheduler_names)
        self.parameters = parameters
        self.total_epochs = total_epochs
        self.last_epoch = 0
        self.group_names = tuple(
            _group_name(group, index)
            for index, group in enumerate(optimizer.param_groups)
        )
        overrides = {} if group_parameters is None else dict(group_parameters)
        self.group_parameters = {
            group_name: overrides.get(group_name, parameters)
            for group_name in self.group_names
        }
        self.base_lrs = tuple(
            float(group["lr"]) for group in optimizer.param_groups
        )
        self._last_lr = list(self.base_lrs)
        self._validate()

    def step(self) -> None:
        self.last_epoch += 1
        learning_rates = [
            self._learning_rate(group_name, base_lr)
            for group_name, base_lr in zip(
                self.group_names,
                self.base_lrs,
                strict=True,
            )
        ]
        for group, learning_rate in zip(
            self.optimizer.param_groups,
            learning_rates,
            strict=True,
        ):
            group["lr"] = learning_rate
        self._last_lr = learning_rates

    def get_last_lr(self) -> list[float]:
        return list(self._last_lr)

    def state_dict(self) -> dict[str, Any]:
        return {
            "scheduler_type": "cp_group",
            "scheduler_names": dict(self.scheduler_names),
            "step_size": self.parameters.step_size,
            "gamma": self.parameters.gamma,
            "min_learning_rate": self.parameters.min_learning_rate,
            "group_parameters": {
                group_name: {
                    "step_size": parameters.step_size,
                    "gamma": parameters.gamma,
                    "min_learning_rate": parameters.min_learning_rate,
                }
                for group_name, parameters in self.group_parameters.items()
            },
            "total_epochs": self.total_epochs,
            "last_epoch": self.last_epoch,
            "group_names": list(self.group_names),
            "base_lrs": list(self.base_lrs),
            "last_lrs": list(self._last_lr),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if state_dict.get("scheduler_type") != "cp_group":
            raise ValueError("checkpoint does not contain a grouped CP scheduler")

        group_names = tuple(str(name) for name in state_dict["group_names"])
        current_group_names = tuple(
            _group_name(group, index)
            for index, group in enumerate(self.optimizer.param_groups)
        )
        if group_names != current_group_names:
            raise ValueError(
                "checkpoint scheduler parameter groups do not match optimizer"
            )

        scheduler_names = state_dict["scheduler_names"]
        if not isinstance(scheduler_names, Mapping):
            raise ValueError("checkpoint scheduler_names must be a mapping")

        self.scheduler_names = {
            str(name): _scheduler_name(value)
            for name, value in scheduler_names.items()
        }
        self.parameters = CPSchedulerParameters(
            step_size=int(state_dict["step_size"]),
            gamma=float(state_dict["gamma"]),
            min_learning_rate=float(state_dict["min_learning_rate"]),
        )
        self.total_epochs = int(state_dict["total_epochs"])
        self.last_epoch = int(state_dict["last_epoch"])
        self.group_names = group_names
        self.group_parameters = _load_group_parameters(
            state_dict.get("group_parameters"),
            group_names=group_names,
            fallback=self.parameters,
        )
        self.base_lrs = tuple(float(value) for value in state_dict["base_lrs"])
        self._last_lr = [float(value) for value in state_dict["last_lrs"]]
        self._validate()

        if len(self._last_lr) != len(self.optimizer.param_groups):
            raise ValueError("checkpoint scheduler LR count does not match optimizer")
        for group, learning_rate in zip(
            self.optimizer.param_groups,
            self._last_lr,
            strict=True,
        ):
            group["lr"] = learning_rate

    def _learning_rate(self, group_name: str, base_lr: float) -> float:
        scheduler_name = self.scheduler_names[group_name]
        parameters = self.group_parameters[group_name]
        if scheduler_name == "none":
            return base_lr
        if scheduler_name == "step":
            reductions = self.last_epoch // parameters.step_size
            return base_lr * parameters.gamma**reductions

        progress = min(self.last_epoch, self.total_epochs) / self.total_epochs
        cosine = (1.0 + math.cos(math.pi * progress)) / 2.0
        return parameters.min_learning_rate + (
            base_lr - parameters.min_learning_rate
        ) * cosine

    def _validate(self) -> None:
        self.parameters.validate()
        for parameters in self.group_parameters.values():
            parameters.validate()
        if self.total_epochs <= 0:
            raise ValueError("total_epochs must be positive")
        if len(self.group_names) != len(self.base_lrs):
            raise ValueError("scheduler group names and base LRs must align")
        if set(self.group_names) != set(self.scheduler_names):
            raise ValueError("each optimizer group requires one scheduler")
        if set(self.group_names) != set(self.group_parameters):
            raise ValueError("each optimizer group requires scheduler parameters")
        for scheduler_name in self.scheduler_names.values():
            _scheduler_name(scheduler_name)
        for group_name, base_lr in zip(
            self.group_names,
            self.base_lrs,
            strict=True,
        ):
            if base_lr <= 0.0:
                raise ValueError("optimizer learning rates must be positive")
            if (
                self.scheduler_names[group_name] == "cosine"
                and self.group_parameters[group_name].min_learning_rate > base_lr
            ):
                raise ValueError(
                    "min_learning_rate cannot exceed a cosine group's initial LR"
                )


def create_cp_scheduler(
    optimizer: Optimizer,
    *,
    scheduler: SchedulerName,
    total_epochs: int,
    parameters: CPSchedulerParameters,
    transformer_scheduler: SchedulerName | None = None,
    resnet_scheduler: SchedulerName | None = None,
    transformer_parameters: CPSchedulerParameters | None = None,
    resnet_parameters: CPSchedulerParameters | None = None,
) -> Any | None:
    """Create legacy common scheduler or independent group schedulers."""
    _validate_scheduler_options(
        scheduler=scheduler,
        total_epochs=total_epochs,
        parameters=parameters,
    )
    if (
        transformer_scheduler is None
        and resnet_scheduler is None
        and transformer_parameters is None
        and resnet_parameters is None
    ):
        return _create_common_scheduler(
            optimizer,
            scheduler=scheduler,
            total_epochs=total_epochs,
            parameters=parameters,
        )

    group_names = tuple(
        _group_name(group, index)
        for index, group in enumerate(optimizer.param_groups)
    )
    if (
        transformer_scheduler is not None or transformer_parameters is not None
    ) and _TRANSFORMER_GROUP not in group_names:
        raise ValueError(
            "transformer scheduler requires a transformer optimizer group"
        )
    if (
        resnet_scheduler is not None or resnet_parameters is not None
    ) and not _RESNET_GROUPS.intersection(group_names):
        raise ValueError("ResNet scheduler requires a ResNet optimizer group")

    scheduler_names: dict[str, SchedulerName] = {
        group_name: _group_scheduler_name(
            group_name,
            default=scheduler,
            transformer=transformer_scheduler,
            resnet=resnet_scheduler,
        )
        for group_name in group_names
    }
    group_parameters = {
        group_name: _group_scheduler_parameters(
            group_name,
            default=parameters,
            transformer=transformer_parameters,
            resnet=resnet_parameters,
        )
        for group_name in group_names
    }
    if all(name == "none" for name in scheduler_names.values()):
        return None
    return CPGroupLRScheduler(
        optimizer,
        scheduler_names,
        parameters=parameters,
        group_parameters=group_parameters,
        total_epochs=total_epochs,
    )


def _create_common_scheduler(
    optimizer: Optimizer,
    *,
    scheduler: SchedulerName,
    total_epochs: int,
    parameters: CPSchedulerParameters,
) -> Any | None:
    if scheduler == "none":
        return None
    if scheduler == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=parameters.min_learning_rate,
        )
    return StepLR(
        optimizer,
        step_size=parameters.step_size,
        gamma=parameters.gamma,
    )


def _group_scheduler_name(
    group_name: str,
    *,
    default: SchedulerName,
    transformer: SchedulerName | None,
    resnet: SchedulerName | None,
) -> SchedulerName:
    if group_name == _TRANSFORMER_GROUP and transformer is not None:
        return transformer
    if group_name in _RESNET_GROUPS and resnet is not None:
        return resnet
    return default


def _group_scheduler_parameters(
    group_name: str,
    *,
    default: CPSchedulerParameters,
    transformer: CPSchedulerParameters | None,
    resnet: CPSchedulerParameters | None,
) -> CPSchedulerParameters:
    if group_name == _TRANSFORMER_GROUP and transformer is not None:
        return transformer
    if group_name in _RESNET_GROUPS and resnet is not None:
        return resnet
    return default


def _validate_scheduler_options(
    *,
    scheduler: SchedulerName,
    total_epochs: int,
    parameters: CPSchedulerParameters,
) -> None:
    _scheduler_name(scheduler)
    parameters.validate()
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")


def _load_group_parameters(
    value: object,
    *,
    group_names: tuple[str, ...],
    fallback: CPSchedulerParameters,
) -> dict[str, CPSchedulerParameters]:
    if value is None:
        return {group_name: fallback for group_name in group_names}
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint group_parameters must be a mapping")

    parameters: dict[str, CPSchedulerParameters] = {}
    for group_name in group_names:
        saved = value.get(group_name)
        if not isinstance(saved, Mapping):
            raise ValueError(
                f"checkpoint lacks scheduler parameters for group {group_name}"
            )
        parameters[group_name] = CPSchedulerParameters(
            step_size=int(saved["step_size"]),
            gamma=float(saved["gamma"]),
            min_learning_rate=float(saved["min_learning_rate"]),
        )
    return parameters


def _group_name(group: Mapping[str, Any], index: int) -> str:
    return str(group.get("group_name", f"group_{index}"))


def _scheduler_name(value: object) -> SchedulerName:
    if value not in SCHEDULER_NAMES:
        raise ValueError(f"scheduler must be one of {SCHEDULER_NAMES}")
    return cast(SchedulerName, value)


__all__ = [
    "CPGroupLRScheduler",
    "CPSchedulerParameters",
    "SCHEDULER_NAMES",
    "SchedulerName",
    "create_cp_scheduler",
]
