from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.optim import Adam, AdamW, Optimizer

from model import CompatibilityPredictor, ImageFineTuneMode, OutfitEncoderConfig


CPOptimizerName = Literal["adam", "adamw"]
CP_OPTIMIZER_NAMES: tuple[CPOptimizerName, ...] = ("adam", "adamw")


@dataclass(frozen=True)
class CPFineTuneCheckpoint:
    """CP model weights and provenance required to start a new training phase."""

    path: Path
    epoch: int
    model_state_dict: Mapping[str, Any]
    run_config: dict[str, Any] | None
    schema_version: int | None

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: torch.device | str = "cpu",
    ) -> "CPFineTuneCheckpoint":
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"fine-tune checkpoint not found: {checkpoint_path}"
            )

        payload = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        if not isinstance(payload, Mapping):
            raise ValueError("fine-tune checkpoint must be a dictionary")

        epoch = payload.get("epoch")
        if not isinstance(epoch, int) or isinstance(epoch, bool):
            raise ValueError("fine-tune checkpoint missing integer epoch")
        model_state = payload.get("model_state_dict")
        if not isinstance(model_state, Mapping):
            raise ValueError("fine-tune checkpoint missing model_state_dict")

        schema_version = payload.get("checkpoint_schema_version")
        if schema_version is not None and (
            not isinstance(schema_version, int) or isinstance(schema_version, bool)
        ):
            raise ValueError("checkpoint_schema_version must be an integer")

        run_config_value = payload.get("run_config")
        run_config = (
            dict(run_config_value)
            if isinstance(run_config_value, Mapping)
            else None
        )
        return cls(
            path=checkpoint_path,
            epoch=epoch,
            model_state_dict=model_state,
            run_config=run_config,
            schema_version=schema_version,
        )

    def model_config(
        self,
        *,
        image_fine_tune_mode: ImageFineTuneMode,
        dropout: float | None = None,
        text_model_name: str | None = None,
    ) -> OutfitEncoderConfig:
        """Reuse shape-defining model settings and override safe settings."""
        values = self._saved_model_config()
        values["pretrained_image_encoder"] = False
        values["image_fine_tune_mode"] = image_fine_tune_mode
        if dropout is not None:
            values["dropout"] = dropout
        if text_model_name is not None:
            values["text_model_name"] = text_model_name

        config = OutfitEncoderConfig(**values)
        config.validate()
        return config

    def load_weights(self, model: nn.Module) -> None:
        try:
            model.load_state_dict(self.model_state_dict, strict=True)
        except RuntimeError as error:
            raise ValueError(
                "checkpoint weights are incompatible with the selected model; "
                "fine-tuning may change dropout and training policy, but not "
                "shape-defining architecture settings"
            ) from error

    def _saved_model_config(self) -> dict[str, Any]:
        if self.run_config is None:
            return {}
        value = self.run_config.get("model")
        if not isinstance(value, Mapping):
            return {}

        supported_names = {field.name for field in fields(OutfitEncoderConfig)}
        return {
            name: nested_value
            for name, nested_value in value.items()
            if name in supported_names
        }


@dataclass(frozen=True)
class CPFineTuneOptimizerConfig:
    name: CPOptimizerName = "adam"
    learning_rate: float = 1e-5
    transformer_learning_rate: float | None = None
    image_backbone_learning_rate: float | None = None
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def validate(self) -> None:
        if self.name not in CP_OPTIMIZER_NAMES:
            raise ValueError(f"optimizer must be one of {CP_OPTIMIZER_NAMES}")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if (
            self.transformer_learning_rate is not None
            and self.transformer_learning_rate <= 0.0
        ):
            raise ValueError("transformer_learning_rate must be positive")
        if (
            self.image_backbone_learning_rate is not None
            and self.image_backbone_learning_rate <= 0.0
        ):
            raise ValueError("image_backbone_learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("Adam betas must be in [0, 1)")
        if self.eps <= 0.0:
            raise ValueError("Adam eps must be positive")

    @property
    def resolved_image_backbone_learning_rate(self) -> float:
        return (
            self.learning_rate
            if self.image_backbone_learning_rate is None
            else self.image_backbone_learning_rate
        )

    @property
    def resolved_transformer_learning_rate(self) -> float:
        return (
            self.learning_rate
            if self.transformer_learning_rate is None
            else self.transformer_learning_rate
        )


def build_cp_fine_tune_optimizer(
    model: CompatibilityPredictor,
    config: CPFineTuneOptimizerConfig,
    *,
    separate_transformer: bool | None = None,
    separate_image_backbone: bool = True,
) -> Optimizer:
    """Build optimizer groups for task, Transformer and ResNet parameters."""
    config.validate()
    backbone = model.encoder.image_encoder.backbone
    image_feature_parameters = tuple(
        parameter
        for name, parameter in backbone.named_parameters()
        if separate_image_backbone
        and not name.startswith("fc.")
        and parameter.requires_grad
    )
    image_feature_ids = {id(parameter) for parameter in image_feature_parameters}
    should_separate_transformer = (
        config.transformer_learning_rate is not None
        if separate_transformer is None
        else separate_transformer
    )
    transformer_parameters = tuple(
        parameter
        for parameter in model.encoder.context_encoder.parameters()
        if should_separate_transformer and parameter.requires_grad
    )
    transformer_ids = {id(parameter) for parameter in transformer_parameters}
    task_parameters = tuple(
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
        and id(parameter) not in image_feature_ids
        and id(parameter) not in transformer_ids
    )
    if not task_parameters:
        raise ValueError("fine-tuning requires trainable non-backbone parameters")

    parameter_groups: list[dict[str, Any]] = [
        {
            "params": task_parameters,
            "lr": config.learning_rate,
            "group_name": "task",
        }
    ]
    if transformer_parameters:
        parameter_groups.append(
            {
                "params": transformer_parameters,
                "lr": config.resolved_transformer_learning_rate,
                "group_name": "transformer",
            }
        )
    if image_feature_parameters:
        parameter_groups.append(
            {
                "params": image_feature_parameters,
                "lr": config.resolved_image_backbone_learning_rate,
                "group_name": "image_backbone",
            }
        )

    optimizer_type = Adam if config.name == "adam" else AdamW
    return optimizer_type(
        parameter_groups,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


def optimizer_learning_rates(optimizer: Optimizer) -> dict[str, float]:
    return {
        str(group.get("group_name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


__all__ = [
    "CPFineTuneCheckpoint",
    "CPFineTuneOptimizerConfig",
    "CPOptimizerName",
    "CP_OPTIMIZER_NAMES",
    "build_cp_fine_tune_optimizer",
    "optimizer_learning_rates",
]
