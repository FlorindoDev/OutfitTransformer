"""Restore CP architecture and weights from project checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from data import get_dataset_source
from model import TransformerConfig
from training.CP import CPTrainingModel, FeatureMode


@dataclass(frozen=True)
class CPCheckpoint:
    """Validated model and dataset metadata stored by CP training."""

    path: Path
    epoch: int
    dataset_name: str
    dataset_id: str
    subset: str
    feature_mode: FeatureMode
    embedding_root: Path | None
    dataset_root: Path | None
    cache_dir: Path | None
    model_config: TransformerConfig
    state_dict: Mapping[str, Tensor]


def load_cp_checkpoint(path: str | Path) -> CPCheckpoint:
    """Load and validate one schema-v2 CP checkpoint on CPU."""
    selected_path = Path(path)
    if not selected_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {selected_path}")
    payload = torch.load(
        selected_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must contain a mapping")
    if payload.get("checkpoint_schema_version") != 2:
        raise ValueError("unsupported CP checkpoint schema_version")

    run_config = _mapping_value(payload, "run_config")
    dataset = _mapping_value(run_config, "dataset")
    model_values = _mapping_value(run_config, "model")
    state_dict = _mapping_value(payload, "model_state_dict")
    if not all(isinstance(value, Tensor) for value in state_dict.values()):
        raise TypeError("model_state_dict values must be tensors")

    dataset_id = _string_value(dataset, "id")
    dataset_name = _string_value(dataset, "name")
    source = get_dataset_source(dataset_name)
    if source.descriptor.dataset_id != dataset_id:
        raise ValueError("checkpoint dataset name and id do not match")
    subset = source.descriptor.validate_subset(
        _string_value(dataset, "subset")
    )
    try:
        feature_mode = FeatureMode(_string_value(dataset, "feature_mode"))
    except ValueError as error:
        raise ValueError("checkpoint contains unsupported feature mode") from error
    try:
        model_config = TransformerConfig(**dict(model_values))
    except TypeError as error:
        raise ValueError("checkpoint contains invalid model configuration") from error
    model_config.validate()

    embedding_root = _optional_path(dataset.get("embedding_root"))
    if feature_mode is FeatureMode.CLIP and embedding_root is None:
        raise ValueError("CLIP checkpoint does not declare embedding_root")
    epoch = payload.get("epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= 0:
        raise ValueError("checkpoint epoch must be a positive integer")

    return CPCheckpoint(
        path=selected_path,
        epoch=epoch,
        dataset_name=dataset_name,
        dataset_id=dataset_id,
        subset=subset,
        feature_mode=feature_mode,
        embedding_root=embedding_root,
        dataset_root=_optional_path(dataset.get("dataset_root")),
        cache_dir=_optional_path(dataset.get("cache_dir")),
        model_config=model_config,
        state_dict=state_dict,
    )


def restore_cp_model(checkpoint: CPCheckpoint) -> CPTrainingModel:
    """Build checkpoint architecture and restore weights strictly."""
    model = CPTrainingModel(
        checkpoint.model_config,
        feature_mode=checkpoint.feature_mode,
    )
    model.load_state_dict(checkpoint.state_dict, strict=True)
    return model


def _mapping_value(
    mapping: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"checkpoint {key} must be a mapping")
    return value


def _string_value(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"checkpoint {key} must be a non-empty string")
    return value


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TypeError("checkpoint path values must be non-empty strings or null")
    return Path(value)
