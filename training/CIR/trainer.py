"""Training loop for Complementary Item Retrieval."""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, distributed as dist, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from metrics import retrieval_rank
from model import InBatchTripletMarginLoss
from training.common import (
    copy_checkpoint,
    save_checkpoint,
    write_json,
)

from .config import (
    BEST_METRIC,
    CIRTrainingConfig,
    MAX_GRAD_NORM,
    ONE_CYCLE_DIV_FACTOR,
    ONE_CYCLE_FINAL_DIV_FACTOR,
    ONE_CYCLE_PCT_START,
)
from .data import RetrievalLoaders, as_single_item_outfits
from .distributed import DistributedContext
from .model import CIRTrainingModel

LOGGER = logging.getLogger("training.CIR")
CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrainingResult:
    """Summary of a completed or early-stopped CIR run."""

    best_checkpoint: Path
    best_metric: str
    best_value: float
    epochs_completed: int


@dataclass(frozen=True)
class TrainEpochMetrics:
    loss: float
    examples: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationEpochMetrics:
    loss: float
    fitb_accuracy: float
    mrr: float
    recall_at_2: float
    examples: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _EpochTotals:
    loss_sum: float = 0.0
    loss_examples: int = 0
    examples: int = 0
    first_place: int = 0
    reciprocal_rank_sum: float = 0.0
    recall_at_2: int = 0

    def update_loss(
        self,
        loss: Tensor,
        examples: int,
        reduction: str,
    ) -> None:
        if examples <= 0:
            raise ValueError("loss examples must be positive")
        detached_loss = loss.detach()
        if detached_loss.numel() != 1 or not bool(torch.isfinite(detached_loss)):
            raise ValueError("loss must be one finite scalar")
        value = float(detached_loss.item())
        self.loss_sum += value * examples if reduction == "mean" else value
        self.loss_examples += examples

    def update_rank(self, rank: int) -> None:
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.examples += 1
        self.first_place += int(rank == 1)
        self.reciprocal_rank_sum += 1.0 / rank
        self.recall_at_2 += int(rank <= 2)


def train(
    model: CIRTrainingModel,
    loaders: RetrievalLoaders,
    config: CIRTrainingConfig,
    runtime: DistributedContext,
) -> TrainingResult:
    """Train CIR, checkpoint every epoch and select by FITB accuracy."""
    from .plots import save_cumulative_plots

    config.validate()
    if config.mixed_precision and runtime.device.type != "cuda":
        raise ValueError("mixed precision requires a CUDA device")
    _prepare_distributed_output(config.checkpoint_dir, runtime)

    run_config = config.as_dict(
        resolved_device=str(runtime.device),
        world_size=runtime.world_size,
        distributed_backend=runtime.backend,
    )
    model.to(runtime.device)
    training_model = _wrap_distributed(model, runtime)
    criterion = InBatchTripletMarginLoss(
        margin=config.triplet_margin,
        reduction=config.loss_reduction,
    )
    optimizer = AdamW(
        training_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(loaders.train) / config.gradient_accumulation_steps
    )
    if updates_per_epoch == 0:
        raise ValueError(
            "training loader is empty; reduce --batch-size or add examples"
        )
    if len(loaders.validation) == 0 and not runtime.enabled:
        raise ValueError("validation loader cannot be empty")
    total_optimizer_steps = updates_per_epoch * config.epochs
    run_config["training"]["optimizer_steps_per_epoch"] = updates_per_epoch
    run_config["training"]["total_optimizer_steps"] = total_optimizer_steps
    if runtime.is_main_process:
        write_json(run_config, config.checkpoint_dir / "config.json")

    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_optimizer_steps,
        pct_start=ONE_CYCLE_PCT_START,
        anneal_strategy="cos",
        div_factor=ONE_CYCLE_DIV_FACTOR,
        final_div_factor=ONE_CYCLE_FINAL_DIV_FACTOR,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.mixed_precision,
    )
    history: dict[str, Any] = {
        "epochs": [],
        "train": [],
        "validation": [],
    }
    best_value: float | None = None
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        _set_distributed_epoch(loaders.train, epoch)
        train_metrics = _train_epoch(
            training_model,
            loaders.train,
            criterion,
            optimizer,
            scheduler,
            scaler,
            config,
            runtime,
            epoch,
        )
        validation_metrics = _evaluate(
            training_model,
            loaders.validation,
            config,
            runtime,
        )
        history["epochs"].append(epoch)
        history["train"].append(train_metrics.as_dict())
        history["validation"].append(validation_metrics.as_dict())

        monitored_value = validation_metrics.fitb_accuracy
        improved = _is_improvement(
            monitored_value,
            best_value,
            min_delta=config.early_stopping_min_delta,
        )
        if improved:
            best_value = monitored_value
            stale_epochs = 0
        else:
            stale_epochs += 1

        if runtime.is_main_process:
            epoch_path = config.checkpoint_dir / "epochs" / (
                f"cir_epoch_{epoch:03d}.pt"
            )
            payload = _checkpoint_payload(
                epoch=epoch,
                model=_unwrap_model(training_model),
                run_config=run_config,
                history=history,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                best_value=best_value,
                is_best=improved,
            )
            save_checkpoint(payload, epoch_path)
            if improved:
                copy_checkpoint(epoch_path, config.checkpoint_dir / "best.pt")
            save_cumulative_plots(
                history,
                config.checkpoint_dir / "plots",
                epoch,
            )
            LOGGER.info(
                "epoch=%d/%d train_loss=%.6f val_loss=%.6f "
                "val_fitb_accuracy=%.4f val_mrr=%.4f "
                "val_recall_at_2=%.4f best_%s=%.6f",
                epoch,
                config.epochs,
                train_metrics.loss,
                validation_metrics.loss,
                validation_metrics.fitb_accuracy,
                validation_metrics.mrr,
                validation_metrics.recall_at_2,
                BEST_METRIC,
                best_value,
            )
        if (
            config.early_stopping_patience is not None
            and stale_epochs >= config.early_stopping_patience
        ):
            if runtime.is_main_process:
                LOGGER.info(
                    "early_stopping epoch=%d stale_epochs=%d",
                    epoch,
                    stale_epochs,
                )
            break

    if best_value is None:
        raise RuntimeError("training completed without a best checkpoint")
    return TrainingResult(
        best_checkpoint=config.checkpoint_dir / "best.pt",
        best_metric=BEST_METRIC,
        best_value=best_value,
        epochs_completed=len(history["epochs"]),
    )


def _train_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: AdamW,
    scheduler: OneCycleLR,
    scaler: Any,
    config: CIRTrainingConfig,
    runtime: DistributedContext,
    epoch: int,
) -> TrainEpochMetrics:
    model.train()
    totals = _EpochTotals()
    optimizer.zero_grad(set_to_none=True)
    batch_count = len(loader)

    for batch_index, batch in enumerate(loader, start=1):
        optimizer_step = _is_optimizer_step(
            batch_index,
            batch_count,
            config.gradient_accumulation_steps,
        )
        synchronization = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and not optimizer_step
            else nullcontext()
        )
        with synchronization:
            with _autocast(config, runtime.device):
                query_embeddings, positive_embeddings = model(
                    batch.partial_outfits,
                    as_single_item_outfits(batch.positive_items),
                    batch.target_categories,
                )
            loss = criterion(
                query_embeddings.float(),
                positive_embeddings.float(),
            )
            group_size = _accumulation_group_size(
                batch_index,
                batch_count,
                config.gradient_accumulation_steps,
            )
            scaler.scale(loss / group_size).backward()

        examples = len(batch.example_ids)
        totals.update_loss(loss, examples, config.loss_reduction)
        totals.examples += examples

        if optimizer_step:
            _optimizer_step(
                model,
                optimizer,
                scheduler,
                scaler,
            )

        if (
            runtime.is_main_process
            and (batch_index % config.log_every == 0 or batch_index == batch_count)
        ):
            LOGGER.info(
                "epoch=%d microbatch=%d/%d loss=%.6f lr=%.8g",
                epoch,
                batch_index,
                batch_count,
                float(loss.detach().item()),
                scheduler.get_last_lr()[0],
            )

    reduced = _reduce_totals(totals, runtime)
    return TrainEpochMetrics(
        loss=_mean_loss(reduced),
        examples=reduced.examples,
    )


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    config: CIRTrainingConfig,
    runtime: DistributedContext,
) -> ValidationEpochMetrics:
    model.eval()
    totals = _EpochTotals()
    for batch in loader:
        candidate_items, candidate_counts = _flatten_candidates(batch)
        with _autocast(config, runtime.device):
            query_embeddings, candidate_embeddings = model(
                batch.partial_outfits,
                as_single_item_outfits(candidate_items),
                batch.target_categories,
            )
        candidate_groups = candidate_embeddings.float().split(
            candidate_counts
        )
        losses: list[Tensor] = []
        for query, candidates in zip(
            query_embeddings.float(),
            candidate_groups,
            strict=True,
        ):
            distances = torch.linalg.vector_norm(
                candidates - query.unsqueeze(0),
                dim=1,
            )
            rank = retrieval_rank(distances.detach(), positive_index=0)
            totals.update_rank(rank)
            losses.append(
                F.relu(
                    distances[0]
                    - distances[1:].min()
                    + config.triplet_margin
                )
            )

        per_example_losses = torch.stack(losses)
        loss = (
            per_example_losses.mean()
            if config.loss_reduction == "mean"
            else per_example_losses.sum()
        )
        totals.update_loss(
            loss,
            len(losses),
            config.loss_reduction,
        )

    reduced = _reduce_totals(totals, runtime)
    if reduced.examples == 0:
        raise ValueError("validation metrics require at least one example")
    return ValidationEpochMetrics(
        loss=_mean_loss(reduced),
        fitb_accuracy=reduced.first_place / reduced.examples,
        mrr=reduced.reciprocal_rank_sum / reduced.examples,
        recall_at_2=reduced.recall_at_2 / reduced.examples,
        examples=reduced.examples,
    )


def _optimizer_step(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: OneCycleLR,
    scaler: Any,
) -> None:
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    if not scaler.is_enabled() or scaler.get_scale() >= previous_scale:
        scheduler.step()


def _flatten_candidates(batch: Any) -> tuple[tuple[Any, ...], tuple[int, ...]]:
    candidate_items: list[Any] = []
    candidate_counts: list[int] = []
    for positive, negatives in zip(
        batch.positive_items,
        batch.negative_items,
        strict=True,
    ):
        candidate_items.append(positive)
        candidate_items.extend(negatives)
        candidate_counts.append(1 + len(negatives))
    return tuple(candidate_items), tuple(candidate_counts)


def _autocast(config: CIRTrainingConfig, device: torch.device) -> Any:
    if not config.mixed_precision:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def _reduce_totals(
    totals: _EpochTotals,
    runtime: DistributedContext,
) -> _EpochTotals:
    if not runtime.enabled:
        return totals
    values = torch.tensor(
        [
            totals.loss_sum,
            totals.loss_examples,
            totals.examples,
            totals.first_place,
            totals.reciprocal_rank_sum,
            totals.recall_at_2,
        ],
        device=runtime.device,
        dtype=torch.float64,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return _EpochTotals(
        loss_sum=float(values[0].item()),
        loss_examples=int(values[1].item()),
        examples=int(values[2].item()),
        first_place=int(values[3].item()),
        reciprocal_rank_sum=float(values[4].item()),
        recall_at_2=int(values[5].item()),
    )


def _mean_loss(totals: _EpochTotals) -> float:
    if totals.loss_examples == 0:
        raise ValueError("epoch loss requires at least one example")
    return totals.loss_sum / totals.loss_examples


def _accumulation_group_size(
    batch_index: int,
    batch_count: int,
    accumulation_steps: int,
) -> int:
    group_start = ((batch_index - 1) // accumulation_steps) * accumulation_steps
    return min(accumulation_steps, batch_count - group_start)


def _is_optimizer_step(
    batch_index: int,
    batch_count: int,
    accumulation_steps: int,
) -> bool:
    return batch_index % accumulation_steps == 0 or batch_index == batch_count


def _is_improvement(
    value: float,
    best_value: float | None,
    *,
    min_delta: float,
) -> bool:
    return best_value is None or value > best_value + min_delta


def _set_distributed_epoch(loader: DataLoader[Any], epoch: int) -> None:
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)


def _wrap_distributed(
    model: CIRTrainingModel,
    runtime: DistributedContext,
) -> nn.Module:
    if not runtime.enabled:
        return model
    if runtime.device.type == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
            broadcast_buffers=False,
        )
    return DistributedDataParallel(model, broadcast_buffers=False)


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _checkpoint_payload(
    *,
    epoch: int,
    model: nn.Module,
    run_config: dict[str, Any],
    history: dict[str, Any],
    train_metrics: TrainEpochMetrics,
    validation_metrics: ValidationEpochMetrics,
    best_value: float | None,
    is_best: bool,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "run_config": run_config,
        "selection": {
            "metric": BEST_METRIC,
            "direction": "max",
            "value": validation_metrics.fitb_accuracy,
            "best_value": best_value,
            "is_best": is_best,
        },
        "train_metrics": train_metrics.as_dict(),
        "validation_metrics": validation_metrics.as_dict(),
        "training_history": history,
    }


def _prepare_output(checkpoint_dir: Path) -> None:
    existing_epochs = tuple(
        (checkpoint_dir / "epochs").glob("cir_epoch_*.pt")
    )
    if existing_epochs or (checkpoint_dir / "best.pt").exists():
        raise FileExistsError(
            f"checkpoint directory already contains a run: {checkpoint_dir}; "
            "choose another --checkpoint-dir"
        )
    (checkpoint_dir / "epochs").mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "plots").mkdir(parents=True, exist_ok=True)


def _prepare_distributed_output(
    checkpoint_dir: Path,
    runtime: DistributedContext,
) -> None:
    error_message: str | None = None
    if runtime.is_main_process:
        try:
            _prepare_output(checkpoint_dir)
        except (FileExistsError, OSError) as error:
            error_message = str(error)

    if runtime.enabled:
        messages: list[str | None] = [error_message]
        dist.broadcast_object_list(messages, src=0)
        error_message = messages[0]
    if error_message is not None:
        raise FileExistsError(error_message)
