from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


@dataclass(frozen=True)
class OutfitExample:
    outfit_id: str
    images: Tensor
    descriptions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.images.ndim != 4:
            raise ValueError("images must have shape [items, channels, height, width]")
        if self.images.size(0) == 0:
            raise ValueError("an outfit must contain at least one item")
        if self.images.size(0) != len(self.descriptions):
            raise ValueError("each image must have one text description")


@dataclass(frozen=True)
class OutfitBatch:
    outfit_ids: tuple[str, ...]
    images: Tensor
    descriptions: tuple[tuple[str, ...], ...]
    padding_mask: Tensor

    def to(self, device: torch.device | str) -> "OutfitBatch":
        return OutfitBatch(
            outfit_ids=self.outfit_ids,
            images=self.images.to(device),
            descriptions=self.descriptions,
            padding_mask=self.padding_mask.to(device),
        )


@dataclass(frozen=True)
class CompatibilityExample:
    outfit_id: str
    images: Tensor
    descriptions: tuple[str, ...]
    label: float

    def __post_init__(self) -> None:
        OutfitExample(self.outfit_id, self.images, self.descriptions)
        if self.label not in (0.0, 1.0):
            raise ValueError("compatibility label must be 0 or 1")


@dataclass(frozen=True)
class CompatibilityBatch:
    outfits: OutfitBatch
    labels: Tensor

    @property
    def outfit_ids(self) -> tuple[str, ...]:
        return self.outfits.outfit_ids

    @property
    def images(self) -> Tensor:
        return self.outfits.images

    @property
    def descriptions(self) -> tuple[tuple[str, ...], ...]:
        return self.outfits.descriptions

    @property
    def padding_mask(self) -> Tensor:
        return self.outfits.padding_mask

    def to(self, device: torch.device | str) -> "CompatibilityBatch":
        return CompatibilityBatch(
            outfits=self.outfits.to(device),
            labels=self.labels.to(device),
        )


def collate_outfits(examples: Sequence[OutfitExample]) -> OutfitBatch:
    if not examples:
        raise ValueError("cannot collate an empty batch")

    item_counts = torch.tensor([example.images.size(0) for example in examples])
    max_items = int(item_counts.max().item())
    images = pad_sequence(
        [example.images for example in examples],
        batch_first=True,
        padding_value=0.0,
    )
    item_positions = torch.arange(max_items).unsqueeze(0)
    padding_mask = item_positions >= item_counts.unsqueeze(1)

    return OutfitBatch(
        outfit_ids=tuple(example.outfit_id for example in examples),
        images=images,
        descriptions=tuple(example.descriptions for example in examples),
        padding_mask=padding_mask,
    )


def collate_compatibility(
    examples: Sequence[CompatibilityExample],
) -> CompatibilityBatch:
    if not examples:
        raise ValueError("cannot collate an empty compatibility batch")
    outfits = collate_outfits(
        [
            OutfitExample(
                outfit_id=example.outfit_id,
                images=example.images,
                descriptions=example.descriptions,
            )
            for example in examples
        ]
    )
    labels = torch.tensor(
        [example.label for example in examples],
        dtype=torch.float32,
    )
    return CompatibilityBatch(outfits=outfits, labels=labels)
