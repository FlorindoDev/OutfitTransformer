"""Structural types shared by Polyvore resource and catalog adapters."""

from typing import Any, Protocol


class ItemRows(Protocol):
    """Minimal indexed row access required from an item collection."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int, /) -> Any: ...

