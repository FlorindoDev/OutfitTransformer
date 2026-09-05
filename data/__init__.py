"""Data ingestion, preprocessing and batching for OutfitTransformer."""

from .collate import collate_compatibility, collate_items, collate_retrieval
from .loaders import (
    LoaderConfig,
    create_compatibility_loader,
    create_item_loader,
    create_retrieval_loader,
)
from .source import (
    DEFAULT_DATASET_NAME,
    DataSplit,
    DatasetDescriptor,
    DatasetDownloadRequest,
    DatasetRequest,
    DatasetSource,
    IndexedDataset,
    RetrievalIndexDataset,
    available_dataset_names,
    get_dataset_source,
)
from .transforms import (
    ImageTransform,
    build_fashion_clip_transform,
    build_openrouter_transform,
    build_resnet18_transform,
)
from .types import (
    CompatibilityBatch,
    CompatibilityExample,
    CompatibilityIndexExample,
    FashionItem,
    ItemBatch,
    RetrievalBatch,
    RetrievalExample,
    RetrievalIndexExample,
)

__all__ = [
    "CompatibilityBatch",
    "CompatibilityExample",
    "CompatibilityIndexExample",
    "DEFAULT_DATASET_NAME",
    "DataSplit",
    "DatasetDescriptor",
    "DatasetDownloadRequest",
    "DatasetRequest",
    "DatasetSource",
    "FashionItem",
    "ImageTransform",
    "IndexedDataset",
    "ItemBatch",
    "LoaderConfig",
    "RetrievalBatch",
    "RetrievalExample",
    "RetrievalIndexExample",
    "RetrievalIndexDataset",
    "build_fashion_clip_transform",
    "build_openrouter_transform",
    "build_resnet18_transform",
    "collate_compatibility",
    "collate_items",
    "collate_retrieval",
    "create_compatibility_loader",
    "create_item_loader",
    "create_retrieval_loader",
    "available_dataset_names",
    "get_dataset_source",
]
