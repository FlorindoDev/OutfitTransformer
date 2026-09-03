"""Polyvore implementation of the public dataset-source contract."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from ..source import (
    DatasetDescriptor,
    DatasetDownloadRequest,
    DatasetRequest,
    IndexedDataset,
)
from ..transforms import ImageTransform
from ..types import (
    CompatibilityExample,
    CompatibilityIndexExample,
    FashionItem,
    RetrievalExample,
    RetrievalIndexExample,
)

from .catalog import PolyvoreCatalog, load_item_categories
from .compatibility_dataset import (
    PolyvoreCompatibilityDataset,
    PolyvoreCompatibilityIndexDataset,
)
from .download import (
    DEFAULT_DATASET_ROOT,
    POLYVORE_DATASET_ID,
    PolyvoreResources,
    PolyvoreSplit,
    PolyvoreTask,
    PolyvoreVariant,
    download_polyvore_resources,
)
from .item_dataset import PolyvoreItemDataset
from .retrieval_dataset import (
    PolyvoreRetrievalDataset,
    PolyvoreRetrievalIndexDataset,
)


class PolyvoreSource:
    """Adapt Polyvore files to project-neutral datasets."""

    descriptor = DatasetDescriptor(
        name="polyvore",
        dataset_id=POLYVORE_DATASET_ID,
        default_root=DEFAULT_DATASET_ROOT,
        subsets=tuple(variant.value for variant in PolyvoreVariant),
        default_subset=PolyvoreVariant.NONDISJOINT.value,
    )

    def item_dataset(
        self,
        request: DatasetRequest,
        image_transform: ImageTransform,
    ) -> Dataset[FashionItem]:
        resources = self._resources(request, PolyvoreTask.ITEMS)
        return PolyvoreItemDataset(_build_catalog(resources, image_transform))

    def compatibility_dataset(
        self,
        request: DatasetRequest,
        image_transform: ImageTransform,
    ) -> Dataset[CompatibilityExample]:
        resources = self._resources(request, PolyvoreTask.COMPATIBILITY)
        return PolyvoreCompatibilityDataset(
            _build_catalog(resources, image_transform),
            _require_path(resources.compatibility_path, "compatibility_path"),
            _require_path(resources.outfits_path, "outfits_path"),
        )

    def compatibility_index_dataset(
        self,
        request: DatasetRequest,
    ) -> IndexedDataset[CompatibilityIndexExample]:
        resources = self._resources(
            request,
            PolyvoreTask.COMPATIBILITY,
            include_items=False,
        )
        return PolyvoreCompatibilityIndexDataset(
            _require_path(resources.compatibility_path, "compatibility_path"),
            _require_path(resources.outfits_path, "outfits_path"),
        )

    def retrieval_dataset(
        self,
        request: DatasetRequest,
        image_transform: ImageTransform,
    ) -> Dataset[RetrievalExample]:
        resources = self._resources(request, PolyvoreTask.RETRIEVAL)
        return PolyvoreRetrievalDataset(
            _build_catalog(resources, image_transform),
            _require_path(resources.retrieval_path, "retrieval_path"),
            _require_path(resources.outfits_path, "outfits_path"),
        )

    def retrieval_index_dataset(
        self,
        request: DatasetRequest,
        *,
        include_categories: bool = True,
    ) -> IndexedDataset[RetrievalIndexExample]:
        resources = self._resources(
            request,
            PolyvoreTask.RETRIEVAL,
            include_items=False,
            include_metadata=include_categories,
        )
        categories = (
            load_item_categories(
                _require_path(resources.metadata_path, "metadata_path")
            )
            if include_categories
            else None
        )
        return PolyvoreRetrievalIndexDataset(
            _require_path(resources.retrieval_path, "retrieval_path"),
            _require_path(resources.outfits_path, "outfits_path"),
            categories,
        )

    def download(self, request: DatasetDownloadRequest) -> Path:
        """Download complete Hugging Face repository to requested directory."""
        request.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise ImportError(
                "Polyvore download requires the 'huggingface-hub' package"
            ) from error

        downloaded_path = snapshot_download(
            repo_id=self.descriptor.dataset_id,
            repo_type="dataset",
            revision=request.revision,
            cache_dir=request.cache_dir,
            local_dir=request.output_dir,
            token=request.token,
            max_workers=request.max_workers,
            force_download=request.force_download,
        )
        destination = Path(downloaded_path)
        if not destination.is_dir():
            raise FileNotFoundError(
                f"downloaded dataset directory does not exist: {destination}"
            )
        return destination

    def _resources(
        self,
        request: DatasetRequest,
        task: PolyvoreTask,
        *,
        include_items: bool = True,
        include_metadata: bool | None = None,
    ) -> PolyvoreResources:
        variant = PolyvoreVariant(
            self.descriptor.validate_subset(request.subset)
        )
        split = PolyvoreSplit(request.split.value)
        return download_polyvore_resources(
            task=task,
            variant=variant,
            split=split,
            token=request.token,
            cache_dir=request.cache_dir,
            dataset_root=request.root,
            include_items=include_items,
            include_metadata=include_metadata,
        )


def _build_catalog(
    resources: PolyvoreResources,
    image_transform: ImageTransform,
) -> PolyvoreCatalog:
    if resources.item_rows is None or resources.metadata_path is None:
        raise RuntimeError("Polyvore catalog resources are incomplete")
    return PolyvoreCatalog(
        resources.item_rows,
        resources.metadata_path,
        image_transform,
    )


def _require_path(path: Path | None, name: str) -> Path:
    if path is None:
        raise RuntimeError(f"Polyvore resources do not contain {name}")
    return path
