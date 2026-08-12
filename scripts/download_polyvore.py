"""Download a registered dataset to a local directory."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from data import (
    DEFAULT_DATASET_NAME,
    DatasetDownloadRequest,
    get_dataset_source,
)

LOGGER = logging.getLogger("download_polyvore")


@dataclass(frozen=True)
class DownloadConfig:
    """Settings for one complete dataset repository download."""

    dataset_name: str
    output_dir: Path
    cache_dir: Path | None
    revision: str | None
    token: bool | str | None
    max_workers: int
    force_download: bool

    def validate(self) -> None:
        get_dataset_source(self.dataset_name)
        self.as_request()

    def as_request(self) -> DatasetDownloadRequest:
        """Return validated request understood by dataset sources."""
        return DatasetDownloadRequest(
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            revision=self.revision,
            token=self.token,
            max_workers=self.max_workers,
            force_download=self.force_download,
        )


def download_dataset(config: DownloadConfig) -> Path:
    """Download all repository files and return the local dataset path."""
    config.validate()
    source = get_dataset_source(config.dataset_name)

    LOGGER.info(
        "dataset=%s destination=%s",
        source.descriptor.dataset_id,
        config.output_dir,
    )
    destination = source.download(config.as_request())
    LOGGER.info("download_complete destination=%s", destination)
    return destination


def parse_args(argv: Sequence[str] | None = None) -> DownloadConfig:
    default_source = get_dataset_source(DEFAULT_DATASET_NAME)
    parser = argparse.ArgumentParser(
        description=(
            "Download a complete registered fashion dataset to a "
            "local directory. Existing unchanged files are reused."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "destination directory "
            f"(default: {default_source.descriptor.default_root})"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="optional Hugging Face cache directory",
    )
    parser.add_argument(
        "--revision",
        help="optional branch, tag or commit to download",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="parallel download workers (default: 8)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="redownload files even when already cached",
    )
    authentication = parser.add_mutually_exclusive_group()
    authentication.add_argument(
        "--token",
        help="explicit Hugging Face access token",
    )
    authentication.add_argument(
        "--no-token",
        action="store_true",
        help="disable Hugging Face authentication",
    )
    arguments = parser.parse_args(argv)
    source = get_dataset_source(arguments.dataset)
    token: bool | str | None = (
        False if arguments.no_token else arguments.token or True
    )
    config = DownloadConfig(
        dataset_name=source.descriptor.name,
        output_dir=arguments.output_dir or source.descriptor.default_root,
        cache_dir=arguments.cache_dir,
        revision=arguments.revision,
        token=token,
        max_workers=arguments.max_workers,
        force_download=arguments.force_download,
    )
    config.validate()
    return config


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    download_dataset(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
