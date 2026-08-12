"""Download the complete Polyvore Outfits dataset to a local directory."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from data.polyvore import DEFAULT_DATASET_ROOT, POLYVORE_DATASET_ID

LOGGER = logging.getLogger("download_polyvore")


@dataclass(frozen=True)
class DownloadConfig:
    """Settings for one complete Polyvore repository download."""

    output_dir: Path
    cache_dir: Path | None
    revision: str | None
    token: bool | str | None
    max_workers: int
    force_download: bool

    def validate(self) -> None:
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise NotADirectoryError(
                f"output path is not a directory: {self.output_dir}"
            )


def download_dataset(config: DownloadConfig) -> Path:
    """Download all repository files and return the local dataset path."""
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError(
            "Polyvore download requires the 'huggingface-hub' package"
        ) from error

    LOGGER.info(
        "dataset=%s destination=%s",
        POLYVORE_DATASET_ID,
        config.output_dir,
    )
    downloaded_path = snapshot_download(
        repo_id=POLYVORE_DATASET_ID,
        repo_type="dataset",
        revision=config.revision,
        cache_dir=config.cache_dir,
        local_dir=config.output_dir,
        token=config.token,
        max_workers=config.max_workers,
        force_download=config.force_download,
    )
    destination = Path(downloaded_path)
    if not destination.is_dir():
        raise FileNotFoundError(
            f"downloaded dataset directory does not exist: {destination}"
        )
    LOGGER.info("download_complete destination=%s", destination)
    return destination


def parse_args(argv: Sequence[str] | None = None) -> DownloadConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Download the complete mvasil/polyvore-outfits dataset to a "
            "local directory. Existing unchanged files are reused."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"destination directory (default: {DEFAULT_DATASET_ROOT})",
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
    token: bool | str | None = (
        False if arguments.no_token else arguments.token or True
    )
    config = DownloadConfig(
        output_dir=arguments.output_dir,
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
