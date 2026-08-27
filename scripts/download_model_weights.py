"""Download pretrained encoder weights into the project directory."""

from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from model import DEFAULT_MODEL_CONFIG

LOGGER = logging.getLogger("download_model_weights")
DEFAULT_OUTPUT_DIR = Path("pretrained_models")
_HUGGING_FACE_RUNTIME_PATTERNS = (
    "*.bin",
    "*.json",
    "*.model",
    "*.safetensors",
    "*.spm",
    "*.txt",
)
_HUGGING_FACE_IGNORED_PATTERNS = (
    "onnx/*",
    "openvino/*",
    "*.h5",
    "*.msgpack",
    "*.onnx",
    "*.ot",
    "*.xml",
)


@dataclass(frozen=True)
class DownloadConfig:
    """Model identifiers and authentication used by one download job."""

    fashion_clip_model_name: str
    sentence_transformer_model_name: str
    output_dir: Path
    token: bool | str | None

    def validate(self) -> None:
        _require_model_name(
            self.fashion_clip_model_name,
            "fashion_clip_model_name",
        )
        _require_model_name(
            self.sentence_transformer_model_name,
            "sentence_transformer_model_name",
        )
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise ValueError(
                f"output_dir must be a directory: {self.output_dir}"
            )


@dataclass(frozen=True)
class DownloadResult:
    """Locations and identifiers of downloaded encoder artifacts."""

    fashion_clip_directory: Path
    resnet18_checkpoint: Path
    sentence_transformer_directory: Path


def download_model_weights(config: DownloadConfig) -> DownloadResult:
    """Download all pretrained encoders required by local feature modes."""
    config.validate()
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fashion_clip_directory = _download_fashion_clip(
        config.fashion_clip_model_name,
        output_dir / _model_slug(config.fashion_clip_model_name),
        config.token,
    )
    resnet18_checkpoint = _download_resnet18(output_dir / "resnet18")
    sentence_transformer_directory = _download_sentence_transformer(
        config.sentence_transformer_model_name,
        output_dir / _model_slug(config.sentence_transformer_model_name),
        config.token,
    )

    result = DownloadResult(
        fashion_clip_directory=fashion_clip_directory,
        resnet18_checkpoint=resnet18_checkpoint,
        sentence_transformer_directory=sentence_transformer_directory,
    )
    LOGGER.info(
        "download_complete fashion_clip=%s resnet18=%s sentence_transformer=%s",
        result.fashion_clip_directory,
        result.resnet18_checkpoint,
        result.sentence_transformer_directory,
    )
    return result


def _download_fashion_clip(
    model_name: str,
    destination: Path,
    token: bool | str | None,
) -> Path:
    try:
        from transformers import AutoImageProcessor, AutoTokenizer, CLIPModel
    except ImportError as error:
        raise ImportError(
            "FashionCLIP download requires the 'transformers' package"
        ) from error

    LOGGER.info("download_start encoder=fashion_clip model=%s", model_name)
    local_directory = _download_hugging_face_snapshot(
        model_name,
        destination,
        token,
    )
    processor = AutoImageProcessor.from_pretrained(
        local_directory,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        local_directory,
        local_files_only=True,
    )
    model = CLIPModel.from_pretrained(
        local_directory,
        local_files_only=True,
    )
    del processor, tokenizer, model
    LOGGER.info(
        "download_ready encoder=fashion_clip directory=%s",
        local_directory,
    )
    return local_directory


def _download_resnet18(destination: Path) -> Path:
    try:
        import torch
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as error:
        raise ImportError(
            "ResNet-18 download requires the 'torch' and 'torchvision' packages"
        ) from error

    weights = ResNet18_Weights.DEFAULT
    LOGGER.info("download_start encoder=resnet18 weights=%s", weights.name)
    destination.mkdir(parents=True, exist_ok=True)
    state_dict = torch.hub.load_state_dict_from_url(
        weights.url,
        model_dir=str(destination.resolve()),
        map_location="cpu",
        progress=True,
        check_hash=True,
        weights_only=True,
    )
    model = resnet18(weights=None)
    model.load_state_dict(state_dict, strict=True)
    del model, state_dict

    filename = Path(urlparse(weights.url).path).name
    checkpoint = (destination / filename).resolve()
    if not checkpoint.is_file():
        raise RuntimeError(
            f"ResNet-18 download completed but checkpoint is missing: {checkpoint}"
        )
    LOGGER.info("download_ready encoder=resnet18 checkpoint=%s", checkpoint)
    return checkpoint


def _download_sentence_transformer(
    model_name: str,
    destination: Path,
    token: bool | str | None,
) -> Path:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError(
            "Sentence-BERT download requires the 'sentence-transformers' package"
        ) from error

    LOGGER.info(
        "download_start encoder=sentence_transformer model=%s",
        model_name,
    )
    local_directory = _download_hugging_face_snapshot(
        model_name,
        destination,
        token,
    )
    model = SentenceTransformer(
        str(local_directory),
        device="cpu",
        local_files_only=True,
    )
    del model
    LOGGER.info(
        "download_ready encoder=sentence_transformer directory=%s",
        local_directory,
    )
    return local_directory


def _download_hugging_face_snapshot(
    model_name: str,
    destination: Path,
    token: bool | str | None,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError(
            "model download requires the 'huggingface-hub' package"
        ) from error

    snapshot = Path(
        snapshot_download(
            repo_id=model_name,
            local_dir=destination,
            token=token,
            allow_patterns=_HUGGING_FACE_RUNTIME_PATTERNS,
            ignore_patterns=_HUGGING_FACE_IGNORED_PATTERNS,
        )
    ).resolve()
    if not snapshot.is_dir():
        raise RuntimeError(
            f"Hugging Face download directory is missing: {snapshot}"
        )
    return snapshot


def _model_slug(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-")
    if not slug:
        raise ValueError("model_name must contain a path-safe character")
    return slug


def _require_model_name(value: str, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} cannot be empty")


def parse_args(argv: Sequence[str] | None = None) -> DownloadConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Download FashionCLIP, ResNet-18 and Sentence-BERT weights into "
            "a local project directory."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"local destination root (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--fashion-clip-model-name",
        default=DEFAULT_MODEL_CONFIG.encoders.fashion_clip_model_name,
        help="Hugging Face FashionCLIP model identifier",
    )
    parser.add_argument(
        "--sentence-transformer-model-name",
        default=DEFAULT_MODEL_CONFIG.encoders.sentence_transformer_model_name,
        help="Hugging Face SentenceTransformer model identifier",
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
    token: bool | str | None = False if arguments.no_token else arguments.token
    config = DownloadConfig(
        fashion_clip_model_name=arguments.fashion_clip_model_name,
        sentence_transformer_model_name=arguments.sentence_transformer_model_name,
        output_dir=arguments.output_dir,
        token=token,
    )
    config.validate()
    return config


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    download_model_weights(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
