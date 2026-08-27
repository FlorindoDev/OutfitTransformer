"""Synchronous client for OpenRouter embedding models."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch
from torch import Tensor

from .config import DEFAULT_MODEL_CONFIG


class OpenRouterEmbeddingError(RuntimeError):
    """Raised when OpenRouter cannot return valid embedding vectors."""


class OpenRouterEmbeddingClient:
    """Send validated, batched requests to OpenRouter's embedding endpoint."""

    DEFAULT_API_BASE = DEFAULT_MODEL_CONFIG.encoders.openrouter_api_base
    _RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 524, 529})

    def __init__(
        self,
        model_name: str,
        api_key: str,
        *,
        output_dim: int = DEFAULT_MODEL_CONFIG.encoders.openrouter_output_dim,
        request_batch_size: int = (
            DEFAULT_MODEL_CONFIG.encoders.openrouter_request_batch_size
        ),
        timeout_seconds: float = (
            DEFAULT_MODEL_CONFIG.encoders.openrouter_timeout_seconds
        ),
        max_retries: int = DEFAULT_MODEL_CONFIG.encoders.openrouter_max_retries,
        api_base: str = DEFAULT_API_BASE,
    ) -> None:
        self.model_name = _require_text(model_name, "model_name")
        self._api_key = _require_text(api_key, "api_key")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if request_batch_size <= 0:
            raise ValueError("request_batch_size must be positive")
        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self.output_dim = output_dim
        self.request_batch_size = request_batch_size
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.api_base = _require_text(api_base, "api_base").rstrip("/")

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Return one embedding for every non-empty input string."""
        normalized = tuple(_require_text(text, "text") for text in texts)
        if not normalized:
            raise ValueError("texts cannot be empty")
        return self._embed_inputs(normalized, device=device)

    def embed_multimodal(
        self,
        inputs: Sequence[Mapping[str, Any]],
        *,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Return embeddings for OpenRouter multimodal content objects."""
        normalized = tuple(dict(value) for value in inputs)
        if not normalized:
            raise ValueError("inputs cannot be empty")
        return self._embed_inputs(normalized, device=device)

    def _embed_inputs(
        self,
        inputs: Sequence[str | Mapping[str, Any]],
        *,
        device: torch.device | str | None,
    ) -> Tensor:
        chunks: list[Tensor] = []
        for start in range(0, len(inputs), self.request_batch_size):
            batch = inputs[start : start + self.request_batch_size]
            response = self._request(batch)
            chunks.append(
                _parse_embeddings(
                    response,
                    expected_count=len(batch),
                    expected_dim=self.output_dim,
                )
            )
        return torch.cat(chunks, dim=0).to(device=device)

    def _request(
        self,
        inputs: Sequence[str | Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        payload = json.dumps(
            {
                "model": self.model_name,
                "input": inputs,
                "dimensions": self.output_dim,
                "encoding_format": "float",
            },
            separators=(",", ":"),
        ).encode("utf-8")
        request = Request(
            f"{self.api_base}/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "OutfitTransformer/1.0",
            },
            method="POST",
        )

        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    return _decode_response(response.read())
            except HTTPError as error:
                if (
                    error.code in self._RETRYABLE_STATUS_CODES
                    and attempt < self.max_retries
                ):
                    time.sleep(_retry_delay(error, attempt))
                    continue
                detail = _decode_error_detail(error.read())
                raise OpenRouterEmbeddingError(
                    f"OpenRouter embedding request failed with HTTP "
                    f"{error.code}: {detail}"
                ) from error
            except URLError as error:
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
                raise OpenRouterEmbeddingError(
                    f"OpenRouter embedding request failed: {error.reason}"
                ) from error

        raise AssertionError("OpenRouter retry loop ended unexpectedly")


def _parse_embeddings(
    payload: Mapping[str, Any],
    *,
    expected_count: int,
    expected_dim: int,
) -> Tensor:
    raw_data = payload.get("data")
    if not isinstance(raw_data, list) or len(raw_data) != expected_count:
        raise OpenRouterEmbeddingError(
            "OpenRouter response contains the wrong number of embeddings"
        )

    vectors_by_index: dict[int, list[float]] = {}
    for position, item in enumerate(raw_data):
        if not isinstance(item, Mapping):
            raise OpenRouterEmbeddingError(
                "OpenRouter embedding entry must be an object"
            )
        index = item.get("index", position)
        if not isinstance(index, int) or isinstance(index, bool):
            raise OpenRouterEmbeddingError(
                "OpenRouter embedding index must be an integer"
            )
        if index in vectors_by_index:
            raise OpenRouterEmbeddingError(
                f"OpenRouter response contains duplicate index {index}"
            )
        raw_vector = item.get("embedding")
        if not isinstance(raw_vector, list) or len(raw_vector) != expected_dim:
            raise OpenRouterEmbeddingError(
                f"OpenRouter embedding must contain {expected_dim} values"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in raw_vector
        ):
            raise OpenRouterEmbeddingError(
                "OpenRouter embedding must contain only numbers"
            )
        vectors_by_index[index] = [float(value) for value in raw_vector]

    expected_indices = set(range(expected_count))
    if set(vectors_by_index) != expected_indices:
        raise OpenRouterEmbeddingError(
            "OpenRouter embedding indices are incomplete"
        )
    embeddings = torch.tensor(
        [vectors_by_index[index] for index in range(expected_count)],
        dtype=torch.float32,
    )
    if not bool(torch.isfinite(embeddings).all()):
        raise OpenRouterEmbeddingError(
            "OpenRouter embeddings must contain only finite values"
        )
    return embeddings


def _decode_response(body: bytes) -> Mapping[str, Any]:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OpenRouterEmbeddingError(
            "OpenRouter returned an invalid JSON response"
        ) from error
    if not isinstance(payload, Mapping):
        raise OpenRouterEmbeddingError("OpenRouter response must be an object")
    return payload


def _decode_error_detail(body: bytes) -> str:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "invalid error response"
    if not isinstance(payload, Mapping):
        return "invalid error response"
    error = payload.get("error")
    message = error.get("message") if isinstance(error, Mapping) else None
    if isinstance(message, str):
        return message
    return "unknown API error"


def _retry_delay(error: HTTPError, attempt: int) -> float:
    retry_after = error.headers.get("Retry-After") if error.headers else None
    if retry_after is not None:
        try:
            return min(max(float(retry_after), 0.0), 60.0)
        except ValueError:
            pass
    return float(2**attempt)


def _require_text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized
