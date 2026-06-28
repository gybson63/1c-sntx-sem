"""OpenAI-compatible embeddings API backend."""

from __future__ import annotations

import time

import httpx
import numpy as np

from sntx_sem.config import EmbeddingConfig
from sntx_sem.embeddings.base import normalize_vectors

_RETRIABLE_ERRORS = (
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    httpx.WriteTimeout,
    httpx.NetworkError,
)
_MAX_RETRIES = 3


class OpenAICompatibleBackend:
    _QUERY_CACHE_MAX = 128

    def __init__(self, cfg: EmbeddingConfig) -> None:
        if not cfg.base_url:
            raise ValueError("embedding.base_url is required for openai_compatible provider")
        self._cfg = cfg
        self._base_url = cfg.base_url.rstrip("/")
        self._client: httpx.Client | None = None
        self._query_cache: dict[str, np.ndarray] = {}

    @property
    def model_id(self) -> str:
        return self._cfg.model

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            timeout = httpx.Timeout(
                connect=30.0,
                read=self._cfg.timeout,
                write=30.0,
                pool=30.0,
            )
            self._client = httpx.Client(timeout=timeout)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _request_batch(self, inputs: list[str]) -> np.ndarray:
        headers = {"Content-Type": "application/json"}
        api_key = self._cfg.resolved_api_key
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": self._cfg.model,
            "input": inputs,
        }
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._get_client().post(
                    f"{self._base_url}/embeddings",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                items = sorted(data["data"], key=lambda item: item["index"])
                vectors = np.asarray([item["embedding"] for item in items], dtype=np.float32)
                return normalize_vectors(vectors)
            except _RETRIABLE_ERRORS as exc:
                last_error = exc
                if attempt + 1 >= _MAX_RETRIES:
                    break
                time.sleep(min(2**attempt, 30))
        assert last_error is not None
        raise last_error

    def embed_passages(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        bs = batch_size or self._cfg.batch_size
        prefix = self._cfg.passage_prefix
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), bs):
            batch = texts[start : start + bs]
            prefixed = [f"{prefix}{t[:2000]}" if prefix else t[:2000] for t in batch]
            batches.append(self._request_batch(prefixed))
        return np.vstack(batches)

    def embed_query(self, query: str) -> np.ndarray:
        cached = self._query_cache.get(query)
        if cached is not None:
            return cached

        prefix = self._cfg.query_prefix
        text = f"{prefix}{query}" if prefix else query
        vectors = self._request_batch([text])
        vector = vectors[0]
        if len(self._query_cache) >= self._QUERY_CACHE_MAX:
            self._query_cache.clear()
        self._query_cache[query] = vector
        return vector  # type: ignore[no-any-return]
