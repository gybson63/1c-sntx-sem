"""Embedding backend protocol."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingBackend(Protocol):
    @property
    def model_id(self) -> str:
        """Model identifier used for index metadata and status."""

    def embed_passages(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """Embed document passages; returns shape (n, dim)."""

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query; returns shape (dim,)."""


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return (vectors / norms).astype(np.float32)  # type: ignore[no-any-return]
