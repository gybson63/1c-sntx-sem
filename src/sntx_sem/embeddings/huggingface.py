"""Local HuggingFace sentence-transformers backend."""

from __future__ import annotations

import numpy as np

from sntx_sem.config import EmbeddingConfig


class HuggingFaceBackend:
    def __init__(self, cfg: EmbeddingConfig) -> None:
        self._cfg = cfg
        self._model = None

    @property
    def model_id(self) -> str:
        return self._cfg.model

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                msg = "Install local embeddings: pip install 'sntx-sem[embeddings]'"
                raise ImportError(msg) from exc

            self._model = SentenceTransformer(self._cfg.model, device=self._cfg.device)
        return self._model

    def embed_passages(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        model = self._load()
        bs = batch_size or self._cfg.batch_size
        prefixed = [f"{self._cfg.passage_prefix}{t[:2000]}" for t in texts]
        vectors = model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
            batch_size=bs,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        model = self._load()
        vector = model.encode(
            f"{self._cfg.query_prefix}{query}",
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vector, dtype=np.float32)
