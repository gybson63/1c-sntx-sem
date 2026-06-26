"""Embedding backends for semantic search."""

from sntx_sem.embeddings.base import EmbeddingBackend
from sntx_sem.embeddings.factory import create_embedding_backend

__all__ = ["EmbeddingBackend", "create_embedding_backend"]
