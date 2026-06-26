"""Factory for embedding backends."""

from __future__ import annotations

from sntx_sem.config import EmbeddingConfig, resolve_embedding_provider
from sntx_sem.embeddings.base import EmbeddingBackend
from sntx_sem.embeddings.huggingface import HuggingFaceBackend
from sntx_sem.embeddings.ollama import OllamaBackend
from sntx_sem.embeddings.openai_compatible import OpenAICompatibleBackend


def create_embedding_backend(cfg: EmbeddingConfig) -> EmbeddingBackend:
    provider = resolve_embedding_provider(cfg)
    if provider in ("sentence_transformers", "huggingface"):
        return HuggingFaceBackend(cfg)
    if provider == "openai_compatible":
        return OpenAICompatibleBackend(cfg)
    if provider == "ollama":
        return OllamaBackend(cfg)
    raise ValueError(f"Unknown embedding provider: {provider}")
