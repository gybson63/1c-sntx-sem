"""Tests for embedding provider resolution and factory."""

from __future__ import annotations

import pytest

from sntx_sem.config import EmbeddingConfig, resolve_embedding_provider
from sntx_sem.embeddings.factory import create_embedding_backend
from sntx_sem.embeddings.huggingface import HuggingFaceBackend
from sntx_sem.embeddings.ollama import OllamaBackend
from sntx_sem.embeddings.openai_compatible import OpenAICompatibleBackend


def test_resolve_provider_defaults_to_sentence_transformers() -> None:
    cfg = EmbeddingConfig()
    assert resolve_embedding_provider(cfg) == "sentence_transformers"


def test_resolve_provider_huggingface_alias() -> None:
    cfg = EmbeddingConfig(provider="huggingface")
    assert resolve_embedding_provider(cfg) == "sentence_transformers"


def test_resolve_provider_from_base_url() -> None:
    cfg = EmbeddingConfig(base_url="https://api.openai.com/v1")
    assert resolve_embedding_provider(cfg) == "openai_compatible"


def test_resolve_provider_explicit() -> None:
    cfg = EmbeddingConfig(provider="ollama")
    assert resolve_embedding_provider(cfg) == "ollama"


def test_factory_sentence_transformers() -> None:
    backend = create_embedding_backend(EmbeddingConfig())
    assert isinstance(backend, HuggingFaceBackend)
    assert backend.model_id == "intfloat/multilingual-e5-base"


def test_factory_huggingface_alias() -> None:
    backend = create_embedding_backend(EmbeddingConfig(provider="huggingface"))
    assert isinstance(backend, HuggingFaceBackend)


def test_factory_openai_compatible() -> None:
    cfg = EmbeddingConfig(
        provider="openai_compatible",
        base_url="https://api.example.com/v1",
        model="text-embedding-3-small",
    )
    backend = create_embedding_backend(cfg)
    assert isinstance(backend, OpenAICompatibleBackend)


def test_factory_ollama() -> None:
    backend = create_embedding_backend(EmbeddingConfig(provider="ollama", model="nomic-embed-text"))
    assert isinstance(backend, OllamaBackend)


def test_factory_unknown_provider() -> None:
    cfg = EmbeddingConfig(provider="unknown")
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        create_embedding_backend(cfg)
