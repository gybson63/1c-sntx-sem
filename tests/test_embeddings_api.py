"""Tests for OpenAI-compatible embedding API backend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from sntx_sem.config import EmbeddingConfig
from sntx_sem.embeddings.openai_compatible import OpenAICompatibleBackend


def _mock_response(vectors: list[list[float]]) -> MagicMock:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "data": [{"index": i, "embedding": vec} for i, vec in enumerate(vectors)],
    }
    return response


@patch("sntx_sem.embeddings.openai_compatible.httpx.Client")
def test_embed_passages_batches(mock_client_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.post.return_value = _mock_response([[1.0, 0.0], [0.0, 1.0]])

    cfg = EmbeddingConfig(
        provider="openai_compatible",
        base_url="https://api.example.com/v1",
        model="embed-model",
        batch_size=64,
        query_prefix="",
        passage_prefix="",
    )
    backend = OpenAICompatibleBackend(cfg)
    vectors = backend.embed_passages(["alpha", "beta"])

    assert vectors.shape == (2, 2)
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args.kwargs
    assert call_kwargs["json"]["input"] == ["alpha", "beta"]


@patch("sntx_sem.embeddings.openai_compatible.httpx.Client")
def test_embed_query_applies_prefix(mock_client_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.post.return_value = _mock_response([[0.6, 0.8]])

    cfg = EmbeddingConfig(
        provider="openai_compatible",
        base_url="https://api.example.com/v1",
        model="embed-model",
        query_prefix="query: ",
        passage_prefix="passage: ",
    )
    backend = OpenAICompatibleBackend(cfg)
    vector = backend.embed_query("test")

    assert vector.shape == (2,)
    np.testing.assert_allclose(np.linalg.norm(vector), 1.0, rtol=1e-5)
    call_kwargs = mock_client.post.call_args.kwargs
    assert call_kwargs["json"]["input"] == ["query: test"]


@patch("sntx_sem.embeddings.openai_compatible.httpx.Client")
def test_embed_query_reuses_http_client(mock_client_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.post.return_value = _mock_response([[1.0, 0.0]])

    cfg = EmbeddingConfig(
        provider="openai_compatible",
        base_url="https://api.example.com/v1",
        model="embed-model",
        query_prefix="",
        passage_prefix="",
    )
    backend = OpenAICompatibleBackend(cfg)
    backend.embed_query("alpha")
    backend.embed_query("beta")

    mock_client_cls.assert_called_once()
    assert mock_client.post.call_count == 2


@patch("sntx_sem.embeddings.openai_compatible.httpx.Client")
def test_embed_query_cache_skips_second_request(mock_client_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.post.return_value = _mock_response([[1.0, 0.0]])

    cfg = EmbeddingConfig(
        provider="openai_compatible",
        base_url="https://api.example.com/v1",
        model="embed-model",
        query_prefix="",
        passage_prefix="",
    )
    backend = OpenAICompatibleBackend(cfg)
    backend.embed_query("same")
    backend.embed_query("same")

    mock_client.post.assert_called_once()


@patch("sntx_sem.embeddings.openai_compatible.httpx.Client")
def test_ollama_backend_uses_default_base_url(mock_client_cls: MagicMock) -> None:
    from sntx_sem.embeddings.factory import create_embedding_backend

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.post.return_value = _mock_response([[1.0, 0.0]])

    backend = create_embedding_backend(
        EmbeddingConfig(
            provider="ollama", model="nomic-embed-text", query_prefix="", passage_prefix=""
        )
    )
    backend.embed_query("hello")

    url = mock_client.post.call_args.args[0]
    assert url == "http://127.0.0.1:11434/v1/embeddings"
