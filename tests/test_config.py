"""Tests for configuration loading."""

from __future__ import annotations

from sntx_sem.config import EmbeddingConfig, LLMConfig, resolve_api_key


def test_resolve_api_key_from_config_value() -> None:
    assert resolve_api_key("secret-key", "OPENAI_API_KEY") == "secret-key"


def test_resolve_api_key_from_env(monkeypatch) -> None:
    monkeypatch.setenv("MY_KEY", "from-env")
    assert resolve_api_key("", "MY_KEY") == "from-env"


def test_embedding_resolved_api_key_prefers_config() -> None:
    cfg = EmbeddingConfig(api_key="inline", api_key_env="OPENAI_API_KEY")
    assert cfg.resolved_api_key == "inline"


def test_llm_resolved_api_key_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    cfg = LLMConfig(api_key_env="DEEPSEEK_API_KEY")
    assert cfg.resolved_api_key == "ds-key"


def test_index_embedding_mismatch_when_model_differs() -> None:
    from sntx_sem.config import AppConfig, index_embedding_mismatch

    cfg = AppConfig(
        embedding=EmbeddingConfig(
            provider="openai_compatible",
            base_url="https://polza.ai/api/v1",
            model="text-embedding-3-small",
        )
    )
    index_meta = {"embedding_model": "intfloat/multilingual-e5-small"}
    assert index_embedding_mismatch(cfg, index_meta, indexed_count=100) is True


def test_index_embedding_mismatch_false_when_in_sync() -> None:
    from sntx_sem.config import AppConfig, index_embedding_mismatch

    cfg = AppConfig(
        embedding=EmbeddingConfig(
            provider="openai_compatible",
            model="text-embedding-3-small",
        )
    )
    index_meta = {
        "embedding_provider": "openai_compatible",
        "embedding_model": "text-embedding-3-small",
    }
    assert index_embedding_mismatch(cfg, index_meta, indexed_count=100) is False
