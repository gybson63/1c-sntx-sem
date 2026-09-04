"""Embedding provider resolution and index alignment."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from sntx_sem.config.models import AppConfig, EmbeddingConfig

LOCAL_EMBEDDING_PROVIDERS = frozenset({"sentence_transformers", "huggingface"})

EMBEDDING_PROVIDER_CHOICES = (
    "sentence_transformers",
    "openai_compatible",
    "ollama",
)

DEFAULT_EMBEDDING_MODELS = {
    "sentence_transformers": "intfloat/multilingual-e5-base",
    "openai_compatible": "text-embedding-3-small",
    "ollama": "nomic-embed-text",
}

OPENAI_EMBEDDING_MODELS = frozenset(
    {
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    }
)

LOCAL_EMBEDDING_MODELS = frozenset(
    {
        "intfloat/multilingual-e5-small",
        "intfloat/multilingual-e5-base",
        "intfloat/multilingual-e5-large",
    }
)


def resolve_embedding_provider(cfg: EmbeddingConfig) -> str:
    """Resolve embedding provider: explicit > base_url > sentence_transformers."""
    if cfg.provider:
        if cfg.provider == "huggingface":
            return "sentence_transformers"
        return cfg.provider
    if cfg.base_url:
        return "openai_compatible"
    return "sentence_transformers"


def normalize_embedding_provider(provider: str) -> str:
    """Map legacy huggingface alias to sentence_transformers."""
    if provider == "huggingface":
        return "sentence_transformers"
    return provider


def default_embedding_model(provider: str) -> str:
    normalized = normalize_embedding_provider(provider)
    return DEFAULT_EMBEDDING_MODELS.get(
        normalized, DEFAULT_EMBEDDING_MODELS["sentence_transformers"]
    )


def coerce_embedding_model(
    provider: str,
    model: str,
    *,
    previous_provider: str | None = None,
) -> tuple[str, str | None]:
    """Return model for provider; second value is a user-facing adjustment note."""
    normalized = normalize_embedding_provider(provider)
    cleaned = model.strip()
    if not cleaned:
        resolved = default_embedding_model(normalized)
        return resolved, None

    if normalized == "sentence_transformers" and cleaned in OPENAI_EMBEDDING_MODELS:
        resolved = default_embedding_model(normalized)
        return resolved, (
            f"Модель {cleaned!r} — для OpenAI API; для локального провайдера "
            f"используется {resolved!r}"
        )

    if (
        normalized in {"openai_compatible", "ollama"}
        and cleaned in LOCAL_EMBEDDING_MODELS
        and previous_provider in {None, "sentence_transformers", "huggingface"}
    ):
        resolved = default_embedding_model(normalized)
        return resolved, (
            f"Модель {cleaned!r} — для локального E5; для {normalized} используется {resolved!r}"
        )

    return cleaned, None


def index_embedding_mismatch(
    cfg: AppConfig,
    index_meta: dict[str, Any],
    *,
    indexed_count: int,
) -> bool:
    """True when config embedding settings differ from the built index metadata."""
    if indexed_count <= 0:
        return False

    current_provider = normalize_embedding_provider(resolve_embedding_provider(cfg.embedding))
    current_model = cfg.embedding.model
    built_provider = index_meta.get("embedding_provider")
    built_model = index_meta.get("embedding_model")

    if (
        built_provider is not None
        and normalize_embedding_provider(str(built_provider)) != current_provider
    ):
        return True
    return built_model is not None and built_model != current_model


def align_embedding_with_index(cfg: AppConfig) -> EmbeddingConfig:
    """Use index build metadata for query embeddings when config.yaml differs."""
    from sntx_sem.index.meta import load_index_meta

    meta = load_index_meta(cfg.index_dir)
    indexed_count = int(meta.get("indexed_count", 0))
    if indexed_count <= 0 or not index_embedding_mismatch(cfg, meta, indexed_count=indexed_count):
        return cfg.embedding

    built_provider = normalize_embedding_provider(str(meta.get("embedding_provider", "")))
    built_model = str(meta.get("embedding_model", ""))
    if not built_model:
        return cfg.embedding

    aligned = replace(
        cfg.embedding,
        provider=built_provider,
        model=built_model,
    )
    if built_model.startswith("intfloat/multilingual-e5"):
        aligned = replace(
            aligned,
            query_prefix="query: ",
            passage_prefix="passage: ",
        )
    return aligned
