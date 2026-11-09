"""API-oriented configuration views."""

from __future__ import annotations

from typing import Any

from sntx_sem.config.diagnostics import index_embedding_info
from sntx_sem.config.embedding import (
    DEFAULT_EMBEDDING_MODELS,
    EMBEDDING_PROVIDER_CHOICES,
    resolve_embedding_provider,
)
from sntx_sem.config.models import AppConfig


def embedding_settings_view(cfg: AppConfig) -> dict[str, Any]:
    """Non-secret embedding settings for API and Web-UI."""
    emb = cfg.embedding
    index_info = index_embedding_info(cfg)
    return {
        "provider": resolve_embedding_provider(emb),
        "model": emb.model,
        "device": emb.device,
        "base_url": emb.base_url or "",
        "api_key_set": bool(emb.resolved_api_key),
        "api_key_env": emb.api_key_env,
        "query_prefix": emb.query_prefix,
        "passage_prefix": emb.passage_prefix,
        "embedding_mismatch": index_info.get("embedding_mismatch", False),
        "index_embedding_model": index_info.get("embedding_model"),
        "index_embedding_provider": index_info.get("embedding_provider"),
        "providers": list(EMBEDDING_PROVIDER_CHOICES),
        "default_models": dict(DEFAULT_EMBEDDING_MODELS),
        "config_writable": bool(cfg.config_path and cfg.config_path.is_file()),
    }
