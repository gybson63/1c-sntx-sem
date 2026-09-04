"""Non-secret configuration summaries."""

from __future__ import annotations

from typing import Any

from sntx_sem.config.embedding import resolve_embedding_provider
from sntx_sem.config.models import AppConfig


def config_summary(cfg: AppConfig) -> dict[str, Any]:
    """Non-secret view of active settings from config.yaml."""
    embedding_provider = resolve_embedding_provider(cfg.embedding)
    return {
        "config_file": str(cfg.config_path) if cfg.config_path else None,
        "embedding": {
            "provider": embedding_provider,
            "model": cfg.embedding.model,
            "base_url": cfg.embedding.base_url or None,
            "api_key_set": bool(cfg.embedding.resolved_api_key),
        },
        "llm": {
            "provider": cfg.llm.provider,
            "model": cfg.llm.model,
            "base_url": cfg.llm.base_url or None,
            "api_key_set": bool(cfg.llm.resolved_api_key),
        },
        "mcp": {
            "log_level": cfg.mcp.log_level,
            "log_file": cfg.mcp.log_file or None,
        },
        "bsp": {
            "path": cfg.bsp.path or None,
            "enabled": cfg.bsp.enabled,
        },
    }
