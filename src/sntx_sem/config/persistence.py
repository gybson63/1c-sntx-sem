"""Persist configuration settings to YAML."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from sntx_sem.config.embedding import (
    EMBEDDING_PROVIDER_CHOICES,
    coerce_embedding_model,
    normalize_embedding_provider,
    resolve_embedding_provider,
)
from sntx_sem.config.loader import load_config
from sntx_sem.config.models import AppConfig
from sntx_sem.config.validation import ConfigError


def save_embedding_settings(
    cfg: AppConfig, updates: dict[str, Any]
) -> tuple[AppConfig, str | None]:
    """Update embedding section in config.yaml and reload config."""
    if cfg.config_path is None or not cfg.config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg.config_path}")

    with cfg.config_path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ConfigError("корень YAML должен быть объектом", path=cfg.config_path)

    emb_raw: dict[str, Any] = dict(raw.get("embedding") or {})
    previous_provider = resolve_embedding_provider(cfg.embedding)

    provider = updates.get("provider")
    if provider is not None:
        if provider not in EMBEDDING_PROVIDER_CHOICES:
            raise ConfigError(
                f"неизвестный provider: {provider!r}",
                field="embedding.provider",
                path=cfg.config_path,
            )
        emb_raw["provider"] = provider

    next_provider = normalize_embedding_provider(str(emb_raw.get("provider") or previous_provider))
    if "model" in updates and updates["model"] is not None:
        requested_model = str(updates["model"])
    else:
        requested_model = str(emb_raw.get("model") or cfg.embedding.model)

    resolved_model, model_adjustment = coerce_embedding_model(
        next_provider,
        requested_model,
        previous_provider=previous_provider if provider is not None else None,
    )
    emb_raw["model"] = resolved_model

    for key in ("device", "base_url", "api_key_env", "query_prefix", "passage_prefix"):
        if key in updates and updates[key] is not None:
            emb_raw[key] = updates[key]

    api_key = updates.get("api_key")
    if api_key:
        emb_raw["api_key"] = api_key

    raw["embedding"] = emb_raw

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{cfg.config_path.name}.",
        suffix=".tmp",
        dir=cfg.config_path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                raw, handle, allow_unicode=True, default_flow_style=False, sort_keys=False
            )
            handle.flush()
            os.fsync(handle.fileno())
        load_config(tmp_path)
        os.replace(tmp_path, cfg.config_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return load_config(cfg.config_path), model_adjustment
