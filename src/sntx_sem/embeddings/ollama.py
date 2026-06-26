"""Ollama preset for OpenAI-compatible embeddings API."""

from __future__ import annotations

from dataclasses import replace

from sntx_sem.config import EmbeddingConfig
from sntx_sem.embeddings.openai_compatible import OpenAICompatibleBackend

OLLAMA_DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"


class OllamaBackend(OpenAICompatibleBackend):
    def __init__(self, cfg: EmbeddingConfig) -> None:
        ollama_cfg = replace(
            cfg,
            base_url=cfg.base_url or OLLAMA_DEFAULT_BASE_URL,
        )
        super().__init__(ollama_cfg)
        self._cfg = ollama_cfg
