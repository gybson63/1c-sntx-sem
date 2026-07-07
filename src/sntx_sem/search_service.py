"""Shared search logic for MCP and HTTP API."""

from __future__ import annotations

import re
from typing import Any

from sntx_sem.config import AppConfig, align_embedding_with_index
from sntx_sem.embeddings import create_embedding_backend
from sntx_sem.examples.store import ExamplesStore
from sntx_sem.index.store import HelpIndex

_EXCERPT_LIMIT = 500


def _find_first_term_position(text: str, terms: list[str]) -> int | None:
    if not text or not terms:
        return None
    lowered = text.lower()
    found_positions = [lowered.find(term.lower()) for term in terms if term]
    positions = [pos for pos in found_positions if pos >= 0]
    if not positions:
        return None
    return min(positions)


def _build_excerpt(
    text: str, terms: list[str], max_chars: int = _EXCERPT_LIMIT
) -> tuple[str, int, int]:
    if not text:
        return "", 0, 0

    if len(text) <= max_chars:
        return text, 0, len(text)

    match_pos = _find_first_term_position(text, terms)
    if match_pos is None:
        snippet = text[:max_chars]
        return snippet + "...", 0, len(snippet)

    half_window = max_chars // 2
    start = max(0, match_pos - half_window)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)
    snippet = text[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet, start, end


def _round_breakdown_value(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) or value is None:
        return value
    if isinstance(value, float):
        return round(value, 4)
    return None


def format_search_result(result: Any, query: str) -> dict[str, Any]:
    highlight_terms = list(getattr(result, "highlight_terms", []) or [])
    if not highlight_terms:
        highlight_terms = re.findall(r"[a-zа-яё0-9]+", query.lower())
    excerpt, excerpt_start, excerpt_end = _build_excerpt(result.text, highlight_terms)
    score_breakdown = {
        key: _round_breakdown_value(value)
        for key, value in dict(getattr(result, "score_breakdown", {}) or {}).items()
    }
    return {
        "id": result.id,
        "domain": result.domain,
        "title": result.title,
        "score": round(result.score, 4),
        "entity_kind": result.entity_kind,
        "html_path": result.html_path,
        "excerpt": excerpt,
        "excerpt_start": excerpt_start,
        "excerpt_end": excerpt_end,
        "highlight_terms": highlight_terms,
        "match_sources": list(getattr(result, "match_sources", []) or []),
        "match_explanation": getattr(result, "match_explanation", "") or "",
        "semantic_excerpt": getattr(result, "semantic_excerpt", "") or "",
        "semantic_highlight_terms": list(getattr(result, "semantic_highlight_terms", []) or []),
        "score_breakdown": score_breakdown,
    }


class HelpSearchService:
    """Lazy-loaded index and examples for search endpoints."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._index: HelpIndex | None = None
        self._examples: ExamplesStore | None = None

    @property
    def config(self) -> AppConfig:
        return self._config

    def update_config(self, config: AppConfig) -> None:
        """Apply new config and drop cached index (lazy reload on next search)."""
        self._config = config
        self._index = None

    def get_index(self) -> HelpIndex:
        if self._index is None:
            emb_cfg = align_embedding_with_index(self._config)
            backend = create_embedding_backend(emb_cfg)
            self._index = HelpIndex(self._config.index_dir, backend, self._config.search)
        return self._index

    def get_examples(self) -> ExamplesStore:
        if self._examples is None:
            self._examples = ExamplesStore(self._config.data_dir / "examples.jsonl")
        return self._examples

    def warm_index(self) -> None:
        """Load LanceDB index and embedding model (first search otherwise blocks ~1 min)."""
        index = self.get_index()
        index._ensure_loaded()
        index.embedding_backend.embed_query("warmup")

    def search(self, query: str, domain: str = "all", limit: int = 5) -> list[dict[str, Any]]:
        results = self.get_index().search(query, domain=domain, limit=limit)
        return [format_search_result(r, query) for r in results]

    def get_topic(self, topic_id: str, *, include_examples: bool = True) -> dict[str, Any] | None:
        topic = self.get_index().get_topic(topic_id)
        if not topic:
            return None
        result = dict(topic)
        if include_examples:
            examples = self.get_examples().find_by_topic(topic_id)
            result["examples"] = [ex.to_dict() for ex in examples]
        return result

    def stats(self) -> dict[str, Any]:
        stats = self.get_index().stats()
        stats["examples"] = self.get_examples().count
        return stats

    def find_examples(
        self,
        *,
        query: str = "",
        topic_id: str = "",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        store = self.get_examples()
        if topic_id:
            examples = store.find_by_topic(topic_id, limit=limit)
        elif query:
            examples = store.search(query, limit=limit)
        else:
            return []
        return [ex.to_dict() for ex in examples]
