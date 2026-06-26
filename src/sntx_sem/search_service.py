"""Shared search logic for MCP and HTTP API."""

from __future__ import annotations

from typing import Any

from sntx_sem.config import AppConfig
from sntx_sem.embeddings import create_embedding_backend
from sntx_sem.examples.store import ExamplesStore
from sntx_sem.index.store import HelpIndex


def format_search_result(result: Any) -> dict[str, Any]:
    return {
        "id": result.id,
        "domain": result.domain,
        "title": result.title,
        "score": round(result.score, 4),
        "entity_kind": result.entity_kind,
        "html_path": result.html_path,
        "excerpt": result.text[:500] + ("..." if len(result.text) > 500 else ""),
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

    def get_index(self) -> HelpIndex:
        if self._index is None:
            backend = create_embedding_backend(self._config.embedding)
            self._index = HelpIndex(self._config.index_dir, backend, self._config.search)
        return self._index

    def get_examples(self) -> ExamplesStore:
        if self._examples is None:
            self._examples = ExamplesStore(self._config.data_dir / "examples.jsonl")
        return self._examples

    def warm_index(self) -> None:
        self.get_index()._ensure_loaded()

    def search(self, query: str, domain: str = "all", limit: int = 5) -> list[dict[str, Any]]:
        results = self.get_index().search(query, domain=domain, limit=limit)
        return [format_search_result(r) for r in results]

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
