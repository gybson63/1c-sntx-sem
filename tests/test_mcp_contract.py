"""Contract tests: MCP in-process vs thin HTTP parity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sntx_sem.config import AppConfig, SearchConfig
from sntx_sem.index.store import HelpIndex
from sntx_sem.mcp.client import SntxSemApiError
from sntx_sem.mcp.stdio_server import _dump
from sntx_sem.search_service import HelpSearchService

SEARCH_RESULT_KEYS = {
    "id",
    "domain",
    "title",
    "score",
    "entity_kind",
    "html_path",
    "excerpt",
    "excerpt_start",
    "excerpt_end",
    "highlight_terms",
    "match_sources",
    "match_explanation",
    "semantic_excerpt",
    "semantic_highlight_terms",
    "score_breakdown",
}

MCP_TOOL_NAMES = (
    "search_help",
    "search_bsl_syntax",
    "search_query_language",
    "get_topic",
    "find_examples",
    "list_domains",
)


class _FlatEmbeddingBackend:
    model_id = "test-flat"

    def embed_passages(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        del batch_size
        return np.ones((len(texts), 2), dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        del query
        return np.ones(2, dtype=np.float32)


class _ServiceApiClient:
    """Thin MCP client backed by HelpSearchService (no HTTP)."""

    def __init__(self, service: HelpSearchService) -> None:
        self._service = service

    def search(
        self,
        query: str,
        *,
        domain: str = "all",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        return self._service.search(query, domain=domain, limit=limit)

    def get_topic(
        self,
        topic_id: str,
        *,
        include_examples: bool = True,
    ) -> dict[str, Any]:
        topic = self._service.get_topic(topic_id, include_examples=include_examples)
        if topic is None:
            raise SntxSemApiError(f"Topic not found: {topic_id}", status_code=404)
        return topic

    def find_examples(
        self,
        *,
        query: str = "",
        topic_id: str = "",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        return self._service.find_examples(query=query, topic_id=topic_id, limit=limit)

    def stats(self) -> dict[str, Any]:
        return self._service.stats()

    def close(self) -> None:
        return None


def _write_chunks(path: Path) -> None:
    chunks = [
        {
            "id": "bsp:РазложитьМассивВСтроку",
            "domain": "bsp",
            "title_ru": "РазложитьМассивВСтроку",
            "title_en": "",
            "text": "Разбивает массив на строку с разделителем.",
            "entity_kind": "function",
            "html_path": "",
            "syntax": "",
        },
        {
            "id": "bsl:Если",
            "domain": "bsl_lang",
            "title_ru": "Если",
            "title_en": "If",
            "text": "Условный оператор Если.",
            "entity_kind": "topic",
            "html_path": "",
            "syntax": "",
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")


@pytest.fixture
def search_service(tmp_path: Path) -> HelpSearchService:
    jsonl = tmp_path / "chunks.jsonl"
    _write_chunks(jsonl)
    index_dir = tmp_path / "index"
    backend = _FlatEmbeddingBackend()
    index = HelpIndex(index_dir, backend, SearchConfig(final_top_k=5))
    index.build(index.load_chunks_from_jsonl(jsonl))

    cfg = AppConfig(
        index_dir=index_dir,
        data_dir=tmp_path / "data",
        export_dir=tmp_path / "export",
        hbk_dir=tmp_path / "hbk",
        search=SearchConfig(final_top_k=5),
    )
    service = HelpSearchService(cfg)
    service._index = index
    return service


def _in_process_json(service: HelpSearchService, tool: str, **kwargs: Any) -> str:
    """Same JSON serialization path as mcp_server.py tool handlers."""
    if tool == "search_help":
        payload = service.search(
            kwargs["query"],
            domain=kwargs.get("domain", "all"),
            limit=kwargs.get("limit", 5),
        )
    elif tool == "search_bsl_syntax":
        payload = service.search(kwargs["query"], domain="bsl", limit=kwargs.get("limit", 5))
    elif tool == "search_query_language":
        payload = service.search(kwargs["query"], domain="query", limit=kwargs.get("limit", 5))
    elif tool == "get_topic":
        topic = service.get_topic(
            kwargs["topic_id"],
            include_examples=kwargs.get("include_examples", True),
        )
        payload = topic if topic else {"error": f"Topic not found: {kwargs['topic_id']}"}
    elif tool == "find_examples":
        if not kwargs.get("query") and not kwargs.get("topic_id"):
            payload = {"error": "Provide query or topic_id"}
        else:
            payload = service.find_examples(
                query=kwargs.get("query", ""),
                topic_id=kwargs.get("topic_id", ""),
                limit=kwargs.get("limit", 5),
            )
    elif tool == "list_domains":
        payload = service.stats()
    else:
        raise ValueError(f"Unknown tool: {tool}")
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _thin_json(client: _ServiceApiClient, tool: str, **kwargs: Any) -> str:
    """Thin MCP JSON path via stdio_server helpers."""
    if tool == "search_help":
        return _dump(
            client.search(
                kwargs["query"],
                domain=kwargs.get("domain", "all"),
                limit=kwargs.get("limit", 5),
            )
        )
    if tool == "search_bsl_syntax":
        return _dump(client.search(kwargs["query"], domain="bsl", limit=kwargs.get("limit", 5)))
    if tool == "search_query_language":
        return _dump(client.search(kwargs["query"], domain="query", limit=kwargs.get("limit", 5)))
    if tool == "get_topic":
        try:
            return _dump(
                client.get_topic(
                    kwargs["topic_id"],
                    include_examples=kwargs.get("include_examples", True),
                )
            )
        except SntxSemApiError as exc:
            if exc.status_code == 404:
                return _dump({"error": f"Topic not found: {kwargs['topic_id']}"})
            raise
    if tool == "find_examples":
        if not kwargs.get("query") and not kwargs.get("topic_id"):
            return _dump({"error": "Provide query or topic_id"})
        return _dump(
            client.find_examples(
                query=kwargs.get("query", ""),
                topic_id=kwargs.get("topic_id", ""),
                limit=kwargs.get("limit", 5),
            )
        )
    if tool == "list_domains":
        return _dump(client.stats())
    raise ValueError(f"Unknown tool: {tool}")


def test_mcp_tool_names_registered_in_process() -> None:
    import sntx_sem.mcp_server as in_process

    for tool in MCP_TOOL_NAMES:
        assert hasattr(in_process, tool), f"in-process MCP missing tool {tool}"


def test_search_help_json_parity(search_service: HelpSearchService) -> None:
    client = _ServiceApiClient(search_service)
    kwargs = {"query": "разложить массив", "domain": "all", "limit": 3}
    assert _in_process_json(search_service, "search_help", **kwargs) == _thin_json(
        client, "search_help", **kwargs
    )


def test_search_query_language_json_parity(search_service: HelpSearchService) -> None:
    client = _ServiceApiClient(search_service)
    kwargs = {"query": "выборка", "limit": 2}
    assert _in_process_json(search_service, "search_query_language", **kwargs) == _thin_json(
        client, "search_query_language", **kwargs
    )


def test_search_bsl_syntax_json_parity(search_service: HelpSearchService) -> None:
    client = _ServiceApiClient(search_service)
    kwargs = {"query": "если", "limit": 2}
    assert _in_process_json(search_service, "search_bsl_syntax", **kwargs) == _thin_json(
        client, "search_bsl_syntax", **kwargs
    )


def test_get_topic_json_parity(search_service: HelpSearchService) -> None:
    client = _ServiceApiClient(search_service)
    kwargs = {"topic_id": "bsp:РазложитьМассивВСтроку"}
    assert _in_process_json(search_service, "get_topic", **kwargs) == _thin_json(
        client, "get_topic", **kwargs
    )


def test_get_topic_not_found_json_parity(search_service: HelpSearchService) -> None:
    client = _ServiceApiClient(search_service)
    kwargs = {"topic_id": "missing:topic"}
    assert _in_process_json(search_service, "get_topic", **kwargs) == _thin_json(
        client, "get_topic", **kwargs
    )


def test_find_examples_missing_args_json_parity(search_service: HelpSearchService) -> None:
    client = _ServiceApiClient(search_service)
    assert _in_process_json(search_service, "find_examples") == _thin_json(client, "find_examples")


def test_list_domains_json_parity(search_service: HelpSearchService) -> None:
    client = _ServiceApiClient(search_service)
    assert _in_process_json(search_service, "list_domains") == _thin_json(client, "list_domains")


def test_search_result_contract_keys(search_service: HelpSearchService) -> None:
    results = search_service.search("массив", limit=2)
    assert results
    assert SEARCH_RESULT_KEYS.issubset(results[0].keys())
