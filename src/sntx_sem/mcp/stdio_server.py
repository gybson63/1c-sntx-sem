"""MCP stdio server proxying sntx-sem HTTP API."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sntx_sem.mcp.client import SntxSemApiClient, SntxSemApiError

JsonText = str


def _dump(data: Any) -> JsonText:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _error_message(exc: Exception) -> JsonText:
    if isinstance(exc, SntxSemApiError):
        payload: dict[str, Any] = {"error": str(exc)}
        if exc.status_code is not None:
            payload["status_code"] = exc.status_code
        return _dump(payload)
    return _dump({"error": str(exc)})


def create_mcp_server(client: SntxSemApiClient | None = None) -> Any:
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        "1c-syntax-sem",
        instructions=(
            "Семантический поиск по справке платформы 1С через sntx-sem HTTP API. "
            "Workflow: search_help → get_topic → find_examples."
        ),
    )

    @contextmanager
    def _client() -> Iterator[SntxSemApiClient]:
        owned = client is None
        api = client or SntxSemApiClient()
        try:
            yield api
        finally:
            if owned:
                api.close()

    def _search_help(query: str, domain: str = "all", limit: int = 5) -> JsonText:
        with _client() as api:
            return _dump(api.search(query, domain=domain, limit=limit))

    @mcp.tool()
    def search_help(query: str, domain: str = "all", limit: int = 5) -> JsonText:
        """Semantic search over 1C platform help (BSL + query language)."""
        try:
            return _search_help(query, domain=domain, limit=limit)
        except Exception as exc:  # noqa: BLE001
            return _error_message(exc)

    @mcp.tool()
    def search_bsl_syntax(query: str, limit: int = 5) -> JsonText:
        """Search built-in 1C language (BSL) syntax help."""
        try:
            return _search_help(query, domain="bsl", limit=limit)
        except Exception as exc:  # noqa: BLE001
            return _error_message(exc)

    @mcp.tool()
    def search_query_language(query: str, limit: int = 5) -> JsonText:
        """Search 1C query language (SDBL) help."""
        try:
            return _search_help(query, domain="query", limit=limit)
        except Exception as exc:  # noqa: BLE001
            return _error_message(exc)

    @mcp.tool()
    def get_topic(topic_id: str, include_examples: bool = True) -> JsonText:
        """Get full help topic by ID, optionally with linked code examples."""
        try:
            with _client() as api:
                return _dump(api.get_topic(topic_id, include_examples=include_examples))
        except SntxSemApiError as exc:
            if exc.status_code == 404:
                return _dump({"error": f"Topic not found: {topic_id}"})
            return _error_message(exc)
        except Exception as exc:  # noqa: BLE001
            return _error_message(exc)

    @mcp.tool()
    def find_examples(query: str = "", topic_id: str = "", limit: int = 5) -> JsonText:
        """Find code examples from local configurations."""
        try:
            if not query and not topic_id:
                return _dump({"error": "Provide query or topic_id"})
            with _client() as api:
                return _dump(api.find_examples(query=query, topic_id=topic_id, limit=limit))
        except Exception as exc:  # noqa: BLE001
            return _error_message(exc)

    @mcp.tool()
    def list_domains() -> JsonText:
        """Return index statistics by domain."""
        try:
            with _client() as api:
                return _dump(api.stats())
        except Exception as exc:  # noqa: BLE001
            return _error_message(exc)

    return mcp


def run() -> None:
    mcp = create_mcp_server()
    mcp.run(transport="stdio")
