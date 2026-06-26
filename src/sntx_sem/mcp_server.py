"""MCP server for 1C semantic help search."""

from __future__ import annotations

import json
import logging
import threading

from mcp.server.fastmcp import FastMCP

from sntx_sem.config import load_config
from sntx_sem.mcp_logging import install_mcp_logging
from sntx_sem.search_service import HelpSearchService

mcp = FastMCP("1c-syntax-sem")

_config = load_config()
_service = HelpSearchService(_config)


def _search_help(query: str, domain: str = "all", limit: int = 5) -> str:
    results = _service.search(query, domain=domain, limit=limit)
    return json.dumps(results, ensure_ascii=False, indent=2)


@mcp.tool()
def search_help(query: str, domain: str = "all", limit: int = 5) -> str:
    """Semantic search over 1C platform help (BSL + query language).

    Args:
        query: Natural language search query (Russian or English).
        domain: Filter — 'all', 'bsl', 'query', 'bsp', 'bsl_lang', 'query_lang', 'platform_api'.
        limit: Maximum number of results.
    """
    return _search_help(query, domain=domain, limit=limit)


@mcp.tool()
def search_bsl_syntax(query: str, limit: int = 5) -> str:
    """Search built-in 1C language (BSL) syntax help."""
    return _search_help(query, domain="bsl", limit=limit)


@mcp.tool()
def search_query_language(query: str, limit: int = 5) -> str:
    """Search 1C query language (SDBL) help."""
    return _search_help(query, domain="query", limit=limit)


@mcp.tool()
def get_topic(topic_id: str, include_examples: bool = True) -> str:
    """Get full help topic by ID, optionally with linked code examples."""
    topic = _service.get_topic(topic_id, include_examples=include_examples)
    if not topic:
        return json.dumps({"error": f"Topic not found: {topic_id}"}, ensure_ascii=False)
    return json.dumps(topic, ensure_ascii=False, indent=2)


@mcp.tool()
def find_examples(query: str = "", topic_id: str = "", limit: int = 5) -> str:
    """Find code examples from local configurations.

    Args:
        query: Natural language query to search examples.
        topic_id: Help topic ID to find linked examples.
        limit: Maximum results.
    """
    store = _service.get_examples()
    if topic_id:
        examples = store.find_by_topic(topic_id, limit=limit)
    elif query:
        examples = store.search(query, limit=limit)
    else:
        return json.dumps({"error": "Provide query or topic_id"}, ensure_ascii=False)

    return json.dumps([ex.to_dict() for ex in examples], ensure_ascii=False, indent=2)


@mcp.tool()
def list_domains() -> str:
    """Return index statistics by domain."""
    stats = _service.stats()
    return json.dumps(stats, ensure_ascii=False, indent=2)


def _warm_index() -> None:
    """Load LanceDB table and BM25 in background so first tool call is fast."""
    try:
        _service.warm_index()
    except Exception:
        logging.getLogger(__name__).exception("Failed to warm up search index")


def run() -> None:
    install_mcp_logging(mcp)
    threading.Thread(target=_warm_index, name="sntx-sem-warmup", daemon=True).start()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run()
