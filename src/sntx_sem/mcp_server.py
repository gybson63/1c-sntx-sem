"""MCP server for 1C semantic help search."""

from __future__ import annotations

import json
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from sntx_sem.config import load_config
from sntx_sem.examples.store import ExamplesStore
from sntx_sem.index.store import EmbeddingModel, HelpIndex

mcp = FastMCP("1c-syntax-sem")

_index: HelpIndex | None = None
_examples: ExamplesStore | None = None
_config = load_config()


def _get_index() -> HelpIndex:
    global _index
    if _index is None:
        embedder = EmbeddingModel(_config.embedding.model, _config.embedding.device)
        _index = HelpIndex(_config.index_dir, embedder, _config.search)
    return _index


def _get_examples() -> ExamplesStore:
    global _examples
    if _examples is None:
        path = _config.data_dir / "examples.jsonl"
        _examples = ExamplesStore(path)
    return _examples


def _format_result(r: Any) -> dict:
    return {
        "id": r.id,
        "domain": r.domain,
        "title": r.title,
        "score": round(r.score, 4),
        "entity_kind": r.entity_kind,
        "html_path": r.html_path,
        "excerpt": r.text[:500] + ("..." if len(r.text) > 500 else ""),
    }


@mcp.tool()
def search_help(query: str, domain: str = "all", limit: int = 5) -> str:
    """Semantic search over 1C platform help (BSL + query language).

    Args:
        query: Natural language search query (Russian or English).
        domain: Filter — 'all', 'bsl', 'query', 'bsl_lang', 'query_lang', 'platform_api'.
        limit: Maximum number of results.
    """
    index = _get_index()
    results = index.search(query, domain=domain, limit=limit)
    return json.dumps([_format_result(r) for r in results], ensure_ascii=False, indent=2)


@mcp.tool()
def search_bsl_syntax(query: str, limit: int = 5) -> str:
    """Search built-in 1C language (BSL) syntax help."""
    return search_help(query, domain="bsl", limit=limit)


@mcp.tool()
def search_query_language(query: str, limit: int = 5) -> str:
    """Search 1C query language (SDBL) help."""
    return search_help(query, domain="query", limit=limit)


@mcp.tool()
def get_topic(topic_id: str, include_examples: bool = True) -> str:
    """Get full help topic by ID, optionally with linked code examples."""
    index = _get_index()
    topic = index.get_topic(topic_id)
    if not topic:
        return json.dumps({"error": f"Topic not found: {topic_id}"}, ensure_ascii=False)

    result = dict(topic)
    if include_examples:
        examples = _get_examples().find_by_topic(topic_id)
        result["examples"] = [ex.to_dict() for ex in examples]
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def find_examples(query: str = "", topic_id: str = "", limit: int = 5) -> str:
    """Find code examples from local configurations.

    Args:
        query: Natural language query to search examples.
        topic_id: Help topic ID to find linked examples.
        limit: Maximum results.
    """
    store = _get_examples()
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
    index = _get_index()
    stats = index.stats()
    examples_count = _get_examples().count
    stats["examples"] = examples_count
    return json.dumps(stats, ensure_ascii=False, indent=2)


def run() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run()
