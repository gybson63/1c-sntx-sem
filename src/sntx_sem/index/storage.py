"""LanceDB storage helpers and domain filter utilities."""

from __future__ import annotations

import re
from typing import Any

_DOMAIN_FILTER_RE = re.compile(r"^[\w]+$")


def resolve_domain_filter(domain: str) -> str:
    if domain == "bsl":
        return "bsl_lang"
    if domain == "query":
        return "query_lang"
    return domain


def domain_where_clause(domain_filter: str) -> str:
    if not _DOMAIN_FILTER_RE.fullmatch(domain_filter):
        raise ValueError(f"Invalid domain filter: {domain_filter!r}")
    return f"domain = '{domain_filter}'"


def indexed_columns(table: Any) -> set[str]:
    columns: set[str] = set()
    for item in table.list_indices():
        if isinstance(item, dict):
            cols = item.get("columns", [])
        else:
            cols = getattr(item, "columns", []) or []
        columns.update(cols)
    return columns


def vector_index_partitions(row_count: int) -> int:
    if row_count <= 32:
        return max(1, row_count // 4 or 1)
    return max(8, min(256, int(row_count**0.5)))
