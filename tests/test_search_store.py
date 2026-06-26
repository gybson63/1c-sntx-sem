"""Unit tests for HelpIndex search optimizations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sntx_sem.config import SearchConfig
from sntx_sem.index.store import HelpIndex, _domain_where_clause, _vector_index_partitions


class _FixedEmbeddingBackend:
    model_id = "test-fixed"

    def embed_passages(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        del batch_size
        return np.asarray(
            [[float(i), float(len(text))] for i, text in enumerate(texts)],
            dtype=np.float32,
        )

    def embed_query(self, query: str) -> np.ndarray:
        return np.asarray([float(len(query)), 1.0], dtype=np.float32)


def _write_chunks(path: Path) -> None:
    chunks = [
        {
            "id": "bsp:split",
            "domain": "bsp",
            "title_ru": "РазложитьСтроку",
            "title_en": "",
            "text": "разложить строку по разделителю",
            "entity_kind": "method",
            "html_path": "",
            "syntax": "",
        },
        {
            "id": "bsp:join",
            "domain": "bsp",
            "title_ru": "СоединитьСтроки",
            "title_en": "",
            "text": "соединить массив строк",
            "entity_kind": "method",
            "html_path": "",
            "syntax": "",
        },
        {
            "id": "bsl:str",
            "domain": "bsl_lang",
            "title_ru": "Строка",
            "title_en": "String",
            "text": "тип строка платформы",
            "entity_kind": "type",
            "html_path": "",
            "syntax": "",
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def test_domain_where_clause_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        _domain_where_clause("bsp;drop")


def test_vector_index_partitions_scales_with_table_size() -> None:
    assert _vector_index_partitions(16) == 4
    assert _vector_index_partitions(10000) == 100


def test_build_creates_vector_and_scalar_indices(tmp_path: Path) -> None:
    jsonl = tmp_path / "chunks.jsonl"
    _write_chunks(jsonl)

    backend = _FixedEmbeddingBackend()
    index = HelpIndex(tmp_path / "index", backend, SearchConfig(final_top_k=2))
    count, _dimensions = index.build(index.load_chunks_from_jsonl(jsonl))
    assert count == 3

    indexed_columns: set[str] = set()
    for item in index._table.list_indices():
        cols = item.get("columns", []) if isinstance(item, dict) else getattr(item, "columns", [])
        indexed_columns.update(cols)
    assert "domain" in indexed_columns
    assert "vector" not in indexed_columns


def test_domain_search_returns_only_matching_domain(tmp_path: Path) -> None:
    jsonl = tmp_path / "chunks.jsonl"
    _write_chunks(jsonl)

    backend = _FixedEmbeddingBackend()
    index = HelpIndex(tmp_path / "index", backend, SearchConfig(final_top_k=5))
    index.build(index.load_chunks_from_jsonl(jsonl))

    results = index.search("разложить строку", domain="bsp", limit=5)
    assert results
    assert all(result.domain == "bsp" for result in results)
    assert any("Разложить" in result.title for result in results)


def test_existing_table_gets_indices_on_load(tmp_path: Path) -> None:
    jsonl = tmp_path / "chunks.jsonl"
    _write_chunks(jsonl)

    backend = _FixedEmbeddingBackend()
    index_dir = tmp_path / "index"
    builder = HelpIndex(index_dir, backend, SearchConfig())
    builder.build(builder.load_chunks_from_jsonl(jsonl))

    reloaded = HelpIndex(index_dir, backend, SearchConfig())
    reloaded._ensure_loaded()
    assert reloaded._indices_ready
