"""Unit tests for HelpIndex search optimizations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sntx_sem.config import SearchConfig
from sntx_sem.index.store import (
    HelpIndex,
    _build_search_text,
    _collect_matched_terms,
    _domain_where_clause,
    _vector_index_partitions,
    split_identifier,
)


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


def test_collect_matched_terms_keeps_query_order_without_duplicates() -> None:
    query = "СтрСоединить строка СОЕДИНИТЬ missing"
    search_text = "описание метода стрсоединить и строка"
    assert _collect_matched_terms(query, search_text) == ["стрсоединить", "строка"]


def test_split_identifier_splits_cyrillic_camel_case() -> None:
    assert split_identifier("РазложитьСтрокуВМассивСлов") == "разложить строку в массив слов"
    assert (
        split_identifier("СтроковыеФункцииКлиентСервер.ПодставитьПараметрыВСтрокуИзМассива")
        == "строковые функции клиент сервер подставить параметры в строку из массива"
    )


def test_build_search_text_includes_split_identifier() -> None:
    text = _build_search_text(
        {
            "title_ru": "СтроковыеФункцииКлиентСервер.РазложитьСтрокуВМассивСлов",
            "title_en": "",
            "domain": "bsp",
            "path": "CommonModules/СтроковыеФункцииКлиентСервер",
            "text": "Разбивает строку на несколько строк.",
        }
    )
    assert "разложить строку в массив слов" in text


def _write_bsp_ranking_chunks(path: Path) -> None:
    chunks = [
        {
            "id": "bsp:from_array",
            "domain": "bsp",
            "title_ru": "СтроковыеФункцииКлиентСервер.ПодставитьПараметрыВСтрокуИзМассива",
            "title_en": "",
            "path": "CommonModules/СтроковыеФункцииКлиентСервер",
            "text": "Подставляет параметры в строку из массива.",
            "entity_kind": "function",
            "html_path": "",
            "syntax": "",
        },
        {
            "id": "bsp:to_array",
            "domain": "bsp",
            "title_ru": "СтроковыеФункцииКлиентСервер.РазложитьСтрокуВМассивСлов",
            "title_en": "",
            "path": "CommonModules/СтроковыеФункцииКлиентСервер",
            "text": "Разбивает строку на несколько строк по разделителям.",
            "entity_kind": "function",
            "html_path": "",
            "syntax": "",
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def _write_multi_domain_chunks(path: Path) -> None:
    chunks = [
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
        {
            "id": "bsp:split",
            "domain": "bsp",
            "title_ru": "СтроковыеФункцииКлиентСервер.РазложитьСтрокуВМассивСлов",
            "title_en": "",
            "path": "CommonModules/СтроковыеФункцииКлиентСервер",
            "text": "разложить строку в массив слов по разделителям",
            "entity_kind": "method",
            "html_path": "",
            "syntax": "",
        },
        {
            "id": "platform:array",
            "domain": "platform_api",
            "title_ru": "ФиксированныйМассив.ВГраница",
            "title_en": "FixedArray.UBound",
            "text": "возвращает наибольший индекс элемента массива",
            "entity_kind": "method",
            "html_path": "",
            "syntax": "",
        },
        {
            "id": "platform:noise1",
            "domain": "platform_api",
            "title_ru": "ДанныеКонтакта.АдресаМгновенныхСообщений",
            "title_en": "",
            "text": "массив адресов мгновенных сообщений",
            "entity_kind": "property",
            "html_path": "",
            "syntax": "",
        },
        {
            "id": "platform:noise2",
            "domain": "platform_api",
            "title_ru": "ДанныеЗапросаПоделиться.Строки",
            "title_en": "",
            "text": "массив строк для запроса поделиться",
            "entity_kind": "property",
            "html_path": "",
            "syntax": "",
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")


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


def test_bsp_search_prefers_to_array_over_from_array(tmp_path: Path) -> None:
    jsonl = tmp_path / "chunks.jsonl"
    _write_bsp_ranking_chunks(jsonl)

    backend = _FixedEmbeddingBackend()
    index = HelpIndex(tmp_path / "index", backend, SearchConfig(final_top_k=2))
    index.build(index.load_chunks_from_jsonl(jsonl))

    results = index.search("строку в массив", domain="bsp", limit=2)
    assert results
    assert results[0].id == "bsp:to_array"


def test_all_domains_search_includes_relevant_bsp_result(tmp_path: Path) -> None:
    jsonl = tmp_path / "chunks.jsonl"
    _write_multi_domain_chunks(jsonl)

    backend = _FixedEmbeddingBackend()
    index = HelpIndex(tmp_path / "index", backend, SearchConfig(final_top_k=3))
    index.build(index.load_chunks_from_jsonl(jsonl))

    results = index.search("строку в массив", domain="all", limit=3)
    domains = {result.domain for result in results}
    assert "bsp" in domains
    bsp_hits = [result for result in results if result.domain == "bsp"]
    assert bsp_hits
    assert bsp_hits[0].id == "bsp:split"
    assert results[0].id == "bsp:split"


def test_build_persists_split_identifier_in_chunks_meta(tmp_path: Path) -> None:
    jsonl = tmp_path / "chunks.jsonl"
    _write_bsp_ranking_chunks(jsonl)

    backend = _FixedEmbeddingBackend()
    index_dir = tmp_path / "index"
    index = HelpIndex(index_dir, backend, SearchConfig(final_top_k=2))
    index.build(index.load_chunks_from_jsonl(jsonl))

    meta = json.loads((index_dir / "chunks_meta.json").read_text(encoding="utf-8"))
    to_array = next(item for item in meta if item["id"] == "bsp:to_array")
    assert "разложить строку в массив слов" in to_array["search_text"]
