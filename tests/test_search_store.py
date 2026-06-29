"""Unit tests for HelpIndex search optimizations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sntx_sem.config import SearchConfig
from sntx_sem.index.store import (
    HelpIndex,
    _build_lexical_text,
    _build_search_text,
    _build_semantic_text,
    _chunk_meta_row,
    _collect_matched_terms,
    _domain_where_clause,
    _tokens_related,
    _vector_index_partitions,
    split_identifier,
)


class _FlatEmbeddingBackend:
    """Equal vectors so BM25 and title bonus drive ranking in tests."""

    model_id = "test-flat"

    def embed_passages(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        del batch_size
        return np.ones((len(texts), 2), dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        del query
        return np.ones(2, dtype=np.float32)


class _DescriptionEmbeddingBackend:
    """Deterministic embedding backend that rewards behavioral descriptions."""

    model_id = "test-description"

    def embed_passages(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        del batch_size
        vectors = [self._embed_text(text) for text in texts]
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        q = query.lower()
        if "строку в массив" in q:
            return np.asarray([1.0, 1.0, 0.0, 0.1], dtype=np.float32)
        if "стрразделить" in q:
            return np.asarray([1.0, 1.0, 0.0, 0.1], dtype=np.float32)
        return self._embed_text(query)

    def _embed_text(self, text: str) -> list[float]:
        normalized = text.lower()
        return [
            float(
                "раздел" in normalized
                or "разбивает строку" in normalized
                or "по разделителям" in normalized
            ),
            float("массив со строк" in normalized or "разделения исходной строки" in normalized),
            float("список слов" in normalized),
            0.1,
        ]


def _write_hbk_only_chunks(path: Path) -> None:
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
            "id": "platform:array",
            "domain": "platform_api",
            "title_ru": "ФиксированныйМассив.ВГраница",
            "title_en": "FixedArray.UBound",
            "text": "возвращает наибольший индекс элемента массива",
            "entity_kind": "method",
            "html_path": "",
            "syntax": "",
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")


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


def test_build_semantic_text_strsplit_excludes_name_tokens() -> None:
    semantic_text = _build_semantic_text(
        {
            "title_ru": "СтрРазделить",
            "title_en": "StrSplit",
            "domain": "platform_api",
            "text": (
                "# Глобальный контекст.СтрРазделить\n\n"
                "Синтаксис:\nСтрРазделить(<Строка>, <Разделитель>, <ВключатьПустые>)\n\n"
                "Параметры:\n<Строка> (обязательный)\nТип: Строка.\nРазделяемая строка.\n\n"
                "Возвращаемое значение:\nТип: Массив.\n"
                "Массив со строками, которые получились в результате разделения "
                "исходной строки.\n\n"
                "Описание:\nРазделяет строку на части по указанным символам-разделителям."
            ),
        }
    )
    assert "массив со строками" in semantic_text.lower()
    assert "разложитьстрокувмассивслов" not in semantic_text.lower()


def test_build_lexical_text_bsp_includes_split_identifier() -> None:
    lexical_text = _build_lexical_text(
        {
            "title_ru": "СтроковыеФункцииКлиентСервер.РазложитьСтрокуВМассивСлов",
            "title_en": "",
            "domain": "bsp",
            "path": "CommonModules/СтроковыеФункцииКлиентСервер",
            "text": "Разбивает строку на несколько строк.",
        }
    )
    assert "разложить строку в массив слов" in lexical_text


def test_build_search_text_combines_semantic_and_lexical_parts() -> None:
    text = _build_search_text(
        {
            "title_ru": "СтроковыеФункцииКлиентСервер.РазложитьСтрокуВМассивСлов",
            "title_en": "",
            "domain": "bsp",
            "path": "CommonModules/СтроковыеФункцииКлиентСервер",
            "text": "Описание:\nРазбивает строку на несколько строк.",
        }
    )
    assert "разложить строку в массив слов" in text
    assert "разбивает строку" in text.lower()


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
    count, _dimensions, vector_index_built = index.build(index.load_chunks_from_jsonl(jsonl))
    assert count == 3
    assert vector_index_built is False

    indexed_columns: set[str] = set()
    for item in index._table.list_indices():
        cols = item.get("columns", []) if isinstance(item, dict) else getattr(item, "columns", [])
        indexed_columns.update(cols)
    assert "domain" in indexed_columns
    assert "vector" not in indexed_columns


def test_build_meta_rows_and_chunks_meta_have_no_vectors(tmp_path: Path) -> None:
    jsonl = tmp_path / "chunks.jsonl"
    _write_chunks(jsonl)

    backend = _FixedEmbeddingBackend()
    index_dir = tmp_path / "index"
    index = HelpIndex(index_dir, backend, SearchConfig(final_top_k=2))
    index.build(index.load_chunks_from_jsonl(jsonl))

    assert all("vector" not in row for row in index._chunks)
    meta = json.loads((index_dir / "chunks_meta.json").read_text(encoding="utf-8"))
    assert len(meta) == 3
    assert all("vector" not in row for row in meta)


def test_build_skips_vector_index_when_disabled(tmp_path: Path) -> None:
    jsonl = tmp_path / "chunks.jsonl"
    _write_chunks(jsonl)

    backend = _FixedEmbeddingBackend()
    index = HelpIndex(
        tmp_path / "index",
        backend,
        SearchConfig(final_top_k=2, build_vector_index=False),
    )
    _, _, vector_index_built = index.build(index.load_chunks_from_jsonl(jsonl))
    assert vector_index_built is False

    indexed_columns: set[str] = set()
    for item in index._table.list_indices():
        cols = item.get("columns", []) if isinstance(item, dict) else getattr(item, "columns", [])
        indexed_columns.update(cols)
    assert "vector" not in indexed_columns


def test_chunk_meta_row_has_no_vector() -> None:
    row = _chunk_meta_row(
        {
            "id": "test:1",
            "domain": "bsp",
            "title_ru": "Тест",
            "text": "текст",
        },
        "тест текст",
    )
    assert "vector" not in row
    assert row["search_text"] == "тест текст"


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

    backend = _DescriptionEmbeddingBackend()
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


def test_tokens_related_matches_russian_morphology() -> None:
    assert _tokens_related("строку", "стр")
    assert _tokens_related("строку", "строка")
    assert not _tokens_related("массив", "строка")


def test_search_empty_bsp_domain_returns_empty(tmp_path: Path) -> None:
    jsonl = tmp_path / "chunks.jsonl"
    _write_hbk_only_chunks(jsonl)

    backend = _FixedEmbeddingBackend()
    index = HelpIndex(tmp_path / "index", backend, SearchConfig(final_top_k=5))
    index.build(index.load_chunks_from_jsonl(jsonl))

    results = index.search("строку в массив", domain="bsp", limit=5)
    assert results == []


def test_fuse_dense_bm25_skips_bm25_for_empty_domain(tmp_path: Path) -> None:
    jsonl = tmp_path / "chunks.jsonl"
    _write_hbk_only_chunks(jsonl)

    backend = _FixedEmbeddingBackend()
    index = HelpIndex(tmp_path / "index", backend, SearchConfig(final_top_k=5))
    index.build(index.load_chunks_from_jsonl(jsonl))
    index._ensure_loaded()

    bm25, indices = index._get_domain_bm25("bsp")
    assert bm25 is None
    assert indices == []

    scores = index._fuse_dense_bm25("тест", [1.0, 1.0], "bsp", 5)
    assert isinstance(scores, dict)


def _write_string_to_array_golden_fixture(path: Path) -> None:
    chunks = [
        {
            "id": "platform:strsplit",
            "domain": "platform_api",
            "title_ru": "СтрРазделить",
            "title_en": "StrSplit",
            "text": (
                "# Глобальный контекст.СтрРазделить\n\n"
                "Синтаксис:\nСтрРазделить(<Строка>, <Разделитель>, <ВключатьПустые>)\n\n"
                "Параметры:\n<Строка> (обязательный)\nТип: Строка.\nРазделяемая строка.\n\n"
                "Возвращаемое значение:\nТип: Массив.\n"
                "Массив со строками, которые получились в результате разделения "
                "исходной строки.\n\n"
                "Описание:\nРазделяет строку на части по указанным символам-разделителям."
            ),
            "entity_kind": "method",
            "html_path": "",
            "syntax": "",
        },
        {
            "id": "bsp:to_array",
            "domain": "bsp",
            "title_ru": "СтроковыеФункцииКлиентСервер.РазложитьСтрокуВМассивСлов",
            "title_en": "",
            "path": "CommonModules/СтроковыеФункцииКлиентСервер",
            "text": (
                "# СтроковыеФункцииКлиентСервер.РазложитьСтрокуВМассивСлов\n\n"
                "Разбивает строку на несколько строк, используя заданный набор разделителей.\n\n"
                "## Параметры\n"
                "Значение - Строка - исходная строка.\n"
                "РазделителиСлов - Строка - перечень символов-разделителей.\n\n"
                "## Возвращаемое значение\n"
                "Массив - список слов."
            ),
            "entity_kind": "function",
            "html_path": "",
            "syntax": "",
        },
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
    ]
    for index in range(220):
        chunks.append(
            {
                "id": f"platform:noise{index}",
                "domain": "platform_api",
                "title_ru": f"ДанныеЗапросаПоделиться{index}.Строки",
                "title_en": "",
                "text": f"массив строк для служебного запроса {index}",
                "entity_kind": "property",
                "html_path": "",
                "syntax": "",
            }
        )
    chunks.append(
        {
            "id": "platform:array",
            "domain": "platform_api",
            "title_ru": "СправочникКонтрагентов",
            "title_en": "CounterpartyCatalog",
            "text": "элемент справочника контрагентов",
            "entity_kind": "topic",
            "html_path": "",
            "syntax": "",
        },
    )
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def test_search_string_to_array_golden_example(tmp_path: Path) -> None:
    jsonl = tmp_path / "chunks.jsonl"
    _write_string_to_array_golden_fixture(jsonl)

    backend = _DescriptionEmbeddingBackend()
    cfg = SearchConfig(final_top_k=5, dense_top_k=25, bm25_top_k=25)
    index = HelpIndex(tmp_path / "index", backend, cfg)
    index.build(index.load_chunks_from_jsonl(jsonl))

    results = index.search("строку в массив", domain="all", limit=5)
    ids = [result.id for result in results]
    ranks = {result.id: idx for idx, result in enumerate(results)}

    assert "platform:strsplit" in ids
    assert "bsp:to_array" in ids
    assert ranks["platform:strsplit"] < ranks["bsp:to_array"]
    assert "bsp:from_array" not in ids[:2]


def test_search_by_name_strsplit(tmp_path: Path) -> None:
    jsonl = tmp_path / "chunks.jsonl"
    _write_string_to_array_golden_fixture(jsonl)

    backend = _DescriptionEmbeddingBackend()
    index = HelpIndex(tmp_path / "index", backend, SearchConfig(final_top_k=3))
    index.build(index.load_chunks_from_jsonl(jsonl))

    results = index.search("СтрРазделить", domain="all", limit=3)
    assert results
    assert results[0].id == "platform:strsplit"
