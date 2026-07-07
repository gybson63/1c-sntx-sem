"""Unit tests for search response formatting."""

from __future__ import annotations

from types import SimpleNamespace

from sntx_sem.search_service import format_search_result


def _make_result(text: str, highlight_terms: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        id="topic:1",
        domain="platform_api",
        title="Глобальный контекст.СтрСоединить",
        text=text,
        score=0.123456,
        entity_kind="method",
        html_path="docs/topic1.html",
        highlight_terms=highlight_terms,
    )


def test_format_search_result_uses_index_highlight_terms() -> None:
    text = "Префикс. " + ("x" * 620) + " СтрСоединить используется для объединения строк."
    payload = format_search_result(_make_result(text, ["стрсоединить", "строк"]), query="ignored")

    assert payload["highlight_terms"] == ["стрсоединить", "строк"]
    assert payload["excerpt_start"] > 0
    assert payload["excerpt_end"] <= len(text)
    assert "СтрСоединить" in payload["excerpt"]


def test_format_search_result_falls_back_to_query_tokens() -> None:
    payload = format_search_result(
        _make_result("Короткий текст без совпадений.", []),
        query="Строка массив",
    )

    assert payload["highlight_terms"] == ["строка", "массив"]
    assert payload["excerpt"] == "Короткий текст без совпадений."
    assert payload["excerpt_start"] == 0
    assert payload["excerpt_end"] == len("Короткий текст без совпадений.")


def test_format_search_result_includes_match_explanation() -> None:
    result = _make_result("Короткий текст.", ["текст"])
    result.match_sources = ["semantic", "bm25"]
    result.match_explanation = "Семантический поиск: rank 1; BM25: rank 2."
    result.semantic_excerpt = "Описание: короткий семантический текст."
    result.semantic_highlight_terms = ["семантический"]
    result.score_breakdown = {
        "total": 0.123456,
        "dense_rank": 1,
        "dense_similarity": 0.987654,
        "bm25_rank": 2,
        "bm25_raw": None,
    }

    payload = format_search_result(result, query="текст")

    assert payload["match_sources"] == ["semantic", "bm25"]
    assert payload["match_explanation"].startswith("Семантический поиск")
    assert payload["semantic_excerpt"] == "Описание: короткий семантический текст."
    assert payload["semantic_highlight_terms"] == ["семантический"]
    assert payload["score_breakdown"]["total"] == 0.1235
    assert payload["score_breakdown"]["dense_rank"] == 1
    assert payload["score_breakdown"]["dense_similarity"] == 0.9877
