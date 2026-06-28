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
