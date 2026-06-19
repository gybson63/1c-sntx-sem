"""Tests for config scanner."""

from __future__ import annotations

from pathlib import Path

from sntx_sem.examples.scanner import extract_queries_from_bsl, scan_config_path


def test_extract_query_from_bsl() -> None:
    code = '''
    Запрос = Новый Запрос;
    Запрос.Текст = "ВЫБРАТЬ
    |    Т.Наименование
    |ИЗ
    |    Справочник.Товары КАК Т
    |ГДЕ
    |    Т.ПометкаУдаления = ЛОЖЬ";
    '''
    queries = extract_queries_from_bsl(code)
    assert len(queries) >= 1
    assert "ВЫБРАТЬ" in queries[0]


def test_scan_temp_config(tmp_path: Path) -> None:
    mod = tmp_path / "CommonModules" / "TestModule" / "Module.bsl"
    mod.parent.mkdir(parents=True)
    mod.write_text(
        'Запрос = Новый Запрос;\n'
        'Запрос.Текст = "ВЫБРАТЬ\n'
        '    |    1 КАК N";\n'
        'Результат = Запрос.Выполнить();',
        encoding="utf-8",
    )
    examples = scan_config_path(tmp_path, "TEST")
    assert len(examples) >= 1
    assert examples[0].config_label == "TEST"
