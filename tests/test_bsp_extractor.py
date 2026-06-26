"""Tests for BSP API extraction."""

from __future__ import annotations

from pathlib import Path

from sntx_sem.bsp.comment_parser import parse_bsl_exports
from sntx_sem.bsp.extractor import extract_bsp_dir, ingest_bsp

FIXTURES = Path(__file__).parent / "fixtures" / "bsp"

VARIANTY_SAMPLE = """
#Область ПрограммныйИнтерфейс

// Возвращает ссылку на вариант отчета.
//
// Параметры:
//  Отчет - Строка - ссылка на отчет.
//  КлючВарианта - Строка - имя варианта отчета.
//
// Возвращаемое значение:
//  СправочникСсылка.ВариантыОтчетов, Неопределено - вариант отчета.
//
Функция ВариантОтчета(Отчет, КлючВарианта) Экспорт
	Возврат Неопределено;
КонецФункции

#КонецОбласти

#Область СлужебныеПроцедурыИФункции

Функция СлужебнаяФункция() Экспорт
	Возврат 0;
КонецФункции

#КонецОбласти
"""

SHABLONY_SAMPLE = """
#Область ПрограммныйИнтерфейс

// Создает сообщение на основании предмета по шаблону сообщения.
//
// Параметры:
//  Шаблон - СправочникСсылка.ШаблоныСообщений - ссылка на шаблон.
//  Предмет - Произвольный - объект основание.
//  ДополнительныеПараметры - Структура:
//      * ЗначенияПараметровСКД - Структура - значения параметров СКД.
//
// Возвращаемое значение:
//  Структура - подготовленное сообщение:
//    * Тема - Строка - тема письма
//
// Пример:
//  Сообщение = СформироватьСообщение(Шаблон, Документ, Новый УникальныйИдентификатор);
//
Функция СформироватьСообщение(Шаблон, Предмет, Ид, ДопПараметры = Неопределено) Экспорт
	Возврат Новый Структура;
КонецФункции

#КонецОбласти
"""


def test_parse_varianty_otchetov() -> None:
    docs = parse_bsl_exports(VARIANTY_SAMPLE)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.name == "ВариантОтчета"
    assert doc.kind == "method"
    assert "вариант отчета" in doc.description.lower()
    assert "КлючВарианта" in doc.parameters
    assert "ВариантыОтчетов" in doc.return_value


def test_parse_excludes_outside_program_interface() -> None:
    docs = parse_bsl_exports(VARIANTY_SAMPLE)
    names = {d.name for d in docs}
    assert "СлужебнаяФункция" not in names


def test_parse_shablony_with_example() -> None:
    docs = parse_bsl_exports(SHABLONY_SAMPLE)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.name == "СформироватьСообщение"
    assert "ЗначенияПараметровСКД" in doc.parameters
    assert "СформироватьСообщение" in doc.example
    assert "Тема" in doc.return_value


def test_extract_bsp_fixture_dir() -> None:
    chunks, stats = extract_bsp_dir(FIXTURES)
    assert stats["modules"] == 2
    assert stats["methods"] == 2
    assert stats["bsp_version"] == "3.1.12.238"

    ids = {c.id for c in chunks}
    assert "bsp:ВариантыОтчетов.ВариантОтчета" in ids
    assert "bsp:ШаблоныСообщений.СформироватьСообщение" in ids

    variant = next(c for c in chunks if "ВариантОтчета" in c.id)
    assert variant.domain == "bsp"
    assert variant.parameters
    assert "## Параметры" in variant.text
    assert variant.signature == "ВариантыОтчетов.ВариантОтчета"


def test_ingest_bsp_writes_jsonl(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    stats = ingest_bsp(FIXTURES, export_dir)

    assert stats["methods"] == 2
    assert (export_dir / "bsp_api.jsonl").is_file()
    assert (export_dir / "all_chunks.jsonl").is_file()

    lines = (export_dir / "all_chunks.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
