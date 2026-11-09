"""Tests for BSL procedure splitting and discovery order."""

from __future__ import annotations

from pathlib import Path

import pytest

from sntx_sem.config.models import LocalConfig
from sntx_sem.examples.procedures import ModuleCache, discover_units, module_rank, split_procedures


def test_split_ignores_procedure_keyword_inside_string() -> None:
    content = """
Процедура Реальная()
    Текст = "Процедура Фальшивая()
    |КонецПроцедуры";
    А = 1;
КонецПроцедуры
"""
    found = split_procedures(content)
    assert [item[0] for item in found] == ["Реальная"]


def test_module_rank_puts_common_modules_before_forms() -> None:
    common = module_rank("CommonModules/Остатки/Ext/Module.bsl")
    form = module_rank("Documents/Заказ/Forms/Форма/Ext/Form/Module.bsl")
    manager = module_rank("Documents/Заказ/Ext/ManagerModule.bsl")
    assert common < manager < form


def test_discover_keeps_same_path_in_two_configs(tmp_path: Path) -> None:
    relative = "CommonModules/Остатки/Ext/Module.bsl"
    for name in ("ut", "erp"):
        module = tmp_path / name / relative
        module.parent.mkdir(parents=True)
        module.write_text(_procedure("ПолучитьОстатки"), encoding="utf-8")
    units, available = discover_units(
        [
            LocalConfig(path=str(tmp_path / "ut"), label="UT"),
            LocalConfig(path=str(tmp_path / "erp"), label="ERP"),
        ],
        min_procedure_lines=1,
    )
    assert available == {"UT", "ERP"}
    assert [unit.key for unit in units] == [
        ("UT", relative, "ПолучитьОстатки"),
        ("ERP", relative, "ПолучитьОстатки"),
    ]
    assert units[0].config_index == 0
    assert units[1].config_index == 1


def test_discover_orders_common_module_before_form(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    _write(root, "Documents/Заказ/Forms/Форма/Ext/Form/Module.bsl", _procedure("НаФорме"))
    _write(root, "CommonModules/Остатки/Ext/Module.bsl", _procedure("ПолучитьОстатки"))
    units, _available = discover_units(
        [LocalConfig(path=str(root), label="UT")],
        min_procedure_lines=1,
    )
    assert [unit.procedure for unit in units] == ["ПолучитьОстатки", "НаФорме"]


def test_short_procedure_is_not_discovered(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    _write(
        root, "CommonModules/Короткий/Ext/Module.bsl", "Процедура А()\n    Б = 1;\nКонецПроцедуры\n"
    )
    units, available = discover_units(
        [LocalConfig(path=str(root), label="UT")],
        min_procedure_lines=8,
    )
    assert available == {"UT"}
    assert units == []


def test_unchanged_module_is_not_reread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "cfg"
    _write(root, "CommonModules/Остатки/Ext/Module.bsl", _procedure("ПолучитьОстатки"))
    cache = ModuleCache(tmp_path / "mine_modules.json")
    configs = [LocalConfig(path=str(root), label="UT")]
    first, _available = discover_units(configs, min_procedure_lines=1, cache=cache)

    def fail_read(self: Path, *args: object, **kwargs: object) -> str:
        raise AssertionError(f"повторное чтение {self}")

    monkeypatch.setattr(Path, "read_text", fail_read)
    second, _available = discover_units(configs, min_procedure_lines=1, cache=cache)
    assert [unit.key for unit in second] == [unit.key for unit in first]
    assert all(unit.text == "" for unit in second)


def test_corrupt_module_cache_is_rebuilt(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    _write(root, "CommonModules/Остатки/Ext/Module.bsl", _procedure("ПолучитьОстатки"))
    cache_path = tmp_path / "mine_modules.json"
    cache_path.write_text("{", encoding="utf-8")
    cache = ModuleCache.load(cache_path)
    units, _available = discover_units(
        [LocalConfig(path=str(root), label="UT")],
        min_procedure_lines=1,
        cache=cache,
    )
    assert [unit.procedure for unit in units] == ["ПолучитьОстатки"]
    reloaded = ModuleCache.load(cache_path)
    assert reloaded.records


def test_duplicate_procedure_names_are_disambiguated(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    text = _procedure("А") + _procedure("А")
    _write(root, "CommonModules/Модуль/Ext/Module.bsl", text)
    units, _available = discover_units(
        [LocalConfig(path=str(root), label="UT")],
        min_procedure_lines=1,
    )
    assert [unit.procedure for unit in units] == ["А", f"А#{units[1].start_line}"]


def _procedure(name: str) -> str:
    return (
        f"Процедура {name}()\n"
        "    Запрос = Новый Запрос;\n"
        '    Запрос.Текст = "ВЫБРАТЬ 1";\n'
        "    Возврат Запрос.Выполнить();\n"
        "КонецПроцедуры\n"
    )


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
