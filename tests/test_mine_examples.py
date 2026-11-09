"""Journal, kbshff miner and question ranking without a live agent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from sntx_sem.cli import main
from sntx_sem.config.loader import load_config
from sntx_sem.config.models import AppConfig, LocalConfig
from sntx_sem.examples.journal import DONE, FAILED, PENDING, REMOVED, JournalEntry, MineJournal
from sntx_sem.examples.miner import (
    MineError,
    MineRequest,
    code_is_citation,
    collapse_patterns,
    ensure_ready,
    run_mine,
    validate_example_payload,
)
from sntx_sem.examples.procedures import ProcedureUnit, discover_units
from sntx_sem.examples.scanner import CodeExample
from sntx_sem.examples.store import ExamplesStore

_CITATION = 'Запрос = Новый Запрос;\n    Запрос.Текст = "ВЫБРАТЬ 1";'
_QUESTIONS = [
    "остатки номенклатуры на дату",
    "получить остатки товаров запросом",
    "остатки на складе",
]


def test_missing_binary_does_not_touch_examples(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    examples = cfg.data_dir / "examples.jsonl"
    examples.write_text("old\n", encoding="utf-8")
    cfg.examples_miner.binary = "kbshff-not-installed-xyz"
    with pytest.raises(MineError):
        run_mine(cfg, limit=1, include="", labels=(), retry_failed=False)
    assert examples.read_text(encoding="utf-8") == "old\n"
    assert not (cfg.data_dir / ".sntx_sem_cache" / "mine_journal.jsonl").exists()


def test_missing_directory_does_not_mark_removed(tmp_path: Path) -> None:
    journal = MineJournal(tmp_path / "mine_journal.jsonl")
    entry = _entry(status=DONE, label="GONE")
    entry.examples = [{"id": "keep"}]
    journal.entries[entry.key] = entry
    journal.sync([], set())
    assert journal.entries[entry.key].status == DONE
    assert journal.entries[entry.key].examples == [{"id": "keep"}]


def test_deleted_procedure_in_available_config_is_removed(tmp_path: Path) -> None:
    root = tmp_path / "ut"
    relative = "CommonModules/Остатки/Ext/Module.bsl"
    _write(root, relative, _procedure("ПолучитьОстатки"))
    cfg = _config(tmp_path, [("UT", root)])
    units, available = discover_units(cfg.local_configs, min_procedure_lines=1)
    journal = MineJournal(cfg.data_dir / ".sntx_sem_cache" / "mine_journal.jsonl")
    journal.sync(units, available)
    assert journal.entries[("UT", relative, "ПолучитьОстатки")].status == PENDING
    (root / relative).unlink()
    units, available = discover_units(cfg.local_configs, min_procedure_lines=1)
    journal.sync(units, available)
    assert journal.entries[("UT", relative, "ПолучитьОстатки")].status == REMOVED


def test_hash_change_returns_unit_to_pending(tmp_path: Path) -> None:
    root = tmp_path / "ut"
    relative = "CommonModules/Остатки/Ext/Module.bsl"
    _write(root, relative, _procedure("ПолучитьОстатки"))
    cfg = _config(tmp_path, [("UT", root)])
    calls: list[str] = []
    run_mine(
        cfg,
        limit=None,
        include="",
        labels=(),
        retry_failed=False,
        runner=_accept(calls),
        check_ready=False,
    )
    journal = MineJournal.load(cfg.data_dir / ".sntx_sem_cache" / "mine_journal.jsonl")
    assert journal.entries[("UT", relative, "ПолучитьОстатки")].status == DONE
    updated = _procedure("ПолучитьОстатки").replace(
        "    Возврат Запрос.Выполнить();",
        "    А = 1;\n    Возврат Запрос.Выполнить();",
    )
    _write(root, relative, updated)
    run_mine(
        cfg,
        limit=1,
        include="",
        labels=(),
        retry_failed=False,
        runner=_accept(calls),
        check_ready=False,
    )
    assert calls == ["ПолучитьОстатки", "ПолучитьОстатки"]


def test_second_batch_skips_done_and_failed_does_not_consume_limit(tmp_path: Path) -> None:
    root = tmp_path / "ut"
    relative = "CommonModules/Остатки/Ext/Module.bsl"
    _write(root, relative, _procedure("Первая") + _procedure("Вторая"))
    cfg = _config(tmp_path, [("UT", root)])
    calls: list[str] = []

    def runner(request: MineRequest) -> int:
        calls.append(request.unit.procedure)
        if request.unit.procedure == "Первая":
            request.stderr = "обрыв"
            return 1
        return _accept()(request)

    run_mine(
        cfg, limit=1, include="", labels=(), retry_failed=False, runner=runner, check_ready=False
    )
    run_mine(
        cfg, limit=1, include="", labels=(), retry_failed=False, runner=runner, check_ready=False
    )
    assert calls == ["Первая", "Вторая"]
    journal = MineJournal.load(cfg.data_dir / ".sntx_sem_cache" / "mine_journal.jsonl")
    assert journal.entries[("UT", relative, "Первая")].status == FAILED
    assert journal.entries[("UT", relative, "Вторая")].status == DONE


def test_retry_failed_leaves_pending_for_the_next_batch(tmp_path: Path) -> None:
    root = tmp_path / "ut"
    relative = "CommonModules/Остатки/Ext/Module.bsl"
    _write(root, relative, _procedure("Первая") + _procedure("Вторая"))
    cfg = _config(tmp_path, [("UT", root)])
    calls: list[str] = []

    def runner(request: MineRequest) -> int:
        calls.append(request.unit.procedure)
        return 1

    run_mine(
        cfg, limit=1, include="", labels=(), retry_failed=False, runner=runner, check_ready=False
    )
    calls.clear()
    run_mine(
        cfg, limit=1, include="", labels=(), retry_failed=True, runner=runner, check_ready=False
    )
    assert calls == ["Первая"]
    journal = MineJournal.load(cfg.data_dir / ".sntx_sem_cache" / "mine_journal.jsonl")
    assert journal.entries[("UT", relative, "Вторая")].status == PENDING


def test_config_filter_does_not_move_other_labels(tmp_path: Path) -> None:
    ut = tmp_path / "ut"
    erp = tmp_path / "erp"
    relative = "CommonModules/Остатки/Ext/Module.bsl"
    _write(ut, relative, _procedure("ПолучитьОстатки"))
    _write(erp, relative, _procedure("ПолучитьОстатки"))
    cfg = _config(tmp_path, [("UT", ut), ("ERP", erp)])
    calls: list[str] = []
    run_mine(
        cfg,
        limit=None,
        include="",
        labels=("UT",),
        retry_failed=False,
        runner=_accept(calls, attribute="config_label"),
        check_ready=False,
    )
    assert calls == ["UT"]
    journal = MineJournal.load(cfg.data_dir / ".sntx_sem_cache" / "mine_journal.jsonl")
    assert journal.entries[("UT", relative, "ПолучитьОстатки")].status == DONE
    assert journal.entries[("ERP", relative, "ПолучитьОстатки")].status == PENDING


def test_include_limits_path_without_removing_other_modules(tmp_path: Path) -> None:
    root = tmp_path / "ut"
    _write(root, "CommonModules/Остатки/Ext/Module.bsl", _procedure("ПолучитьОстатки"))
    _write(root, "Documents/Заказ/Forms/Форма/Ext/Form/Module.bsl", _procedure("НаФорме"))
    cfg = _config(tmp_path, [("UT", root)])
    run_mine(
        cfg,
        limit=None,
        include="CommonModules",
        labels=(),
        retry_failed=False,
        runner=_accept(),
        check_ready=False,
    )
    journal = MineJournal.load(cfg.data_dir / ".sntx_sem_cache" / "mine_journal.jsonl")
    done = journal.entries[("UT", "CommonModules/Остатки/Ext/Module.bsl", "ПолучитьОстатки")]
    assert done.status == DONE
    form = journal.entries[("UT", "Documents/Заказ/Forms/Форма/Ext/Form/Module.bsl", "НаФорме")]
    assert form.status == PENDING


def test_pattern_collapses_inside_one_config_only(tmp_path: Path) -> None:
    root = tmp_path / "ut"
    relative = "CommonModules/Остатки/Ext/Module.bsl"
    _write(root, relative, _procedure("Первая") + _procedure("Вторая"))
    other = tmp_path / "erp"
    _write(other, relative, _procedure("Первая"))
    cfg = _config(tmp_path, [("UT", root), ("ERP", other)])
    report = run_mine(
        cfg,
        limit=None,
        include="",
        labels=(),
        retry_failed=False,
        runner=_accept(),
        check_ready=False,
    )
    assert report.published == 2
    store = ExamplesStore(cfg.data_dir / "examples.jsonl")
    by_label = {item.config_label: item for item in store.search("остатки", limit=5)}
    assert by_label["UT"].occurrences == 2
    assert by_label["ERP"].occurrences == 1


def test_boilerplate_does_not_replace_existing_examples(tmp_path: Path) -> None:
    root = tmp_path / "ut"
    _write(root, "CommonModules/Остатки/Ext/Module.bsl", _procedure("ПолучитьОстатки"))
    cfg = _config(tmp_path, [("UT", root)])
    examples = cfg.data_dir / "examples.jsonl"
    examples.write_text("old\n", encoding="utf-8")

    def empty(request: MineRequest) -> int:
        request.output_file.parent.mkdir(parents=True, exist_ok=True)
        request.output_file.write_text('{"examples": []}', encoding="utf-8")
        return 0

    run_mine(
        cfg, limit=1, include="", labels=(), retry_failed=False, runner=empty, check_ready=False
    )
    assert examples.read_text(encoding="utf-8") == "old\n"


def test_invalid_json_is_failed(tmp_path: Path) -> None:
    root = tmp_path / "ut"
    relative = "CommonModules/Остатки/Ext/Module.bsl"
    _write(root, relative, _procedure("ПолучитьОстатки"))
    cfg = _config(tmp_path, [("UT", root)])

    def broken(request: MineRequest) -> int:
        request.output_file.parent.mkdir(parents=True, exist_ok=True)
        request.output_file.write_text("{", encoding="utf-8")
        return 0

    run_mine(
        cfg, limit=1, include="", labels=(), retry_failed=False, runner=broken, check_ready=False
    )
    journal = MineJournal.load(cfg.data_dir / ".sntx_sem_cache" / "mine_journal.jsonl")
    assert journal.entries[("UT", relative, "ПолучитьОстатки")].status == FAILED


def test_search_prefers_questions_over_code(tmp_path: Path) -> None:
    path = tmp_path / "examples.jsonl"
    store = ExamplesStore(path)
    store.save(
        [
            CodeExample(
                id="code",
                config_label="UT",
                module_path="a.bsl",
                code="остатки номенклатуры на складе",
                questions=["печать штрихкода этикетки товара"],
            ),
            CodeExample(
                id="question",
                config_label="UT",
                module_path="b.bsl",
                code="Запрос.Выполнить()",
                questions=_QUESTIONS,
                task="Остатки на дату",
            ),
        ]
    )
    found = store.search("остатки номенклатуры", limit=2)
    assert [item.id for item in found] == ["question", "code"]


def test_old_example_without_questions_still_loads(tmp_path: Path) -> None:
    path = tmp_path / "examples.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "legacy",
                "config_label": "UT",
                "module_path": "a.bsl",
                "code": "Запрос.Выполнить остатки",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    store = ExamplesStore(path)
    assert store.search("остатки")[0].id == "legacy"
    assert store._examples[0].questions is None


def test_citation_must_come_from_procedure() -> None:
    unit = ProcedureUnit(
        config_label="UT",
        config_index=0,
        module_path="a.bsl",
        module_rank=0,
        procedure="ПолучитьОстатки",
        start_line=1,
        end_line=4,
        content_hash="abc",
        text=_procedure("ПолучитьОстатки"),
    )
    with pytest.raises(ValueError, match="не цитата"):
        validate_example_payload(
            {"examples": [{**_payload(), "code": "ВыдуманныйВызов()"}]},
            unit,
        )
    validated = validate_example_payload({"examples": [_payload()]}, unit)
    assert validated[0]["questions"] == _QUESTIONS
    assert code_is_citation(_CITATION, unit.text)


def test_collapse_keeps_configs_separate() -> None:
    first = CodeExample(
        id="1", config_label="UT", module_path="a.bsl", code="А", pattern_key="остатки"
    )
    second = CodeExample(
        id="2", config_label="UT", module_path="b.bsl", code="Б", pattern_key="остатки"
    )
    third = CodeExample(
        id="3", config_label="ERP", module_path="a.bsl", code="В", pattern_key="остатки"
    )
    collapsed = collapse_patterns([first, second, third])
    assert len(collapsed) == 2
    assert next(item for item in collapsed if item.config_label == "UT").occurrences == 2
    assert next(item for item in collapsed if item.config_label == "ERP").occurrences == 1


def test_load_examples_miner_resolves_project_root(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "examples_miner:\n"
        "  binary: kbshff\n"
        '  project_root: "."\n'
        "  agent: typical-examples\n"
        '  max_cost: "USD:0.10"\n'
        "  min_procedure_lines: 4\n",
        encoding="utf-8",
    )
    cfg = load_config(config_path)
    assert cfg.examples_miner.max_cost == "USD:0.10"
    assert cfg.examples_miner.min_procedure_lines == 4
    assert cfg.examples_miner.project_root == str(tmp_path.resolve())


def test_cli_status_and_rejected_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("local_configs: []\n", encoding="utf-8")
    monkeypatch.setenv("SNTX_SEM_CONFIG", str(config_path))
    runner = CliRunner()
    status = runner.invoke(main, ["mine-examples", "status"])
    assert status.exit_code == 0
    assert "журнал пуст" in status.output
    rejected = runner.invoke(main, ["mine-examples", "--limit", "0"])
    assert rejected.exit_code != 0


def test_scan_examples_refuses_when_miner_owns_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    cache = data / ".sntx_sem_cache"
    cache.mkdir(parents=True)
    (cache / "mine_owned").write_text("1", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"data_dir: {data.as_posix()}\nlocal_configs: []\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SNTX_SEM_CONFIG", str(config_path))
    result = CliRunner().invoke(main, ["scan-examples"])
    assert result.exit_code != 0
    assert "mine-examples" in result.output


def test_ensure_ready_requires_profile(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    binary = tmp_path / "kbshff.cmd"
    binary.write_text("@echo off\n", encoding="utf-8")
    cfg.examples_miner.binary = str(binary)
    with pytest.raises(MineError, match="профиль"):
        ensure_ready(cfg)


def _config(tmp_path: Path, configs: list[tuple[str, Path]] | None = None) -> AppConfig:
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    (data / "export").mkdir(exist_ok=True)
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    cfg = AppConfig(
        data_dir=data,
        export_dir=data / "export",
        local_configs=[LocalConfig(path=str(path), label=label) for label, path in configs or []],
    )
    cfg.examples_miner.project_root = str(project)
    cfg.examples_miner.min_procedure_lines = 1
    cfg.examples_miner.binary = "kbshff"
    return cfg


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


def _payload() -> dict[str, object]:
    return {
        "questions": _QUESTIONS,
        "task": "Остатки на дату",
        "code": _CITATION,
        "query_text": "ВЫБРАТЬ 1",
        "calls": ["Запрос"],
        "pattern_key": "остатки-на-дату",
    }


def _accept(record: list[str] | None = None, *, attribute: str = "procedure"):
    def run(request: MineRequest) -> int:
        if record is not None:
            record.append(getattr(request.unit, attribute))
        request.output_file.parent.mkdir(parents=True, exist_ok=True)
        request.output_file.write_text(
            json.dumps({"examples": [_payload()]}, ensure_ascii=False),
            encoding="utf-8",
        )
        return 0

    return run


def _entry(*, status: str, label: str) -> JournalEntry:
    return JournalEntry(
        config_label=label,
        module_path="CommonModules/Остатки/Ext/Module.bsl",
        procedure="ПолучитьОстатки",
        start_line=1,
        end_line=5,
        content_hash="abc",
        status=status,
        updated_at="2026-11-10T00:00:00+00:00",
    )
