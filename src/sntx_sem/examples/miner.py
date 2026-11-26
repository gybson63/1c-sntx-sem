"""Mine live-usage examples from local configs by calling kbshff one procedure at a time."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from sntx_sem.config.models import AppConfig
from sntx_sem.examples.journal import (
    DONE,
    FAILED,
    MineJournal,
    format_status,
)
from sntx_sem.examples.procedures import (
    ModuleCache,
    ProcedureUnit,
    discover_units,
    read_procedure_text,
)
from sntx_sem.examples.scanner import CodeExample
from sntx_sem.examples.store import ExamplesStore

_MIN_QUESTIONS = 3
_MAX_QUESTIONS = 5
_RUN_TIMEOUT_SEC = 240


class MineError(RuntimeError):
    """The miner cannot start. Existing examples.jsonl is left untouched."""


@dataclass
class MineRequest:
    prompt: str
    procedure_file: Path
    home_rel: str
    output_file: Path
    unit: ProcedureUnit
    stderr: str = ""


@dataclass
class MineReport:
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    published: int | None = None
    messages: list[str] = field(default_factory=list)


Runner = Callable[[MineRequest], int]


def run_mine(
    cfg: AppConfig,
    *,
    limit: int | None,
    include: str,
    labels: list[str] | tuple[str, ...],
    retry_failed: bool,
    runner: Runner | None = None,
    check_ready: bool = True,
) -> MineReport:
    if check_ready:
        ensure_ready(cfg)
    report = MineReport()
    journal, available, units = _sync(cfg)
    published = publish_examples(cfg, journal)
    if published is not None:
        report.published = published
    by_key = {unit.key: unit for unit in units}
    batch = journal.select(
        limit=limit,
        labels=set(labels),
        include=include,
        retry_failed=retry_failed,
        available=available,
    )
    active_runner = runner or (lambda request: kbshff_runner(cfg, request))
    for entry in batch:
        unit = by_key.get(entry.key)
        if unit is None:
            entry.status = FAILED
            entry.error = "процедура не найдена после сверки"
            entry.examples = []
            journal.save()
            report.processed += 1
            report.failed += 1
            report.messages.append(f"{_unit_label(entry)} -> failed: {entry.error}")
            continue
        try:
            examples, skip_reason = mine_procedure(cfg, unit, active_runner)
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            entry.status = FAILED
            entry.error = str(exc)[:500]
            entry.examples = []
            entry.skip_reason = ""
            journal.save()
            report.processed += 1
            report.failed += 1
            report.messages.append(f"{_unit_label(entry)} -> failed: {entry.error}")
            continue
        entry.status = DONE
        entry.examples = examples
        entry.skip_reason = skip_reason
        entry.error = ""
        journal.save()
        report.processed += 1
        report.succeeded += 1
        if skip_reason:
            report.messages.append(f"{_unit_label(entry)} -> done, {skip_reason}")
        else:
            report.messages.append(f"{_unit_label(entry)} -> done, примеров: {len(examples)}")
        published = publish_examples(cfg, journal)
        if published is not None:
            report.published = published
    report.messages.append(format_status(journal, available))
    return report


def mine_status(cfg: AppConfig) -> str:
    journal, available, _units = _sync(cfg)
    publish_examples(cfg, journal)
    return format_status(journal, available)


def ensure_ready(cfg: AppConfig) -> None:
    miner = cfg.examples_miner
    if not miner.project_root:
        raise MineError("не задан examples_miner.project_root")
    root = Path(miner.project_root)
    if not root.is_dir():
        raise MineError(f"examples_miner.project_root не является каталогом: {root}")
    binary = miner.binary
    if not binary or not (Path(binary).is_file() or shutil.which(binary)):
        raise MineError(f"не найден исполняемый файл kbshff: {binary or '(пусто)'}")
    profile = root / ".kuibysheff" / "protected" / "agents" / miner.agent / "agent-config.yaml"
    if not profile.is_file():
        raise MineError(
            "не найден профиль агента "
            f"{profile}. Импортируйте agents/typical-examples "
            "(перед импортом создайте .kuibysheff/mine-input)."
        )


def mine_procedure(
    cfg: AppConfig,
    unit: ProcedureUnit,
    runner: Runner,
) -> tuple[list[dict[str, Any]], str]:
    filled = unit if unit.text else replace(unit, text=_procedure_text(cfg, unit))
    digest = _digest(filled.key)
    home_rel = f"homes/mine/{digest}"
    root = Path(cfg.examples_miner.project_root)
    input_dir = root / ".kuibysheff" / "mine-input"
    input_dir.mkdir(parents=True, exist_ok=True)
    procedure_file = input_dir / f"{digest}.bsl"
    procedure_file.write_text(filled.text, encoding="utf-8")
    output_file = root / ".kuibysheff" / home_rel / "out" / "example.json"
    request = MineRequest(
        prompt=build_prompt(filled),
        procedure_file=procedure_file,
        home_rel=home_rel,
        output_file=output_file,
        unit=filled,
    )
    return_code = runner(request)
    if return_code != 0:
        detail = request.stderr.strip() or f"код {return_code}"
        raise ValueError(detail[:500])
    if not output_file.is_file():
        raise ValueError("нет out/example.json")
    try:
        payload = json.loads(output_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"битый JSON: {exc}") from exc
    examples = validate_example_payload(payload, filled)
    if not examples:
        return [], "каркас"
    return examples, ""


def build_prompt(unit: ProcedureUnit) -> str:
    return (
        "Разбери одну процедуру типовой конфигурации 1С и запиши только файл out/example.json.\n"
        f"Конфигурация: {unit.config_label}\n"
        f"Модуль: {unit.module_path}\n"
        f"Процедура: {unit.procedure}\n"
        f"Строки в модуле: {unit.start_line}-{unit.end_line}\n"
        "\n"
        "Формат:\n"
        '{"examples":[{"questions":["...","...","..."],"task":"...","code":"...",'
        '"query_text":"","calls":["..."],"pattern_key":"..."}]}\n'
        "\n"
        "Правила:\n"
        "- questions: 3–5 фраз, как их ввёл бы разработчик, без имени процедуры.\n"
        "- code: точная цитата из процедуры, достаточная чтобы повторить приём. "
        "Не выдумывай код.\n"
        "- query_text: текст запроса, если он есть, иначе пустая строка.\n"
        "- calls: методы платформы и БСП.\n"
        "- pattern_key: короткий ключ приёма.\n"
        '- Если это каркас без прикладной задачи, верни {"examples":[]}.\n'
    )


def validate_example_payload(payload: Any, unit: ProcedureUnit) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("examples"), list):
        raise ValueError("ожидается объект с полем examples")
    validated: list[dict[str, Any]] = []
    procedure_name = unit.procedure.split("#", 1)[0].casefold()
    for index, item in enumerate(payload["examples"]):
        if not isinstance(item, dict):
            raise ValueError(f"examples[{index}] не объект")
        questions = item.get("questions")
        if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
            raise ValueError(f"examples[{index}].questions должен быть списком строк")
        cleaned = [
            question.strip()
            for question in questions
            if question.strip() and question.strip().casefold() != procedure_name
        ]
        if not _MIN_QUESTIONS <= len(cleaned) <= _MAX_QUESTIONS:
            raise ValueError(
                f"examples[{index}].questions: нужно от {_MIN_QUESTIONS} до {_MAX_QUESTIONS}"
            )
        task = str(item.get("task") or "").strip()
        code = str(item.get("code") or "").strip()
        if not task or not code:
            raise ValueError(f"examples[{index}]: нужны task и code")
        if not code_is_citation(code, unit.text):
            raise ValueError(f"examples[{index}].code не цитата из процедуры")
        calls = item.get("calls") or []
        if not isinstance(calls, list) or not all(isinstance(call, str) for call in calls):
            raise ValueError(f"examples[{index}].calls должен быть списком строк")
        example = CodeExample(
            id=f"{unit.config_label}:{unit.module_path}:{unit.procedure}:{index}",
            config_label=unit.config_label,
            module_path=unit.module_path,
            code=code,
            query_text=str(item.get("query_text") or "").strip(),
            start_line=unit.start_line,
            summary=task,
            questions=cleaned,
            task=task,
            calls=[call.strip() for call in calls if call.strip()][:30],
            pattern_key=str(item.get("pattern_key") or "").strip()[:200],
            occurrences=1,
        )
        validated.append(example.to_dict())
    return validated


def code_is_citation(code: str, source: str) -> bool:
    normalized_code = re.sub(r"\s+", " ", code).strip()
    normalized_source = re.sub(r"\s+", " ", source).strip()
    return bool(normalized_code) and normalized_code in normalized_source


def collapse_patterns(examples: list[CodeExample]) -> list[CodeExample]:
    """Keep one canonical example per config_label + pattern_key."""
    grouped: dict[tuple[str, str], CodeExample] = {}
    order: list[tuple[str, str]] = []
    for example in examples:
        pattern = example.pattern_key.strip()
        key = (example.config_label, pattern or example.id)
        if key not in grouped:
            example.occurrences = 1
            grouped[key] = example
            order.append(key)
            continue
        grouped[key].occurrences += 1
    return [grouped[key] for key in order]


def publish_examples(cfg: AppConfig, journal: MineJournal) -> int | None:
    """Rewrite examples.jsonl from done journal rows once the miner owns the file."""
    owned_flag = journal.path.with_name("mine_owned")
    done = [entry for entry in journal.entries.values() if entry.status == DONE]
    collected: list[CodeExample] = []
    ordered = sorted(
        done,
        key=lambda entry: (
            entry.config_index,
            entry.module_rank,
            entry.module_path.lower(),
            entry.start_line,
            entry.procedure,
        ),
    )
    for entry in ordered:
        for raw in entry.examples:
            collected.append(CodeExample.from_dict(raw))
    collapsed = collapse_patterns(collected)
    if not collapsed and not owned_flag.is_file():
        return None
    destination = cfg.data_dir / "examples.jsonl"
    ExamplesStore(destination).save(collapsed)
    owned_flag.write_text("1", encoding="utf-8")
    return len(collapsed)


def kbshff_runner(cfg: AppConfig, request: MineRequest) -> int:
    miner = cfg.examples_miner
    command = [
        miner.binary,
        "run",
        "--project-root",
        miner.project_root,
        "--agent",
        miner.agent,
        "--prompt",
        request.prompt,
        "--files",
        str(request.procedure_file),
        "--max-cost",
        miner.max_cost,
        "--home",
        request.home_rel,
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_RUN_TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired:
        request.stderr = "таймаут kbshff"
        return 124
    request.stderr = (completed.stderr or completed.stdout or "")[-1000:]
    return completed.returncode


def _sync(cfg: AppConfig) -> tuple[MineJournal, set[str], list[ProcedureUnit]]:
    cache_dir = cfg.data_dir / ".sntx_sem_cache"
    journal = MineJournal.load(cache_dir / "mine_journal.jsonl")
    module_cache = ModuleCache.load(cache_dir / "mine_modules.json")
    units, available = discover_units(
        cfg.local_configs,
        min_procedure_lines=cfg.examples_miner.min_procedure_lines,
        cache=module_cache,
    )
    journal.sync(units, available)
    journal.save()
    return journal, available, units


def _procedure_text(cfg: AppConfig, unit: ProcedureUnit) -> str:
    for config in cfg.local_configs:
        if config.label == unit.config_label:
            return read_procedure_text(Path(config.path), unit)
    raise OSError(f"нет каталога для метки {unit.config_label}")


def _digest(key: tuple[str, str, str]) -> str:
    return hashlib.sha256("\0".join(key).encode("utf-8")).hexdigest()[:16]


def _unit_label(entry: Any) -> str:
    return f"{entry.config_label} {entry.module_path} :: {entry.procedure}"
