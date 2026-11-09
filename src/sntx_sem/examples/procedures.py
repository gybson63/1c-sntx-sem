"""Split BSL modules into procedures and order them for mining."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sntx_sem.config.models import LocalConfig
from sntx_sem.examples.scanner import BSL_EXTENSIONS, write_text_atomic

_PROC_START = re.compile(
    r"^\s*(?:(?:Асинх|Async)\s+)?"
    r"(?:Процедура|Функция|Procedure|Function)\s+"
    r"([A-Za-zА-Яа-яЁё_][A-Za-zА-Яа-яЁё0-9_]*)\s*\(",
    re.IGNORECASE,
)
_PROC_END = re.compile(
    r"^\s*(?:КонецПроцедуры|КонецФункции|EndProcedure|EndFunction)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ProcedureUnit:
    config_label: str
    config_index: int
    module_path: str
    module_rank: int
    procedure: str
    start_line: int
    end_line: int
    content_hash: str
    text: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.config_label, self.module_path, self.procedure)


def module_rank(relative_path: str) -> int:
    """Lower rank is mined earlier. Forms stay at the end of a configuration."""
    normalized = relative_path.replace("\\", "/").lower()
    name = normalized.rsplit("/", 1)[-1]
    if normalized.startswith("commonmodules/") or "/commonmodules/" in f"/{normalized}":
        return 0
    if name == "managermodule.bsl":
        return 1
    if name == "objectmodule.bsl":
        return 2
    if name == "recordsetmodule.bsl":
        return 3
    if name == "commandmodule.bsl" or "/commands/" in f"/{normalized}":
        return 4
    if "/forms/" in f"/{normalized}" or "/form/" in f"/{normalized}":
        return 5
    return 6


def split_procedures(content: str) -> list[tuple[str, int, int, str]]:
    """Return (name, start_line, end_line, text) for each closed procedure."""
    lines = content.splitlines()
    blanked = _blank_strings_and_comments(lines)
    found: list[tuple[str, int, int, str]] = []
    start_index: int | None = None
    name = ""
    for index, line in enumerate(blanked):
        if start_index is None:
            match = _PROC_START.match(line)
            if match:
                start_index = index
                name = match.group(1)
            continue
        if _PROC_END.match(line):
            chunk = "\n".join(lines[start_index : index + 1])
            found.append((name, start_index + 1, index + 1, chunk))
            start_index = None
            name = ""
    return found


@dataclass
class CachedProcedure:
    procedure: str
    start_line: int
    end_line: int
    content_hash: str


@dataclass
class CachedModule:
    mtime_ns: int
    size: int
    procedures: list[CachedProcedure] = field(default_factory=list)


class ModuleCache:
    """Skip re-parsing a module while its size and mtime stay the same."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: dict[tuple[str, str], CachedModule] = {}

    @classmethod
    def load(cls, path: Path) -> ModuleCache:
        cache = cls(path)
        if not path.is_file():
            return cache
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return cache
        if not isinstance(payload, list):
            return cache
        for item in payload:
            try:
                procedures = [CachedProcedure(**proc) for proc in item["procedures"]]
                key = (item["config_label"], item["module_path"])
                cache.records[key] = CachedModule(item["mtime_ns"], item["size"], procedures)
            except (KeyError, TypeError):
                continue
        return cache

    def save(self) -> None:
        payload = [
            {
                "config_label": label,
                "module_path": module_path,
                "mtime_ns": record.mtime_ns,
                "size": record.size,
                "procedures": [asdict(proc) for proc in record.procedures],
            }
            for (label, module_path), record in sorted(self.records.items())
        ]
        write_text_atomic(self.path, json.dumps(payload, ensure_ascii=False))

    def drop_unseen(self, available: set[str], seen: set[tuple[str, str]]) -> None:
        for key in list(self.records):
            if key[0] in available and key not in seen:
                del self.records[key]


def discover_units(
    configs: list[LocalConfig],
    *,
    min_procedure_lines: int,
    cache: ModuleCache | None = None,
) -> tuple[list[ProcedureUnit], set[str]]:
    """Walk every available local config. Missing directories are not available.

    Procedure bodies are not kept. Unchanged modules are taken from `cache`.
    """
    units: list[ProcedureUnit] = []
    available: set[str] = set()
    seen: set[tuple[str, str]] = set()
    for config_index, config in enumerate(configs):
        root = Path(config.path)
        if not root.is_dir():
            continue
        available.add(config.label)
        files: list[tuple[int, str, Path]] = []
        for path in root.rglob("*"):
            if path.suffix.lower() not in BSL_EXTENSIONS or not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            files.append((module_rank(relative), relative, path))
        files.sort(key=lambda item: (item[0], item[1].lower()))
        for rank, relative, path in files:
            key = (config.label, relative)
            seen.add(key)
            try:
                stat = path.stat()
            except OSError:
                continue
            cached = cache.records.get(key) if cache is not None else None
            if (
                cached is not None
                and cached.mtime_ns == stat.st_mtime_ns
                and cached.size == stat.st_size
            ):
                procedures = cached.procedures
            else:
                try:
                    content = path.read_text(encoding="utf-8-sig", errors="replace")
                except OSError:
                    continue
                procedures = _index_procedures(content)
                if cache is not None:
                    cache.records[key] = CachedModule(stat.st_mtime_ns, stat.st_size, procedures)
            for item in procedures:
                if item.end_line - item.start_line + 1 < min_procedure_lines:
                    continue
                units.append(
                    ProcedureUnit(
                        config_label=config.label,
                        config_index=config_index,
                        module_path=relative,
                        module_rank=rank,
                        procedure=item.procedure,
                        start_line=item.start_line,
                        end_line=item.end_line,
                        content_hash=item.content_hash,
                        text="",
                    )
                )
    if cache is not None:
        cache.drop_unseen(available, seen)
        cache.save()
    return units, available


def read_procedure_text(root: Path, unit: ProcedureUnit) -> str:
    raw = (root / unit.module_path).read_text(encoding="utf-8-sig", errors="replace")
    lines = raw.splitlines()
    chunk = lines[unit.start_line - 1 : unit.end_line]
    if not chunk:
        raise OSError(f"пустой фрагмент {unit.module_path}:{unit.start_line}")
    return "\n".join(chunk)


def _index_procedures(content: str) -> list[CachedProcedure]:
    seen_names: set[str] = set()
    indexed: list[CachedProcedure] = []
    for name, start_line, end_line, text in split_procedures(content):
        procedure = name
        if procedure in seen_names:
            procedure = f"{name}#{start_line}"
        seen_names.add(procedure)
        indexed.append(CachedProcedure(procedure, start_line, end_line, _content_hash(text)))
    return indexed


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _blank_strings_and_comments(lines: list[str]) -> list[str]:
    blanked: list[str] = []
    in_string = False
    for line in lines:
        output: list[str] = []
        index = 0
        length = len(line)
        while index < length:
            char = line[index]
            if in_string:
                if char == '"':
                    if index + 1 < length and line[index + 1] == '"':
                        output.append("  ")
                        index += 2
                        continue
                    in_string = False
                    output.append(" ")
                    index += 1
                    continue
                output.append(" ")
                index += 1
                continue
            if char == '"':
                in_string = True
                output.append(" ")
                index += 1
                continue
            if char == "/" and index + 1 < length and line[index + 1] == "/":
                output.append(" " * (length - index))
                break
            output.append(char)
            index += 1
        blanked.append("".join(output))
    return blanked
