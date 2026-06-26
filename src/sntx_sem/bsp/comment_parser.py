"""Parse 1C export-method comment blocks from BSL source."""

from __future__ import annotations

import re
from dataclasses import dataclass

EXPORT_SIGNATURE = re.compile(
    r"^(?:Функция|Процедура)\s+(\w+)\s*\([^)]*\)\s+Экспорт\s*$",
    re.IGNORECASE,
)
REGION_START = re.compile(r"^\s*#\s*Область\s+(\S+)", re.IGNORECASE)
REGION_END = re.compile(r"^\s*#\s*КонецОбласти", re.IGNORECASE)

SECTION_HEADERS = {
    "parameters": re.compile(r"^Параметры\s*:?\s*$", re.IGNORECASE),
    "return_value": re.compile(r"^Возвращаемое\s+значение\s*:?\s*$", re.IGNORECASE),
    "example": re.compile(r"^Пример(?:\s+реализации)?\s*:?\s*$", re.IGNORECASE),
}

PROGRAM_INTERFACE = "ПрограммныйИнтерфейс"


@dataclass
class ExportMethodDoc:
    name: str
    kind: str
    signature: str
    description: str
    parameters: str
    return_value: str
    example: str
    module_path: str
    line_no: int


def _strip_comment(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("//"):
        return None
    text = stripped[2:]
    if text.startswith(" "):
        text = text[1:]
    return text.rstrip()


def _collect_comment_block(lines: list[str], end_idx: int) -> list[str]:
    """Collect consecutive // comment lines immediately above end_idx."""
    block: list[str] = []
    idx = end_idx - 1
    while idx >= 0:
        text = _strip_comment(lines[idx])
        if text is None:
            break
        block.insert(0, text)
        idx -= 1
    return block


def parse_comment_block(comment_lines: list[str]) -> tuple[str, str, str, str]:
    """Split comment block into description, parameters, return_value, example."""
    if not comment_lines:
        return "", "", "", ""

    sections: dict[str, list[str]] = {
        "description": [],
        "parameters": [],
        "return_value": [],
        "example": [],
    }
    current = "description"

    for line in comment_lines:
        if not line.strip():
            if sections[current]:
                sections[current].append("")
            continue

        matched_section: str | None = None
        for key, pattern in SECTION_HEADERS.items():
            if pattern.match(line.strip()):
                matched_section = key
                break

        if matched_section:
            current = matched_section
            continue

        sections[current].append(line)

    def join_section(key: str) -> str:
        return "\n".join(sections[key]).strip()

    return (
        join_section("description"),
        join_section("parameters"),
        join_section("return_value"),
        join_section("example"),
    )


def _in_program_interface(region_stack: list[str]) -> bool:
    return bool(region_stack) and region_stack[-1] == PROGRAM_INTERFACE


def parse_bsl_exports(content: str, module_path: str = "") -> list[ExportMethodDoc]:
    """Extract export method docs from BSL module content."""
    lines = content.splitlines()
    region_stack: list[str] = []
    results: list[ExportMethodDoc] = []

    for line_no, line in enumerate(lines, start=1):
        region_match = REGION_START.match(line)
        if region_match:
            region_stack.append(region_match.group(1))
            continue

        if REGION_END.match(line):
            if region_stack:
                region_stack.pop()
            continue

        if not _in_program_interface(region_stack):
            continue

        sig_match = EXPORT_SIGNATURE.match(line.strip())
        if not sig_match:
            continue

        name = sig_match.group(1)
        kind = "method" if line.strip().lower().startswith("функция") else "procedure"
        comment_lines = _collect_comment_block(lines, line_no - 1)
        description, parameters, return_value, example = parse_comment_block(comment_lines)

        results.append(
            ExportMethodDoc(
                name=name,
                kind=kind,
                signature=line.strip(),
                description=description,
                parameters=parameters,
                return_value=return_value,
                example=example,
                module_path=module_path,
                line_no=line_no,
            )
        )

    return results
