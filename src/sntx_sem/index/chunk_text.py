"""Chunk text fields for lexical, semantic, and hybrid search."""

from __future__ import annotations

import json
from pathlib import Path

from sntx_sem.index.tokens import split_identifier

_SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "description": ("описание", "description"),
    "parameters": ("параметры", "parameters", "arguments"),
    "returns": ("возвращаемое значение", "return value", "returns"),
    "syntax": ("синтаксис", "syntax"),
}


def _normalize_section_header(line: str) -> str:
    return line.lstrip("#").strip().rstrip(":").strip().lower()


def _match_section_key(line: str) -> str | None:
    normalized = _normalize_section_header(line)
    for key, aliases in _SECTION_ALIASES.items():
        if normalized in aliases:
            return key
    return None


def _compact_lines(lines: list[str]) -> str:
    compact: list[str] = []
    previous_blank = True
    for line in lines:
        text = line.strip()
        if not text:
            if not previous_blank:
                compact.append("")
            previous_blank = True
            continue
        compact.append(text)
        previous_blank = False
    return "\n".join(compact).strip()


def _extract_behavior_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {
        "description": [],
        "parameters": [],
        "returns": [],
        "syntax": [],
    }
    lead_description: list[str] = []
    current: str | None = None
    lead_locked = False
    for line in text.splitlines():
        section_key = _match_section_key(line)
        if section_key:
            current = section_key
            continue
        if line.strip().startswith("#"):
            continue
        if not line.strip():
            if current:
                sections[current].append(line)
            elif lead_description:
                lead_locked = True
            continue
        if current:
            sections[current].append(line)
        else:
            if lead_locked:
                continue
            lead_description.append(line)
    if lead_description and not sections["description"]:
        sections["description"] = lead_description
    return {key: _compact_lines(lines) for key, lines in sections.items()}


def build_lexical_text(chunk: dict) -> str:
    parts: list[str] = []
    for key in ("title_ru", "title_en"):
        title = chunk.get(key, "")
        if title:
            parts.append(title)
            parts.append(split_identifier(title))
    if chunk.get("domain") == "bsp":
        path = chunk.get("path", "")
        if path:
            parts.append(path)
            parts.append(split_identifier(path))
    return " ".join(p for p in parts if p).strip()


def build_semantic_text(chunk: dict) -> str:
    text = chunk.get("text", "")
    if not text:
        return ""
    sections = _extract_behavior_sections(text)
    parts: list[str] = []
    for key in ("description", "syntax", "parameters", "returns"):
        section = sections.get(key, "")
        if section:
            parts.append(section)
    syntax = str(chunk.get("syntax", "")).strip()
    if syntax and syntax not in parts:
        parts.append(syntax)
    parameters = str(chunk.get("parameters", "")).strip()
    if parameters and parameters not in parts:
        parts.append(parameters)
    if not parts:
        body_lines = [line for line in text.splitlines() if not line.strip().startswith("#")]
        parts.append(_compact_lines(body_lines))
    return "\n\n".join(part for part in parts if part).strip()


def build_search_text(
    chunk: dict,
    *,
    lexical_text: str | None = None,
    semantic_text: str | None = None,
) -> str:
    lexical = lexical_text if lexical_text is not None else build_lexical_text(chunk)
    semantic = semantic_text if semantic_text is not None else build_semantic_text(chunk)
    return " ".join(part for part in (lexical, semantic) if part).strip()


def ensure_chunk_text_fields(chunk: dict) -> dict:
    lexical_text = str(chunk.get("lexical_text") or "").strip()
    if not lexical_text:
        lexical_text = build_lexical_text(chunk)
    semantic_text = str(chunk.get("semantic_text") or "").strip()
    if not semantic_text:
        semantic_text = build_semantic_text(chunk)
    search_text = str(chunk.get("search_text") or "").strip()
    if not search_text:
        search_text = build_search_text(
            chunk,
            lexical_text=lexical_text,
            semantic_text=semantic_text,
        )
    chunk["lexical_text"] = lexical_text
    chunk["semantic_text"] = semantic_text
    chunk["search_text"] = search_text
    return chunk


def chunk_meta_row(
    chunk: dict,
    text: str,
    *,
    lexical_text: str | None = None,
    semantic_text: str | None = None,
) -> dict:
    return {
        "id": chunk["id"],
        "domain": chunk.get("domain", ""),
        "title_ru": chunk.get("title_ru", ""),
        "title_en": chunk.get("title_en", ""),
        "entity_kind": chunk.get("entity_kind", ""),
        "html_path": chunk.get("html_path", ""),
        "syntax": chunk.get("syntax", ""),
        "text": chunk.get("text", ""),
        "search_text": text,
        "lexical_text": lexical_text if lexical_text is not None else chunk.get("lexical_text", ""),
        "semantic_text": semantic_text
        if semantic_text is not None
        else chunk.get("semantic_text", ""),
        "parameters": chunk.get("parameters", ""),
        "signature": chunk.get("signature", ""),
    }


def write_chunks_meta_json(meta_path: Path, meta_rows: list[dict]) -> None:
    """Write chunks_meta.json as a JSON array without one large json.dumps buffer."""
    with meta_path.open("w", encoding="utf-8") as handle:
        handle.write("[\n")
        for index, row in enumerate(meta_rows):
            if index:
                handle.write(",\n")
            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "domain": row["domain"],
                        "title_ru": row["title_ru"],
                        "title_en": row["title_en"],
                        "search_text": row["search_text"],
                        "lexical_text": row.get("lexical_text", ""),
                        "semantic_text": row.get("semantic_text", ""),
                        "entity_kind": row["entity_kind"],
                        "html_path": row["html_path"],
                        "syntax": row["syntax"],
                        "text": row["text"],
                        "parameters": row.get("parameters", ""),
                        "signature": row.get("signature", ""),
                    },
                    ensure_ascii=False,
                )
            )
        handle.write("\n]")
