"""Extract help chunks from HBK files."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from sntx_sem.hbk.container import (
    HbkReader,
    inflate_pack_block,
    open_file_storage,
)
from sntx_sem.hbk.html_to_markdown import extract_sections, html_to_markdown
from sntx_sem.hbk.toc_parser import flatten_toc, parse_toc

DOMAIN_MAP = {
    "shquery": "query_lang",
    "shlang": "bsl_lang",
    "shcntx": "platform_api",
}

LOCALE_MAP = {
    "ru": "ru",
    "root": "en",
}


@dataclass
class HelpChunk:
    id: str
    domain: str
    title_ru: str = ""
    title_en: str = ""
    path: str = ""
    parent_path: str = ""
    entity_kind: str = "page"
    platform_version: str = ""
    hbk_source: str = ""
    locale: str = ""
    text: str = ""
    syntax: str = ""
    parameters: str = ""
    signature: str = ""
    html_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def hbk_stem(filename: str) -> tuple[str, str]:
    """Return (book_kind, locale) from e.g. shquery_root.hbk."""
    name = Path(filename).stem
    parts = name.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], LOCALE_MAP.get(parts[1], parts[1])
    return name, "ru"


def infer_entity_kind(html_path: str, title: str) -> str:
    path = html_path.lower()
    if "/methods/" in path or path.startswith("methods/"):
        return "method"
    if "/properties/" in path or path.startswith("properties/"):
        return "property"
    if "/constructors/" in path or "ctor_" in path:
        return "constructor"
    if "/enums/" in path or "/enum" in path:
        return "enum"
    if "/events/" in path:
        return "event"
    if any(kw in title.upper() for kw in ("SELECT", "JOIN", "WHERE", "GROUP", "UNION")):
        return "query_construct"
    if path.endswith(".html"):
        return "page"
    return "topic"


def resolve_html_entry(html_path: str, zip_names: set[str]) -> str | None:
    path = html_path.replace("\\", "/").lstrip("/")
    if path in zip_names:
        return path
    if not path.endswith(".html"):
        if path in zip_names:
            return path
        for name in zip_names:
            if name.lower() == path.lower():
                return name
    base = Path(path).name
    if base in zip_names:
        return base
    for name in zip_names:
        if name.endswith("/" + base) or name.endswith(base):
            return name
    return None


def extract_hbk(
    hbk_path: Path,
    platform_version: str,
    parent_titles: dict[int, str] | None = None,
) -> list[HelpChunk]:
    reader = HbkReader.from_path(hbk_path)
    pack = reader.get_entity("PackBlock")
    storage = reader.get_entity("FileStorage")
    if not pack or not storage:
        return []

    book_kind, locale = hbk_stem(hbk_path.name)
    domain = DOMAIN_MAP.get(book_kind, "platform_api")

    toc_text = inflate_pack_block(pack).decode("utf-8", errors="replace")
    roots = parse_toc(toc_text)
    pages = flatten_toc(roots)

    with open_file_storage(storage) as zf:
        zip_names = set(zf.namelist())
        chunks: list[HelpChunk] = []

        for page in pages:
            if not page.html_path and not (page.title_ru or page.title_en):
                continue

            html_entry = resolve_html_entry(page.html_path, zip_names) if page.html_path else None
            text = ""
            syntax = ""
            parameters = ""

            if html_entry:
                html = zf.read(html_entry).decode("utf-8", errors="replace")
                markdown = html_to_markdown(html)
                sections = extract_sections(markdown)
                text = markdown
                syntax = sections.get("syntax", "")
                parameters = sections.get("parameters", "")

            title_ru = page.title_ru
            title_en = page.title_en
            if parent_titles and locale == "en" and page.block_id in parent_titles:
                pass

            if locale == "ru":
                title_ru = title_ru or page.title_en
            else:
                title_en = title_en or page.title_ru

            if not text and not (title_ru or title_en):
                continue

            entity_kind = infer_entity_kind(page.html_path or "", title_en or title_ru)
            slug = (
                (html_entry or page.html_path or str(page.block_id))
                .replace("/", "_")
                .replace("\\", "_")
            )
            chunk_id = f"{domain}:{slug}"

            parent_path = ""
            if page.parent_id:
                parent_path = str(page.parent_id)

            chunks.append(
                HelpChunk(
                    id=chunk_id,
                    domain=domain,
                    title_ru=title_ru if locale == "ru" else "",
                    title_en=title_en if locale == "en" else (page.title_en or page.title_ru),
                    path=page.html_path,
                    parent_path=parent_path,
                    entity_kind=entity_kind,
                    platform_version=platform_version,
                    hbk_source=hbk_path.name,
                    locale=locale,
                    text=text or f"# {title_en or title_ru}",
                    syntax=syntax,
                    parameters=parameters,
                    html_path=html_entry or page.html_path,
                )
            )
        return chunks


def merge_bilingual_chunks(chunks: list[HelpChunk]) -> list[HelpChunk]:
    """Merge ru/en pairs by html_path within same domain."""
    by_key: dict[str, HelpChunk] = {}
    for chunk in chunks:
        key = f"{chunk.domain}:{chunk.path or chunk.html_path or chunk.id}"
        existing = by_key.get(key)
        if not existing:
            by_key[key] = chunk
            continue
        if chunk.locale == "ru":
            existing.title_ru = chunk.title_ru or existing.title_ru
            if chunk.text and len(chunk.text) > len(existing.text):
                existing.text = chunk.text
        else:
            existing.title_en = chunk.title_en or existing.title_en
        if chunk.syntax:
            existing.syntax = chunk.syntax
        if chunk.parameters:
            existing.parameters = chunk.parameters
    return list(by_key.values())


def export_chunks_jsonl(chunks: list[HelpChunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def ingest_hbk_dir(hbk_dir: Path, platform_version: str, export_dir: Path) -> dict[str, int]:
    stats: dict[str, int] = {}
    all_chunks: list[HelpChunk] = []

    for hbk_path in sorted(hbk_dir.glob("*.hbk")):
        book_kind, _ = hbk_stem(hbk_path.name)
        chunks = extract_hbk(hbk_path, platform_version)
        domain = DOMAIN_MAP.get(book_kind, book_kind)
        out_file = export_dir / f"{domain}_{hbk_path.stem}.jsonl"
        export_chunks_jsonl(chunks, out_file)
        stats[hbk_path.name] = len(chunks)
        all_chunks.extend(chunks)

    merged = merge_bilingual_chunks(all_chunks)
    export_chunks_jsonl(merged, export_dir / "all_chunks.jsonl")
    stats["merged_total"] = len(merged)
    return stats
