"""Extract BSP public API as HelpChunk records."""

from __future__ import annotations

import json
import re
from pathlib import Path

from sntx_sem.bsp.comment_parser import ExportMethodDoc, parse_bsl_exports
from sntx_sem.hbk.extractor import HelpChunk, export_chunks_jsonl

BSP_DOMAIN = "bsp"
MODULE_BSL_GLOB = "CommonModules/*/Ext/Module.bsl"
VERSION_PATTERN = re.compile(r"<Version>([^<]+)</Version>")


def read_bsp_version(bsp_dir: Path) -> str:
    config_xml = bsp_dir / "Configuration.xml"
    if not config_xml.is_file():
        return ""
    text = config_xml.read_text(encoding="utf-8", errors="replace")
    match = VERSION_PATTERN.search(text)
    return match.group(1).strip() if match else ""


def module_name_from_path(rel_path: str) -> str:
    parts = Path(rel_path.replace("\\", "/")).parts
    if len(parts) >= 2 and parts[0] == "CommonModules":
        return parts[1]
    return Path(rel_path).stem


def build_markdown_text(module_name: str, doc: ExportMethodDoc) -> str:
    parts = [f"# {module_name}.{doc.name}", ""]

    if doc.description:
        parts.extend([doc.description, ""])

    if doc.parameters:
        parts.extend(["## Параметры", "", doc.parameters, ""])

    if doc.return_value:
        parts.extend(["## Возвращаемое значение", "", doc.return_value, ""])

    if doc.example:
        parts.extend(["## Пример", "", doc.example, ""])

    parts.extend(["## Синтаксис", "", f"```bsl\n{doc.signature}\n```", ""])
    parts.append(f"Модуль: `{module_name}`")
    return "\n".join(parts).strip()


def doc_to_chunk(
    doc: ExportMethodDoc,
    module_name: str,
    rel_path: str,
    bsp_version: str,
) -> HelpChunk:
    qualified = f"{module_name}.{doc.name}"
    return HelpChunk(
        id=f"{BSP_DOMAIN}:{qualified}",
        domain=BSP_DOMAIN,
        title_ru=qualified,
        title_en="",
        path=f"CommonModules/{module_name}",
        parent_path=f"CommonModules/{module_name}",
        entity_kind=doc.kind,
        platform_version=bsp_version,
        hbk_source="bsp",
        locale="ru",
        text=build_markdown_text(module_name, doc),
        syntax=doc.signature,
        parameters=doc.parameters,
        signature=qualified,
        html_path=rel_path.replace("\\", "/"),
    )


def extract_bsp_dir(bsp_dir: Path) -> tuple[list[HelpChunk], dict[str, int | str]]:
    """Scan CommonModules and extract public API from #Область ПрограммныйИнтерфейс."""
    if not bsp_dir.is_dir():
        return [], {"modules": 0, "methods": 0, "without_comment": 0}

    bsp_version = read_bsp_version(bsp_dir)
    chunks: list[HelpChunk] = []
    modules_seen: set[str] = set()
    without_comment = 0

    for bsl_file in sorted(bsp_dir.glob(MODULE_BSL_GLOB)):
        if not bsl_file.is_file():
            continue

        rel_path = str(bsl_file.relative_to(bsp_dir))
        module_name = module_name_from_path(rel_path)
        modules_seen.add(module_name)

        try:
            content = bsl_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for doc in parse_bsl_exports(content, rel_path):
            if not doc.description and not doc.parameters and not doc.return_value:
                without_comment += 1
            chunks.append(doc_to_chunk(doc, module_name, rel_path, bsp_version))

    stats: dict[str, int | str] = {
        "modules": len(modules_seen),
        "methods": len(chunks),
        "without_comment": without_comment,
        "bsp_version": bsp_version,
    }
    return chunks, stats


def merge_bsp_into_all_chunks(bsp_chunks: list[HelpChunk], all_chunks_path: Path) -> int:
    """Replace bsp domain entries in all_chunks.jsonl, keep HBK chunks."""
    existing: list[dict] = []
    if all_chunks_path.is_file():
        with all_chunks_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    if chunk.get("domain") != BSP_DOMAIN:
                        existing.append(chunk)

    merged = existing + [c.to_dict() for c in bsp_chunks]
    all_chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with all_chunks_path.open("w", encoding="utf-8") as f:
        for chunk in merged:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return len(merged)


def ingest_bsp(bsp_dir: Path, export_dir: Path) -> dict[str, int | str]:
    """Extract BSP API and write bsp_api.jsonl + update all_chunks.jsonl."""
    export_dir.mkdir(parents=True, exist_ok=True)
    chunks, stats = extract_bsp_dir(bsp_dir)

    bsp_jsonl = export_dir / "bsp_api.jsonl"
    export_chunks_jsonl(chunks, bsp_jsonl)

    all_chunks_path = export_dir / "all_chunks.jsonl"
    stats["merged_total"] = merge_bsp_into_all_chunks(chunks, all_chunks_path)
    stats["bsp_chunks"] = len(chunks)
    return stats
