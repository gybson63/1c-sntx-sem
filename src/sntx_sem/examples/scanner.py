"""Scan local 1C configurations for code examples."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

BSL_EXTENSIONS = {".bsl", ".os"}
QUERY_PATTERN = re.compile(
    r'(?i)(?:Запрос\s*=\s*Новый\s+Запрос|New\s+Query)\s*\(\s*"([^"]*(?:\\.[^"]*)*)"',
    re.DOTALL,
)
STRING_QUERY_PATTERN = re.compile(
    r'"((?:[^"\\]|\\.)*(?:SELECT|ВЫБРАТЬ|SELECT\s|ВЫБРАТЬ\s)[^"]*(?:[^"\\]|\\.)*)"',
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class CodeExample:
    id: str
    config_label: str
    module_path: str
    code: str
    query_text: str = ""
    start_line: int = 0
    topic_ids: list[str] | None = None
    relevance: float = 0.0
    summary: str = ""
    questions: list[str] | None = None
    task: str = ""
    calls: list[str] | None = None
    pattern_key: str = ""
    occurrences: int = 1

    def to_dict(self) -> dict:
        data = asdict(self)
        if data["topic_ids"] is None:
            data["topic_ids"] = []
        if data["questions"] is None:
            data["questions"] = []
        if data["calls"] is None:
            data["calls"] = []
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodeExample:
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _load_cache(cache_dir: Path) -> dict[str, Any]:
    cache_file = cache_dir / "scan_cache.json"
    if cache_file.is_file():
        return cast(dict[str, Any], json.loads(cache_file.read_text(encoding="utf-8")))
    return {}


def _save_cache(cache_dir: Path, cache: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "scan_cache.json").write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def extract_queries_from_bsl(content: str) -> list[str]:
    queries: list[str] = []
    for pattern in (QUERY_PATTERN, STRING_QUERY_PATTERN):
        for match in pattern.finditer(content):
            text = match.group(1).replace('\\"', '"').replace("\\n", "\n")
            if len(text.strip()) > 20:
                queries.append(text.strip())
    return queries


def scan_config_path(
    config_path: Path,
    label: str,
    cache_dir: Path | None = None,
) -> list[CodeExample]:
    if not config_path.is_dir():
        return []

    cache = _load_cache(cache_dir) if cache_dir else {}
    examples: list[CodeExample] = []

    for bsl_file in config_path.rglob("*"):
        if bsl_file.suffix.lower() not in BSL_EXTENSIONS:
            continue
        if not bsl_file.is_file():
            continue

        rel = str(bsl_file.relative_to(config_path))
        file_key = rel
        current_hash = _file_hash(bsl_file)
        if cache.get(file_key) == current_hash:
            continue

        try:
            content = bsl_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        queries = extract_queries_from_bsl(content)
        lines = content.splitlines()

        if queries:
            for qi, query_text in enumerate(queries):
                pos = content.find(query_text[: min(40, len(query_text))])
                line_no = content[:pos].count("\n") + 1 if pos >= 0 else 0
                start = max(0, line_no - 15)
                end = min(len(lines), line_no + 15)
                snippet = "\n".join(lines[start:end])
                ex_id = f"{label}:{rel}:{qi}"
                examples.append(
                    CodeExample(
                        id=ex_id,
                        config_label=label,
                        module_path=rel,
                        code=snippet,
                        query_text=query_text,
                        start_line=line_no,
                    )
                )
        elif len(content.strip()) > 50:
            examples.append(
                CodeExample(
                    id=f"{label}:{rel}:0",
                    config_label=label,
                    module_path=rel,
                    code=content[:2000],
                )
            )

        if cache_dir is not None:
            cache[file_key] = current_hash

    if cache_dir is not None:
        _save_cache(cache_dir, cache)
    return examples


def keyword_link_examples(
    examples: list[CodeExample],
    topics: list[dict],
    min_overlap: int = 2,
) -> list[CodeExample]:
    """Smoke-test linking via keyword overlap (no LLM)."""
    linked: list[CodeExample] = []
    for ex in examples:
        text = (ex.code + " " + ex.query_text).lower()
        scores: list[tuple[str, float]] = []
        for topic in topics:
            title = f"{topic.get('title_ru', '')} {topic.get('title_en', '')}".lower()
            words = [w for w in title.split() if len(w) > 3]
            if not words:
                continue
            overlap = sum(1 for w in words if w in text)
            if overlap >= min_overlap:
                scores.append((topic["id"], overlap / len(words)))

        scores.sort(key=lambda x: x[1], reverse=True)
        if scores:
            ex.topic_ids = [s[0] for s in scores[:3]]
            ex.relevance = scores[0][1]
            linked.append(ex)
    return linked


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def export_examples_jsonl(examples: list[CodeExample], path: Path) -> None:
    payload = "".join(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n" for ex in examples)
    write_text_atomic(path, payload)
