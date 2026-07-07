"""Vector index and hybrid search."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
from rank_bm25 import BM25Okapi

from sntx_sem.embeddings.base import EmbeddingBackend
from sntx_sem.index.integrity import drop_lance_table, lance_table_present

_DOMAIN_FILTER_RE = re.compile(r"^[\w]+$")
_IDENTIFIER_WORD_RE = re.compile(r"[А-ЯЁA-Z][а-яёa-z]*")
_QUERY_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


def split_identifier(name: str) -> str:
    """Split 1C CamelCase identifier (Cyrillic/Latin) into spaced lowercase words."""
    words: list[str] = []
    for segment in re.split(r"[.\s]+", name.strip()):
        if not segment:
            continue
        parts = _IDENTIFIER_WORD_RE.findall(segment)
        if parts:
            words.extend(parts)
        else:
            words.append(segment)
    return " ".join(word.lower() for word in words)


def _normalize_text(text: str) -> str:
    return " ".join(_QUERY_TOKEN_RE.findall(text.lower()))


def _tokenize_text(text: str) -> list[str]:
    return _QUERY_TOKEN_RE.findall(text.lower())


_RU_SUFFIXES = (
    "ами",
    "ями",
    "ого",
    "ему",
    "ими",
    "ому",
    "ах",
    "ях",
    "ов",
    "ев",
    "ей",
    "ий",
    "ый",
    "ая",
    "ое",
    "ые",
    "у",
    "а",
    "е",
    "и",
    "о",
    "ы",
    "ь",
    "ю",
    "я",
)

_QUERY_STOPWORDS = {"и", "в", "во", "на", "по", "к", "с", "со", "из", "для", "о", "об", "от"}
_EXECUTABLE_KINDS = {"method", "function", "constructor", "procedure"}
_STATIC_KINDS = {"property", "type", "enum", "topic", "page"}
_ACTION_STEMS = (
    "раздел",
    "разбив",
    "преобраз",
    "конверт",
    "соедин",
    "подстав",
    "объедин",
    "split",
    "concat",
)


def _token_stem(token: str) -> str:
    for suffix in _RU_SUFFIXES:
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _tokens_related(query_token: str, title_token: str) -> bool:
    if query_token == title_token:
        return True
    if (
        len(query_token) >= 3
        and len(title_token) >= 3
        and (query_token.startswith(title_token) or title_token.startswith(query_token))
    ):
        return True
    query_stem = _token_stem(query_token)
    title_stem = _token_stem(title_token)
    if query_stem == title_stem:
        return True
    return (
        len(query_stem) >= 3
        and len(title_stem) >= 3
        and (query_stem.startswith(title_stem) or title_stem.startswith(query_stem))
    )


def _query_content_stems(query: str) -> list[str]:
    stems: list[str] = []
    for token in _tokenize_text(query):
        stem = _token_stem(token)
        if len(stem) < 3:
            continue
        if stem in _QUERY_STOPWORDS:
            continue
        if stem not in stems:
            stems.append(stem)
    return stems


def _semantic_stem_set(chunk: dict) -> set[str]:
    semantic_text = chunk.get("semantic_text", "")
    if not semantic_text:
        semantic_text = chunk.get("text", "")
    return {_token_stem(token) for token in _tokenize_text(semantic_text)}


def _looks_like_transformation_query(query: str) -> bool:
    tokens = _tokenize_text(query)
    return "в" in tokens and len(_query_content_stems(query)) >= 2


def _semantic_query_text(query: str) -> str:
    if not _looks_like_transformation_query(query):
        return query
    tokens = _tokenize_text(query)
    content_tokens = [
        token
        for token in tokens
        if len(_token_stem(token)) >= 3 and _token_stem(token) not in _QUERY_STOPWORDS
    ]
    if len(content_tokens) < 2:
        return query
    source = content_tokens[0]
    target = content_tokens[-1]
    expanded = [
        query,
        f"преобразовать {source} в {target}",
        f"получить {target} из {source}",
    ]
    source_stem = _token_stem(source)
    target_stem = _token_stem(target)
    if source_stem.startswith("строк") and target_stem.startswith("масс"):
        expanded.append("разделить строку")
    return ". ".join(expanded)


def _resolve_domain_filter(domain: str) -> str:
    if domain == "bsl":
        return "bsl_lang"
    if domain == "query":
        return "query_lang"
    return domain


def _domain_where_clause(domain_filter: str) -> str:
    if not _DOMAIN_FILTER_RE.fullmatch(domain_filter):
        raise ValueError(f"Invalid domain filter: {domain_filter!r}")
    return f"domain = '{domain_filter}'"


def _indexed_columns(table: Any) -> set[str]:
    columns: set[str] = set()
    for item in table.list_indices():
        if isinstance(item, dict):
            cols = item.get("columns", [])
        else:
            cols = getattr(item, "columns", []) or []
        columns.update(cols)
    return columns


def _vector_index_partitions(row_count: int) -> int:
    if row_count <= 32:
        return max(1, row_count // 4 or 1)
    return max(8, min(256, int(row_count**0.5)))


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


def _build_lexical_text(chunk: dict) -> str:
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


def _build_semantic_text(chunk: dict) -> str:
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


def _build_search_text(
    chunk: dict,
    *,
    lexical_text: str | None = None,
    semantic_text: str | None = None,
) -> str:
    lexical = lexical_text if lexical_text is not None else _build_lexical_text(chunk)
    semantic = semantic_text if semantic_text is not None else _build_semantic_text(chunk)
    return " ".join(part for part in (lexical, semantic) if part).strip()


def _bm25_tokens(text: str) -> list[str]:
    return [_token_stem(token) for token in _tokenize_text(text)]


def _ensure_chunk_text_fields(chunk: dict) -> dict:
    lexical_text = str(chunk.get("lexical_text") or "").strip()
    if not lexical_text:
        lexical_text = _build_lexical_text(chunk)
    semantic_text = str(chunk.get("semantic_text") or "").strip()
    if not semantic_text:
        semantic_text = _build_semantic_text(chunk)
    search_text = str(chunk.get("search_text") or "").strip()
    if not search_text:
        search_text = _build_search_text(
            chunk,
            lexical_text=lexical_text,
            semantic_text=semantic_text,
        )
    chunk["lexical_text"] = lexical_text
    chunk["semantic_text"] = semantic_text
    chunk["search_text"] = search_text
    return chunk


def _chunk_meta_row(
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


def _write_chunks_meta_json(meta_path: Path, meta_rows: list[dict]) -> None:
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


@dataclass
class SearchScoreBreakdown:
    dense_rrf: float = 0.0
    dense_similarity: float = 0.0
    bm25_rrf: float = 0.0
    title_bonus: float = 0.0
    semantic_intent_bonus: float = 0.0
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_distance: float | None = None
    bm25_raw: float | None = None
    total: float = 0.0

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "total": self.total,
            "dense_rrf": self.dense_rrf,
            "dense_similarity": self.dense_similarity,
            "bm25_rrf": self.bm25_rrf,
            "title_bonus": self.title_bonus,
            "semantic_intent_bonus": self.semantic_intent_bonus,
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "dense_distance": self.dense_distance,
            "bm25_raw": self.bm25_raw,
        }


@dataclass
class SearchFusion:
    scores: dict[str, float]
    breakdowns: dict[str, SearchScoreBreakdown]


@dataclass
class SearchResult:
    id: str
    domain: str
    title: str
    text: str
    score: float
    entity_kind: str = ""
    html_path: str = ""
    syntax: str = ""
    highlight_terms: list[str] = field(default_factory=list)
    match_sources: list[str] = field(default_factory=list)
    match_explanation: str = ""
    semantic_excerpt: str = ""
    semantic_highlight_terms: list[str] = field(default_factory=list)
    score_breakdown: dict[str, float | int | None] = field(default_factory=dict)


def _collect_matched_terms(query: str, search_text: str) -> list[str]:
    query_tokens = _tokenize_text(query)
    if not query_tokens or not search_text:
        return []
    search_tokens = set(_tokenize_text(search_text))
    matched: list[str] = []
    for token in query_tokens:
        if token in search_tokens and token not in matched:
            matched.append(token)
    return matched


def _semantic_text_for_explanation(chunk: dict) -> str:
    semantic_text = chunk.get("semantic_text", "")
    if isinstance(semantic_text, str) and semantic_text:
        return semantic_text
    text = chunk.get("text", "")
    return text if isinstance(text, str) else ""


def _collect_semantic_highlight_terms(query: str, chunk: dict) -> list[str]:
    query_stems = _query_content_stems(query)
    semantic_text = _semantic_text_for_explanation(chunk)
    if not query_stems or not semantic_text:
        return []

    terms: list[str] = []
    transformation_intent = _looks_like_transformation_query(query)
    for token in _tokenize_text(semantic_text):
        stem = _token_stem(token)
        if len(stem) < 3:
            continue
        matches_query = any(
            stem == query_stem or stem.startswith(query_stem) or query_stem.startswith(stem)
            for query_stem in query_stems
        )
        matches_intent = transformation_intent and any(
            stem.startswith(action_stem) for action_stem in _ACTION_STEMS
        )
        if (matches_query or matches_intent) and token not in terms:
            terms.append(token)
        if len(terms) >= 8:
            break
    return terms


def _find_first_term_position(text: str, terms: list[str]) -> int | None:
    lowered = text.lower()
    positions = [lowered.find(term.lower()) for term in terms if term]
    found = [position for position in positions if position >= 0]
    return min(found) if found else None


def _build_semantic_excerpt(text: str, terms: list[str], max_chars: int = 360) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text

    match_pos = _find_first_term_position(text, terms)
    if match_pos is None:
        return text[:max_chars] + "..."

    half_window = max_chars // 2
    start = max(0, match_pos - half_window)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)
    snippet = text[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


class HelpIndex:
    TABLE_NAME = "help_chunks"
    MIN_VECTOR_INDEX_ROWS = 256

    def __init__(
        self,
        index_dir: Path,
        embedding_backend: EmbeddingBackend,
        search_config: Any,
    ) -> None:
        self.index_dir = index_dir
        self.embedding_backend = embedding_backend
        self.search_config = search_config
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.index_dir.resolve()))
        self._chunks: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._bm25_by_domain: dict[str, BM25Okapi] = {}
        self._domain_chunk_indices: dict[str, list[int]] = {}
        self._chunk_map: dict[str, dict] | None = None
        self._table = None
        self._indices_ready = False

    def load_chunks_from_jsonl(self, jsonl_path: Path) -> list[dict]:
        chunks: list[dict] = []
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        return chunks

    def build(
        self,
        chunks: list[dict],
        rebuild: bool = True,
        batch_size: int = 256,
        on_progress: Callable[[int, int], None] | None = None,
        on_log: Callable[[str], None] | None = None,
    ) -> tuple[int, int | None, bool]:
        if not chunks:
            return 0, None, False

        total = len(chunks)
        if rebuild and lance_table_present(self.index_dir, self.TABLE_NAME):
            drop_lance_table(self.db, self.index_dir, self.TABLE_NAME)
            self._table = None
            self._indices_ready = False
            self._bm25_by_domain.clear()
            self._domain_chunk_indices.clear()
            self._chunk_map = None

        meta_rows: list[dict] = []
        dimensions: int | None = None
        schema: pa.Schema | None = None
        row_count = 0

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            semantic_texts = [_build_semantic_text(c) for c in batch]
            lexical_texts = [_build_lexical_text(c) for c in batch]
            search_texts = [
                _build_search_text(c, lexical_text=lexical, semantic_text=semantic)
                for c, lexical, semantic in zip(batch, lexical_texts, semantic_texts, strict=True)
            ]
            vectors = self.embedding_backend.embed_passages(semantic_texts)
            if dimensions is None and len(vectors) > 0:
                dimensions = int(vectors.shape[1])
            batch_rows: list[dict] = []
            for chunk, vector, semantic_text, lexical_text, search_text in zip(
                batch, vectors, semantic_texts, lexical_texts, search_texts, strict=True
            ):
                batch_rows.append(
                    {
                        "id": chunk["id"],
                        "domain": chunk.get("domain", ""),
                        "title_ru": chunk.get("title_ru", ""),
                        "title_en": chunk.get("title_en", ""),
                        "entity_kind": chunk.get("entity_kind", ""),
                        "html_path": chunk.get("html_path", ""),
                        "syntax": chunk.get("syntax", ""),
                        "text": chunk.get("text", ""),
                        "semantic_text": semantic_text,
                        "lexical_text": lexical_text,
                        "search_text": search_text,
                        "vector": vector.tolist(),
                    }
                )
                meta_rows.append(
                    _chunk_meta_row(
                        chunk,
                        search_text,
                        lexical_text=lexical_text,
                        semantic_text=semantic_text,
                    )
                )

            if schema is None:
                dim = len(batch_rows[0]["vector"])
                schema = pa.schema(
                    [
                        pa.field("id", pa.string()),
                        pa.field("domain", pa.string()),
                        pa.field("title_ru", pa.string()),
                        pa.field("title_en", pa.string()),
                        pa.field("entity_kind", pa.string()),
                        pa.field("html_path", pa.string()),
                        pa.field("syntax", pa.string()),
                        pa.field("text", pa.string()),
                        pa.field("semantic_text", pa.string()),
                        pa.field("lexical_text", pa.string()),
                        pa.field("search_text", pa.string()),
                        pa.field("vector", pa.list_(pa.float32(), dim)),
                    ]
                )
                self._table = self.db.create_table(
                    self.TABLE_NAME,
                    data=batch_rows,
                    schema=schema,
                    mode="overwrite",
                )
            else:
                assert self._table is not None
                self._table.add(batch_rows)

            row_count += len(batch_rows)
            if on_progress:
                on_progress(min(start + len(batch), total), total)

        del chunks
        assert self._table is not None
        vector_index_built = self._create_search_indices(row_count, on_log=on_log)
        self._chunks = [_ensure_chunk_text_fields(row) for row in meta_rows]
        self._chunk_map = None
        self._bm25 = BM25Okapi([_bm25_tokens(r.get("lexical_text", "")) for r in self._chunks])
        _write_chunks_meta_json(self.index_dir / "chunks_meta.json", meta_rows)
        return len(meta_rows), dimensions, vector_index_built

    def _create_search_indices(
        self,
        row_count: int,
        on_log: Callable[[str], None] | None = None,
    ) -> bool:
        if self._table is None or row_count <= 0:
            return False
        indexed = _indexed_columns(self._table)
        build_vector_index = bool(getattr(self.search_config, "build_vector_index", True))
        vector_index_built = "vector" in indexed

        if (
            build_vector_index
            and not vector_index_built
            and row_count >= self.MIN_VECTOR_INDEX_ROWS
        ):
            partitions = _vector_index_partitions(row_count)
            if on_log:
                on_log(f"Создание векторного индекса LanceDB (IVF, partitions={partitions})…")
            self._table.create_index(
                metric="cosine",
                num_partitions=partitions,
                vector_column_name="vector",
            )
            vector_index_built = True

        if "domain" not in indexed:
            self._table.create_scalar_index("domain")
        self._indices_ready = True
        return vector_index_built

    def _ensure_search_indices(self) -> None:
        if self._table is None or self._indices_ready:
            return
        indexed = _indexed_columns(self._table)
        row_count = self._table.count_rows()
        build_vector_index = bool(getattr(self.search_config, "build_vector_index", True))
        has_vector_index = (
            "vector" in indexed or row_count < self.MIN_VECTOR_INDEX_ROWS or not build_vector_index
        )
        if has_vector_index and "domain" in indexed:
            self._indices_ready = True
            return
        self._create_search_indices(row_count)

    def _ensure_loaded(self) -> None:
        if self._table is None:
            self._table = self.db.open_table(self.TABLE_NAME)
        assert self._table is not None
        self._ensure_search_indices()
        if not self._chunks:
            meta_path = self.index_dir / "chunks_meta.json"
            if meta_path.is_file():
                self._chunks = json.loads(meta_path.read_text(encoding="utf-8"))
            else:
                self._chunks = self._table.to_arrow().to_pylist()
            self._chunks = [_ensure_chunk_text_fields(chunk) for chunk in self._chunks]
            self._chunk_map = None
            texts = [c.get("lexical_text", "") for c in self._chunks]
            self._bm25 = BM25Okapi([_bm25_tokens(t) for t in texts])

    def _get_domain_bm25(self, domain_filter: str) -> tuple[BM25Okapi | None, list[int]]:
        if domain_filter in self._domain_chunk_indices:
            indices = self._domain_chunk_indices[domain_filter]
            return self._bm25_by_domain.get(domain_filter), indices

        indices = [
            idx for idx, chunk in enumerate(self._chunks) if chunk.get("domain") == domain_filter
        ]
        self._domain_chunk_indices[domain_filter] = indices
        if not indices:
            return None, []

        tokens = [_bm25_tokens(self._chunks[idx].get("lexical_text", "")) for idx in indices]
        bm25 = BM25Okapi(tokens)
        self._bm25_by_domain[domain_filter] = bm25
        return bm25, indices

    def _count_domain_chunks(self, domain: str) -> int:
        return sum(1 for chunk in self._chunks if chunk.get("domain") == domain)

    def _chunk_by_id(self) -> dict[str, dict]:
        if self._chunk_map is None:
            self._chunk_map = {chunk["id"]: chunk for chunk in self._chunks}
        return self._chunk_map

    def _dense_search(
        self,
        query_vector: list[float],
        domain_filter: str | None,
        limit: int,
    ) -> list[dict]:
        assert self._table is not None
        search = self._table.search(query_vector)
        if domain_filter:
            search = search.where(_domain_where_clause(domain_filter))
        return search.limit(limit).to_list()

    def _known_domains(self) -> list[str]:
        domains = {chunk.get("domain", "") for chunk in self._chunks}
        domains.discard("")
        return sorted(domains)

    def _title_text(self, chunk: dict) -> str:
        lexical_text = chunk.get("lexical_text", "")
        if isinstance(lexical_text, str) and lexical_text:
            return lexical_text
        return _build_lexical_text(chunk)

    def _title_bonus(self, query: str, chunk: dict) -> float:
        query_tokens = _tokenize_text(query)
        if not query_tokens:
            return 0.0

        title_tokens = _tokenize_text(self._title_text(chunk))
        if not title_tokens:
            return 0.0

        matched_tokens = sum(
            1
            for query_token in query_tokens
            if any(_tokens_related(query_token, title_token) for title_token in title_tokens)
        )
        coverage = matched_tokens / len(query_tokens)

        normalized_query = _normalize_text(query)
        normalized_title = _normalize_text(" ".join(title_tokens))
        is_name_query = len(_query_content_stems(query)) <= 1
        if is_name_query:
            phrase_bonus = (
                0.03 if normalized_query and normalized_query in normalized_title else 0.0
            )
            return (coverage * 0.02) + phrase_bonus
        # For natural-language queries keep title bonus very small.
        phrase_bonus = 0.006 if normalized_query and normalized_query in normalized_title else 0.0
        return (coverage * 0.004) + phrase_bonus

    def _semantic_intent_bonus(self, query: str, chunk: dict) -> float:
        query_stems = _query_content_stems(query)
        if len(query_stems) < 2:
            return 0.0
        semantic_stems = _semantic_stem_set(chunk)
        if not semantic_stems:
            return 0.0

        matched = sum(1 for stem in query_stems if stem in semantic_stems)
        if matched < 2:
            return 0.0

        transformation_intent = _looks_like_transformation_query(query)
        kind = str(chunk.get("entity_kind", "")).lower()
        domain = str(chunk.get("domain", "")).lower()
        has_action = any(
            any(stem.startswith(action_stem) for action_stem in _ACTION_STEMS)
            for stem in semantic_stems
        )

        bonus = 0.0
        if kind in _EXECUTABLE_KINDS and has_action:
            bonus += 0.06
            if transformation_intent and domain == "platform_api":
                bonus += 0.04
        elif transformation_intent and kind in _STATIC_KINDS:
            bonus -= 0.02
        return bonus

    def _fuse_dense_bm25(
        self,
        query: str,
        query_vector: list[float],
        domain_filter: str | None,
        candidate_limit: int,
    ) -> SearchFusion:
        assert self._bm25 is not None
        dense_limit = max(self.search_config.dense_top_k, candidate_limit)
        dense_hits = self._dense_search(
            query_vector,
            domain_filter,
            dense_limit,
        )

        rrf_k = self.search_config.rrf_k
        dense_weight = float(getattr(self.search_config, "dense_rrf_weight", 1.35))
        bm25_weight = float(getattr(self.search_config, "bm25_rrf_weight", 1.0))
        dense_similarity_weight = float(getattr(self.search_config, "dense_similarity_weight", 1.0))
        scores: dict[str, float] = {}
        breakdowns: dict[str, SearchScoreBreakdown] = {}

        for rank, row in enumerate(dense_hits, 1):
            cid = row["id"]
            dense_rank_score = dense_weight / (rrf_k + rank)
            distance = float(row.get("_distance", 1.0))
            dense_similarity_score = dense_similarity_weight / (1.0 + max(distance, 0.0))
            scores[cid] = scores.get(cid, 0) + dense_rank_score + dense_similarity_score
            breakdown = breakdowns.setdefault(cid, SearchScoreBreakdown())
            breakdown.dense_rank = rank
            breakdown.dense_distance = distance
            breakdown.dense_rrf += dense_rank_score
            breakdown.dense_similarity += dense_similarity_score

        if domain_filter:
            bm25, chunk_indices = self._get_domain_bm25(domain_filter)
            if bm25 is not None:
                bm25_scores = bm25.get_scores(_bm25_tokens(query))
                bm25_ranked = sorted(
                    enumerate(bm25_scores),
                    key=lambda x: x[1],
                    reverse=True,
                )[: self.search_config.bm25_top_k]
                for rank, (idx, raw_score) in enumerate(bm25_ranked, 1):
                    chunk = self._chunks[chunk_indices[idx]]
                    cid = chunk["id"]
                    bm25_rank_score = bm25_weight / (rrf_k + rank)
                    scores[cid] = scores.get(cid, 0) + bm25_rank_score
                    breakdown = breakdowns.setdefault(cid, SearchScoreBreakdown())
                    breakdown.bm25_rank = rank
                    breakdown.bm25_raw = float(raw_score)
                    breakdown.bm25_rrf += bm25_rank_score
        else:
            bm25_scores = self._bm25.get_scores(_bm25_tokens(query))
            bm25_ranked = sorted(
                enumerate(bm25_scores),
                key=lambda x: x[1],
                reverse=True,
            )[: self.search_config.bm25_top_k]
            for rank, (idx, raw_score) in enumerate(bm25_ranked, 1):
                chunk = self._chunks[idx]
                cid = chunk["id"]
                bm25_rank_score = bm25_weight / (rrf_k + rank)
                scores[cid] = scores.get(cid, 0) + bm25_rank_score
                breakdown = breakdowns.setdefault(cid, SearchScoreBreakdown())
                breakdown.bm25_rank = rank
                breakdown.bm25_raw = float(raw_score)
                breakdown.bm25_rrf += bm25_rank_score

        natural_max = self.search_config.dense_top_k + self.search_config.bm25_top_k
        if len(scores) <= candidate_limit or len(scores) <= natural_max:
            return SearchFusion(scores=scores, breakdowns=breakdowns)
        top_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:candidate_limit]
        return SearchFusion(
            scores={cid: scores[cid] for cid in top_ids},
            breakdowns={cid: breakdowns[cid] for cid in top_ids if cid in breakdowns},
        )

    def _fuse_bm25_only(
        self,
        query: str,
        domain_filter: str,
        candidate_limit: int,
    ) -> SearchFusion:
        assert self._bm25 is not None
        rrf_k = self.search_config.rrf_k
        bm25_weight = float(getattr(self.search_config, "bm25_rrf_weight", 1.0))
        scores: dict[str, float] = {}
        breakdowns: dict[str, SearchScoreBreakdown] = {}

        bm25, chunk_indices = self._get_domain_bm25(domain_filter)
        if bm25 is not None:
            bm25_limit = max(self.search_config.bm25_top_k, candidate_limit)
            bm25_scores = bm25.get_scores(_bm25_tokens(query))
            bm25_ranked = sorted(
                enumerate(bm25_scores),
                key=lambda x: x[1],
                reverse=True,
            )[:bm25_limit]
            for rank, (idx, raw_score) in enumerate(bm25_ranked, 1):
                chunk = self._chunks[chunk_indices[idx]]
                cid = chunk["id"]
                bm25_rank_score = bm25_weight / (rrf_k + rank)
                scores[cid] = scores.get(cid, 0) + bm25_rank_score
                breakdown = breakdowns.setdefault(cid, SearchScoreBreakdown())
                breakdown.bm25_rank = rank
                breakdown.bm25_raw = float(raw_score)
                breakdown.bm25_rrf += bm25_rank_score

        if len(scores) <= candidate_limit:
            return SearchFusion(scores=scores, breakdowns=breakdowns)
        top_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:candidate_limit]
        return SearchFusion(
            scores={cid: scores[cid] for cid in top_ids},
            breakdowns={cid: breakdowns[cid] for cid in top_ids if cid in breakdowns},
        )

    def _per_domain_candidate_limit(self, domain: str, final_k: int) -> int:
        domain_size = self._count_domain_chunks(domain)
        if domain_size == 0:
            return 0
        scaled = max(final_k * 4, int(domain_size**0.5) * 2)
        minimum = max(self.search_config.dense_top_k, self.search_config.bm25_top_k)
        return int(min(domain_size, max(minimum, scaled)))

    def _all_domain_candidates(
        self,
        query: str,
        query_vector: list[float],
        final_k: int,
    ) -> SearchFusion:
        candidate_limit = max(
            final_k * 8,
            self.search_config.dense_top_k,
            self.search_config.bm25_top_k,
        )
        merged = self._fuse_dense_bm25(query, query_vector, None, candidate_limit)

        # Keep domain coverage so small domains are not drowned by the global pool.
        # Per-domain BM25 only — one global Lance dense search is enough for semantics.
        for domain in self._known_domains():
            per_domain_limit = self._per_domain_candidate_limit(domain, final_k)
            if per_domain_limit <= 0:
                continue
            domain_scores = self._fuse_bm25_only(query, domain, per_domain_limit)
            for cid, score in domain_scores.scores.items():
                prev = merged.scores.get(cid)
                if prev is None or score > prev:
                    merged.scores[cid] = score
                    if cid in domain_scores.breakdowns:
                        merged.breakdowns[cid] = domain_scores.breakdowns[cid]

        return merged

    def _apply_title_bonus(self, query: str, fusion: SearchFusion) -> SearchFusion:
        chunk_map = self._chunk_by_id()
        boosted: dict[str, float] = {}
        for cid, score in fusion.scores.items():
            chunk = chunk_map.get(cid)
            if not chunk:
                continue
            title_bonus = self._title_bonus(query, chunk)
            semantic_intent_bonus = self._semantic_intent_bonus(query, chunk)
            total = score + title_bonus + semantic_intent_bonus
            boosted[cid] = total
            breakdown = fusion.breakdowns.setdefault(cid, SearchScoreBreakdown())
            breakdown.title_bonus = title_bonus
            breakdown.semantic_intent_bonus = semantic_intent_bonus
            breakdown.total = total
        return SearchFusion(scores=boosted, breakdowns=fusion.breakdowns)

    def _match_sources(self, breakdown: SearchScoreBreakdown) -> list[str]:
        sources: list[str] = []
        if breakdown.dense_rrf > 0 or breakdown.dense_similarity > 0:
            sources.append("semantic")
        if breakdown.bm25_rrf > 0:
            sources.append("bm25")
        if breakdown.title_bonus > 0:
            sources.append("title")
        if breakdown.semantic_intent_bonus > 0:
            sources.append("intent")
        return sources

    def _match_explanation(self, breakdown: SearchScoreBreakdown) -> str:
        parts: list[str] = []
        if breakdown.dense_rank is not None:
            parts.append(f"Семантический поиск: rank {breakdown.dense_rank}")
        if breakdown.bm25_rank is not None:
            parts.append(f"BM25: rank {breakdown.bm25_rank}")
        if breakdown.title_bonus > 0:
            parts.append("совпадение с названием добавило бонус")
        if breakdown.semantic_intent_bonus > 0:
            parts.append("описание совпало с намерением запроса")
        if not parts:
            return "Результат попал в выдачу по суммарному гибридному score."
        return "; ".join(parts) + "."

    def _results_from_scores(self, query: str, fusion: SearchFusion) -> list[SearchResult]:
        scores = fusion.scores
        ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        chunk_map = self._chunk_by_id()
        results: list[SearchResult] = []
        for cid in ranked_ids:
            c = chunk_map.get(cid)
            if not c:
                continue
            title = c.get("title_ru") or c.get("title_en") or c.get("id", "")
            breakdown = fusion.breakdowns.get(cid, SearchScoreBreakdown(total=scores[cid]))
            if not breakdown.total:
                breakdown.total = scores[cid]
            semantic_terms = _collect_semantic_highlight_terms(query, c)
            semantic_text = _semantic_text_for_explanation(c)
            results.append(
                SearchResult(
                    id=cid,
                    domain=c.get("domain", ""),
                    title=title,
                    text=c.get("text", ""),
                    score=scores[cid],
                    entity_kind=c.get("entity_kind", ""),
                    html_path=c.get("html_path", ""),
                    syntax=c.get("syntax", ""),
                    highlight_terms=_collect_matched_terms(query, c.get("search_text", "")),
                    match_sources=self._match_sources(breakdown),
                    match_explanation=self._match_explanation(breakdown),
                    semantic_excerpt=_build_semantic_excerpt(semantic_text, semantic_terms),
                    semantic_highlight_terms=semantic_terms,
                    score_breakdown=breakdown.to_dict(),
                )
            )
        return results

    def search(
        self,
        query: str,
        domain: str | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        self._ensure_loaded()
        assert self._table is not None
        assert self._bm25 is not None
        final_k = limit or self.search_config.final_top_k
        domain_filter = _resolve_domain_filter(domain) if domain and domain != "all" else None

        if domain_filter and self._count_domain_chunks(domain_filter) == 0:
            return []

        query_vector = self.embedding_backend.embed_query(_semantic_query_text(query)).tolist()

        if domain_filter:
            fusion = self._fuse_dense_bm25(
                query,
                query_vector,
                domain_filter,
                max(final_k * 4, self.search_config.dense_top_k),
            )
        else:
            fusion = self._all_domain_candidates(query, query_vector, final_k)

        boosted_fusion = self._apply_title_bonus(query, fusion)
        top_ids = sorted(
            boosted_fusion.scores.keys(),
            key=lambda x: boosted_fusion.scores[x],
            reverse=True,
        )[:final_k]
        final_fusion = SearchFusion(
            scores={cid: boosted_fusion.scores[cid] for cid in top_ids},
            breakdowns={
                cid: boosted_fusion.breakdowns[cid]
                for cid in top_ids
                if cid in boosted_fusion.breakdowns
            },
        )
        return self._results_from_scores(query, final_fusion)

    def get_topic(self, topic_id: str) -> dict | None:
        self._ensure_loaded()
        for chunk in self._chunks:
            if chunk["id"] == topic_id:
                return chunk
        return None

    def stats(self) -> dict[str, int]:
        self._ensure_loaded()
        stats: dict[str, int] = {"total": len(self._chunks)}
        for chunk in self._chunks:
            domain = chunk.get("domain", "unknown")
            stats[domain] = stats.get(domain, 0) + 1
        return stats
