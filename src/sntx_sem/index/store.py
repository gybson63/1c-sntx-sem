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


def _build_search_text(chunk: dict) -> str:
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
    parts.append(chunk.get("text", ""))
    return " ".join(p for p in parts if p).strip()


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
        self.db = lancedb.connect(str(self.index_dir))
        self._chunks: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._bm25_by_domain: dict[str, BM25Okapi] = {}
        self._domain_chunk_indices: dict[str, list[int]] = {}
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
    ) -> tuple[int, int | None]:
        if not chunks:
            return 0, None

        total = len(chunks)
        table_path = self.index_dir / self.TABLE_NAME
        if rebuild and table_path.exists():
            self.db.drop_table(self.TABLE_NAME)

        meta_rows: list[dict] = []
        texts_for_bm25: list[str] = []
        dimensions: int | None = None
        schema: pa.Schema | None = None
        row_count = 0

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [_build_search_text(c) for c in batch]
            vectors = self.embedding_backend.embed_passages(texts)
            if dimensions is None and len(vectors) > 0:
                dimensions = int(vectors.shape[1])
            batch_rows: list[dict] = []
            for chunk, vector, text in zip(batch, vectors, texts, strict=True):
                row = {
                    "id": chunk["id"],
                    "domain": chunk.get("domain", ""),
                    "title_ru": chunk.get("title_ru", ""),
                    "title_en": chunk.get("title_en", ""),
                    "entity_kind": chunk.get("entity_kind", ""),
                    "html_path": chunk.get("html_path", ""),
                    "syntax": chunk.get("syntax", ""),
                    "text": chunk.get("text", ""),
                    "search_text": text,
                    "vector": vector.tolist(),
                }
                batch_rows.append(row)
                meta_rows.append(
                    {
                        **row,
                        "parameters": chunk.get("parameters", ""),
                        "signature": chunk.get("signature", ""),
                    }
                )
            texts_for_bm25.extend(texts)

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

        assert self._table is not None
        self._create_search_indices(row_count)
        self._chunks = meta_rows
        self._bm25 = BM25Okapi([t.lower().split() for t in texts_for_bm25])
        meta_path = self.index_dir / "chunks_meta.json"
        meta_path.write_text(
            json.dumps(
                [
                    {
                        "id": r["id"],
                        "domain": r["domain"],
                        "title_ru": r["title_ru"],
                        "title_en": r["title_en"],
                        "search_text": r["search_text"],
                        "entity_kind": r["entity_kind"],
                        "html_path": r["html_path"],
                        "syntax": r["syntax"],
                        "text": r["text"],
                        "parameters": r.get("parameters", ""),
                        "signature": r.get("signature", ""),
                    }
                    for r in meta_rows
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return len(meta_rows), dimensions

    def _create_search_indices(self, row_count: int) -> None:
        if self._table is None or row_count <= 0:
            return
        indexed = _indexed_columns(self._table)
        if "vector" not in indexed and row_count >= self.MIN_VECTOR_INDEX_ROWS:
            self._table.create_index(
                metric="cosine",
                num_partitions=_vector_index_partitions(row_count),
                vector_column_name="vector",
            )
        if "domain" not in indexed:
            self._table.create_scalar_index("domain")
        self._indices_ready = True

    def _ensure_search_indices(self) -> None:
        if self._table is None or self._indices_ready:
            return
        indexed = _indexed_columns(self._table)
        row_count = self._table.count_rows()
        has_vector_index = "vector" in indexed or row_count < self.MIN_VECTOR_INDEX_ROWS
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
            texts = [c.get("search_text", "") for c in self._chunks]
            self._bm25 = BM25Okapi([t.lower().split() for t in texts])

    def _get_domain_bm25(self, domain_filter: str) -> tuple[BM25Okapi, list[int]]:
        cached = self._bm25_by_domain.get(domain_filter)
        indices = self._domain_chunk_indices.get(domain_filter)
        if cached is not None and indices is not None:
            return cached, indices

        indices = [
            idx for idx, chunk in enumerate(self._chunks) if chunk.get("domain") == domain_filter
        ]
        tokens = [self._chunks[idx].get("search_text", "").lower().split() for idx in indices]
        bm25 = BM25Okapi(tokens)
        self._bm25_by_domain[domain_filter] = bm25
        self._domain_chunk_indices[domain_filter] = indices
        return bm25, indices

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
        parts: list[str] = []
        for key in ("title_ru", "title_en"):
            title = chunk.get(key, "")
            if title:
                parts.append(title)
                parts.append(split_identifier(title))
        return " ".join(parts)

    def _title_bonus(self, query: str, chunk: dict) -> float:
        query_tokens = _tokenize_text(query)
        if not query_tokens:
            return 0.0

        title_tokens = _tokenize_text(self._title_text(chunk))
        if not title_tokens:
            return 0.0

        title_token_set = set(title_tokens)
        matched_tokens = sum(1 for token in query_tokens if token in title_token_set)
        coverage = matched_tokens / len(query_tokens)

        normalized_query = _normalize_text(query)
        normalized_title = _normalize_text(" ".join(title_tokens))
        phrase_bonus = 0.0
        if normalized_query and normalized_query in normalized_title:
            phrase_bonus = 0.03

        return (coverage * 0.02) + phrase_bonus

    def _fuse_dense_bm25(
        self,
        query: str,
        query_vector: list[float],
        domain_filter: str | None,
        candidate_limit: int,
    ) -> dict[str, float]:
        assert self._bm25 is not None
        dense_hits = self._dense_search(
            query_vector,
            domain_filter,
            self.search_config.dense_top_k,
        )

        rrf_k = self.search_config.rrf_k
        scores: dict[str, float] = {}

        for rank, row in enumerate(dense_hits, 1):
            scores[row["id"]] = scores.get(row["id"], 0) + 1 / (rrf_k + rank)

        if domain_filter:
            bm25, chunk_indices = self._get_domain_bm25(domain_filter)
            bm25_scores = bm25.get_scores(query.lower().split())
            bm25_ranked = sorted(
                enumerate(bm25_scores),
                key=lambda x: x[1],
                reverse=True,
            )[: self.search_config.bm25_top_k]
            for rank, (idx, _) in enumerate(bm25_ranked, 1):
                chunk = self._chunks[chunk_indices[idx]]
                cid = chunk["id"]
                scores[cid] = scores.get(cid, 0) + 1 / (rrf_k + rank)
        else:
            bm25_scores = self._bm25.get_scores(query.lower().split())
            bm25_ranked = sorted(
                enumerate(bm25_scores),
                key=lambda x: x[1],
                reverse=True,
            )[: self.search_config.bm25_top_k]
            for rank, (idx, _) in enumerate(bm25_ranked, 1):
                chunk = self._chunks[idx]
                cid = chunk["id"]
                scores[cid] = scores.get(cid, 0) + 1 / (rrf_k + rank)

        if len(scores) <= candidate_limit:
            return scores
        top_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:candidate_limit]
        return {cid: scores[cid] for cid in top_ids}

    def _all_domain_candidates(
        self,
        query: str,
        query_vector: list[float],
        final_k: int,
    ) -> dict[str, float]:
        candidate_limit = max(
            final_k * 8,
            self.search_config.dense_top_k,
            self.search_config.bm25_top_k,
        )
        merged = self._fuse_dense_bm25(query, query_vector, None, candidate_limit)

        # Keep domain coverage so small domains are not drowned by the global pool.
        per_domain_limit = max(final_k, 2)
        for domain in self._known_domains():
            domain_scores = self._fuse_dense_bm25(
                query,
                query_vector,
                domain,
                per_domain_limit,
            )
            for cid, score in domain_scores.items():
                prev = merged.get(cid)
                if prev is None or score > prev:
                    merged[cid] = score

        return merged

    def _apply_title_bonus(self, query: str, scores: dict[str, float]) -> dict[str, float]:
        chunk_map = {c["id"]: c for c in self._chunks}
        boosted: dict[str, float] = {}
        for cid, score in scores.items():
            chunk = chunk_map.get(cid)
            if not chunk:
                continue
            boosted[cid] = score + self._title_bonus(query, chunk)
        return boosted

    def _results_from_scores(self, query: str, scores: dict[str, float]) -> list[SearchResult]:
        ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        chunk_map = {c["id"]: c for c in self._chunks}
        results: list[SearchResult] = []
        for cid in ranked_ids:
            c = chunk_map.get(cid)
            if not c:
                continue
            title = c.get("title_ru") or c.get("title_en") or c.get("id", "")
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

        query_vector = self.embedding_backend.embed_query(query).tolist()

        if domain_filter:
            scores = self._fuse_dense_bm25(
                query,
                query_vector,
                domain_filter,
                max(final_k * 4, self.search_config.dense_top_k),
            )
        else:
            scores = self._all_domain_candidates(query, query_vector, final_k)

        boosted_scores = self._apply_title_bonus(query, scores)
        top_ids = sorted(
            boosted_scores.keys(),
            key=lambda x: boosted_scores[x],
            reverse=True,
        )[:final_k]
        final_scores = {cid: boosted_scores[cid] for cid in top_ids}
        return self._results_from_scores(query, final_scores)

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
