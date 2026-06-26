"""Vector index and hybrid search."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
from rank_bm25 import BM25Okapi

from sntx_sem.embeddings.base import EmbeddingBackend

_DOMAIN_FILTER_RE = re.compile(r"^[\w]+$")


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
    parts = [chunk.get("title_ru", ""), chunk.get("title_en", "")]
    if chunk.get("domain") == "bsp":
        parts.append(chunk.get("path", ""))
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
        self, chunks: list[dict], rebuild: bool = True, batch_size: int = 256
    ) -> tuple[int, int | None]:
        if not chunks:
            return 0, None

        table_path = self.index_dir / self.TABLE_NAME
        if rebuild and table_path.exists():
            self.db.drop_table(self.TABLE_NAME)

        all_rows: list[dict] = []
        meta_rows: list[dict] = []
        texts_for_bm25: list[str] = []
        dimensions: int | None = None

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [_build_search_text(c) for c in batch]
            vectors = self.embedding_backend.embed_passages(texts, batch_size=64)
            if dimensions is None and len(vectors) > 0:
                dimensions = int(vectors.shape[1])
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
                all_rows.append(row)
                meta_rows.append(
                    {
                        **row,
                        "parameters": chunk.get("parameters", ""),
                        "signature": chunk.get("signature", ""),
                    }
                )
            texts_for_bm25.extend(texts)

        dim = len(all_rows[0]["vector"])
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
            data=all_rows,
            schema=schema,
            mode="overwrite",
        )
        self._create_search_indices(len(all_rows))
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
        return len(all_rows), dimensions

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

        query_vector = self.embedding_backend.embed_query(query)
        dense_hits = self._dense_search(
            query_vector.tolist(),
            domain_filter,
            self.search_config.dense_top_k,
        )

        if domain_filter:
            bm25, chunk_indices = self._get_domain_bm25(domain_filter)
            bm25_scores = bm25.get_scores(query.lower().split())
            bm25_ranked = sorted(
                enumerate(bm25_scores),
                key=lambda x: x[1],
                reverse=True,
            )[: self.search_config.bm25_top_k]
        else:
            bm25_scores = self._bm25.get_scores(query.lower().split())
            bm25_ranked = sorted(
                enumerate(bm25_scores),
                key=lambda x: x[1],
                reverse=True,
            )[: self.search_config.bm25_top_k]

        rrf_k = self.search_config.rrf_k
        scores: dict[str, float] = {}

        for rank, row in enumerate(dense_hits, 1):
            scores[row["id"]] = scores.get(row["id"], 0) + 1 / (rrf_k + rank)

        for rank, (idx, _) in enumerate(bm25_ranked, 1):
            chunk = self._chunks[chunk_indices[idx]] if domain_filter else self._chunks[idx]
            cid = chunk["id"]
            scores[cid] = scores.get(cid, 0) + 1 / (rrf_k + rank)

        ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:final_k]
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
                )
            )
        return results

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
