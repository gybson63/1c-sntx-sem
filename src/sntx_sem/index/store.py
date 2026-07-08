"""Vector index and hybrid search."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
from rank_bm25 import BM25Okapi

from sntx_sem.embeddings.base import EmbeddingBackend
from sntx_sem.index.chunk_text import (
    build_lexical_text,
    build_search_text,
    build_semantic_text,
    chunk_meta_row,
    ensure_chunk_text_fields,
    write_chunks_meta_json,
)
from sntx_sem.index.explain import SearchResult, build_search_results, collect_matched_terms
from sntx_sem.index.integrity import drop_lance_table, lance_table_present
from sntx_sem.index.ranking import (
    RankerContext,
    SearchFusion,
    all_domain_candidates,
    apply_title_bonus,
    fuse_dense_bm25,
)
from sntx_sem.index.storage import (
    domain_where_clause,
    indexed_columns,
    resolve_domain_filter,
    vector_index_partitions,
)
from sntx_sem.index.tokens import (
    bm25_tokens,
    semantic_query_text,
    tokens_related,
)
from sntx_sem.index.tokens import split_identifier as split_identifier  # noqa: F401

# Backward-compatible re-exports for tests and external callers.
_build_lexical_text = build_lexical_text
_build_search_text = build_search_text
_build_semantic_text = build_semantic_text
_chunk_meta_row = chunk_meta_row
_collect_matched_terms = collect_matched_terms
_domain_where_clause = domain_where_clause
_tokens_related = tokens_related
_vector_index_partitions = vector_index_partitions


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
            semantic_texts = [build_semantic_text(c) for c in batch]
            lexical_texts = [build_lexical_text(c) for c in batch]
            search_texts = [
                build_search_text(c, lexical_text=lexical, semantic_text=semantic)
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
                    chunk_meta_row(
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
        self._chunks = [ensure_chunk_text_fields(row) for row in meta_rows]
        self._chunk_map = None
        self._bm25 = BM25Okapi([bm25_tokens(r.get("lexical_text", "")) for r in self._chunks])
        write_chunks_meta_json(self.index_dir / "chunks_meta.json", meta_rows)
        return len(meta_rows), dimensions, vector_index_built

    def _create_search_indices(
        self,
        row_count: int,
        on_log: Callable[[str], None] | None = None,
    ) -> bool:
        if self._table is None or row_count <= 0:
            return False
        indexed = indexed_columns(self._table)
        build_vector_index = bool(getattr(self.search_config, "build_vector_index", True))
        vector_index_built = "vector" in indexed

        if (
            build_vector_index
            and not vector_index_built
            and row_count >= self.MIN_VECTOR_INDEX_ROWS
        ):
            partitions = vector_index_partitions(row_count)
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
        indexed = indexed_columns(self._table)
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
            self._chunks = [ensure_chunk_text_fields(chunk) for chunk in self._chunks]
            self._chunk_map = None
            texts = [c.get("lexical_text", "") for c in self._chunks]
            self._bm25 = BM25Okapi([bm25_tokens(t) for t in texts])

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

        tokens = [bm25_tokens(self._chunks[idx].get("lexical_text", "")) for idx in indices]
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
            search = search.where(domain_where_clause(domain_filter))
        return search.limit(limit).to_list()

    def _known_domains(self) -> list[str]:
        domains = {chunk.get("domain", "") for chunk in self._chunks}
        domains.discard("")
        return sorted(domains)

    def _ranker_context(self) -> RankerContext:
        assert self._bm25 is not None
        return RankerContext(
            search_config=self.search_config,
            chunks=self._chunks,
            bm25=self._bm25,
            get_domain_bm25=self._get_domain_bm25,
            dense_search=self._dense_search,
            known_domains=self._known_domains,
            count_domain_chunks=self._count_domain_chunks,
        )

    def search(
        self,
        query: str,
        domain: str | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        self._ensure_loaded()
        assert self._table is not None
        final_k = limit or self.search_config.final_top_k
        domain_filter = resolve_domain_filter(domain) if domain and domain != "all" else None

        if domain_filter and self._count_domain_chunks(domain_filter) == 0:
            return []

        query_vector = self.embedding_backend.embed_query(semantic_query_text(query)).tolist()
        ctx = self._ranker_context()

        if domain_filter:
            fusion = fuse_dense_bm25(
                ctx,
                query,
                query_vector,
                domain_filter,
                max(final_k * 4, self.search_config.dense_top_k),
            )
        else:
            fusion = all_domain_candidates(ctx, query, query_vector, final_k)

        boosted_fusion = apply_title_bonus(query, fusion, self._chunk_by_id())
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
        return build_search_results(
            query,
            final_fusion.scores,
            final_fusion.breakdowns,
            self._chunk_by_id(),
        )

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
