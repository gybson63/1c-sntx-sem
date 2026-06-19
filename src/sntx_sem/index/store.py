"""Vector index and hybrid search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa
from rank_bm25 import BM25Okapi


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


class EmbeddingModel:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        model = self._load()
        prefixed = [f"passage: {t[:2000]}" for t in texts]
        vectors = model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
            batch_size=batch_size,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        model = self._load()
        vector = model.encode(f"query: {query}", normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vector, dtype=np.float32)


class HelpIndex:
    TABLE_NAME = "help_chunks"

    def __init__(
        self,
        index_dir: Path,
        embedding_model: EmbeddingModel,
        search_config: Any,
    ) -> None:
        self.index_dir = index_dir
        self.embedding_model = embedding_model
        self.search_config = search_config
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.index_dir))
        self._chunks: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._table = None

    def load_chunks_from_jsonl(self, jsonl_path: Path) -> list[dict]:
        chunks: list[dict] = []
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        return chunks

    def build(self, chunks: list[dict], rebuild: bool = True, batch_size: int = 256) -> int:
        if not chunks:
            return 0

        table_path = self.index_dir / self.TABLE_NAME
        if rebuild and table_path.exists():
            self.db.drop_table(self.TABLE_NAME)

        all_rows: list[dict] = []
        texts_for_bm25: list[str] = []

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [
                f"{c.get('title_ru', '')} {c.get('title_en', '')} {c.get('text', '')}".strip()
                for c in batch
            ]
            vectors = self.embedding_model.embed_passages(texts, batch_size=64)
            for chunk, vector, text in zip(batch, vectors, texts, strict=True):
                all_rows.append(
                    {
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
        self._chunks = all_rows
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
                    }
                    for r in all_rows
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return len(all_rows)

    def _ensure_loaded(self) -> None:
        if self._table is None:
            self._table = self.db.open_table(self.TABLE_NAME)
        assert self._table is not None
        if not self._chunks:
            meta_path = self.index_dir / "chunks_meta.json"
            if meta_path.is_file():
                self._chunks = json.loads(meta_path.read_text(encoding="utf-8"))
            else:
                self._chunks = self._table.to_arrow().to_pylist()
            texts = [c.get("search_text", "") for c in self._chunks]
            self._bm25 = BM25Okapi([t.lower().split() for t in texts])

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

        query_vector = self.embedding_model.embed_query(query)
        dense_hits = (
            self._table.search(query_vector.tolist())
            .limit(self.search_config.dense_top_k)
            .to_list()
        )

        if domain and domain != "all":
            domain_filter = domain
            if domain == "bsl":
                domain_filter = "bsl_lang"
            elif domain == "query":
                domain_filter = "query_lang"
            dense_hits = [h for h in dense_hits if h.get("domain") == domain_filter]

        bm25_scores = self._bm25.get_scores(query.lower().split())
        bm25_ranked = sorted(
            enumerate(bm25_scores),
            key=lambda x: x[1],
            reverse=True,
        )[: self.search_config.bm25_top_k]

        rrf_k = self.search_config.rrf_k
        scores: dict[str, float] = {}

        for rank, row in enumerate(dense_hits, 1):
            if domain and domain != "all":
                dfilter = domain
                if domain == "bsl":
                    dfilter = "bsl_lang"
                elif domain == "query":
                    dfilter = "query_lang"
                if row.get("domain") != dfilter:
                    continue
            scores[row["id"]] = scores.get(row["id"], 0) + 1 / (rrf_k + rank)

        for rank, (idx, _) in enumerate(bm25_ranked, 1):
            chunk = self._chunks[idx]
            if domain and domain != "all":
                dfilter = domain
                if domain == "bsl":
                    dfilter = "bsl_lang"
                elif domain == "query":
                    dfilter = "query_lang"
                if chunk.get("domain") != dfilter:
                    continue
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
