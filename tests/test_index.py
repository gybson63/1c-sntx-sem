"""Integration tests for search index."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sntx_sem.config import EmbeddingConfig, SearchConfig
from sntx_sem.embeddings.factory import create_embedding_backend
from sntx_sem.hbk.extractor import extract_hbk
from sntx_sem.index.store import HelpIndex

FIXTURE_JSONL = Path(__file__).resolve().parent / "fixtures" / "minimal_query_chunks.jsonl"
HBK_PATH = Path(__file__).resolve().parents[1] / "hbk" / "shquery_root.hbk"


def _build_and_search(jsonl: Path, tmp_path: Path) -> None:
    backend = create_embedding_backend(
        EmbeddingConfig(model="intfloat/multilingual-e5-small", device="cpu")
    )
    index = HelpIndex(
        tmp_path / "index",
        backend,
        SearchConfig(final_top_k=5, build_vector_index=False),
    )
    raw = index.load_chunks_from_jsonl(jsonl)
    count, _dimensions, _vector_index_built = index.build(raw)
    assert count > 0

    results = index.search("left join query", domain="query", limit=5)
    assert len(results) > 0
    joined = " ".join(r.title.lower() + r.id.lower() for r in results)
    assert "join" in joined or "left" in joined or "LEFTJOIN" in joined


def test_index_and_search_from_fixture(tmp_path: Path) -> None:
    """Lightweight integration test without HBK files (runs in CI)."""
    _build_and_search(FIXTURE_JSONL, tmp_path)


@pytest.mark.nightly
@pytest.mark.skipif(not HBK_PATH.is_file(), reason="shquery_root.hbk not present")
def test_index_and_search_from_hbk(tmp_path: Path) -> None:
    """Full HBK extraction + index pipeline (nightly when HBK is available)."""
    chunks = extract_hbk(HBK_PATH, "8.3.27")
    jsonl = tmp_path / "chunks.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    _build_and_search(jsonl, tmp_path)
