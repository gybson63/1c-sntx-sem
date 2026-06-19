"""Integration tests for search index."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sntx_sem.config import SearchConfig
from sntx_sem.hbk.extractor import extract_hbk
from sntx_sem.index.store import EmbeddingModel, HelpIndex

HBK_PATH = Path(__file__).resolve().parents[1] / "hbk" / "shquery_root.hbk"


@pytest.mark.skipif(not HBK_PATH.is_file(), reason="shquery_root.hbk not present")
def test_index_and_search(tmp_path: Path) -> None:
    chunks = extract_hbk(HBK_PATH, "8.3.27")
    jsonl = tmp_path / "chunks.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    embedder = EmbeddingModel("intfloat/multilingual-e5-small", "cpu")
    index = HelpIndex(tmp_path / "index", embedder, SearchConfig(final_top_k=5))
    raw = index.load_chunks_from_jsonl(jsonl)
    count = index.build(raw)
    assert count > 0

    results = index.search("left join query", domain="query", limit=5)
    assert len(results) > 0
    joined = " ".join(r.title.lower() + r.id.lower() for r in results)
    assert "join" in joined or "left" in joined or "LEFTJOIN" in joined
