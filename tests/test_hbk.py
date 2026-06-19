"""Tests for HBK parsing and ingest."""

from __future__ import annotations

from pathlib import Path

import pytest

from sntx_sem.hbk.container import HbkReader, inflate_pack_block, open_file_storage
from sntx_sem.hbk.extractor import extract_hbk, ingest_hbk_dir
from sntx_sem.hbk.toc_parser import flatten_toc, parse_toc

HBK_PATH = Path(__file__).resolve().parents[1] / "hbk" / "shquery_root.hbk"


@pytest.mark.skipif(not HBK_PATH.is_file(), reason="shquery_root.hbk not present")
class TestHbkParsing:
    def test_entities_extracted(self) -> None:
        reader = HbkReader.from_path(HBK_PATH)
        assert "PackBlock" in reader.entities
        assert "FileStorage" in reader.entities

    def test_toc_not_empty(self) -> None:
        reader = HbkReader.from_path(HBK_PATH)
        toc = inflate_pack_block(reader.get_entity("PackBlock")).decode("utf-8")
        pages = flatten_toc(parse_toc(toc))
        assert len(pages) > 50

    def test_sample_topics(self) -> None:
        reader = HbkReader.from_path(HBK_PATH)
        toc = inflate_pack_block(reader.get_entity("PackBlock")).decode("utf-8")
        pages = flatten_toc(parse_toc(toc))
        titles = " ".join(p.title_en.lower() for p in pages)
        assert "select" in titles or "query" in titles

    def test_chunks_extracted(self) -> None:
        chunks = extract_hbk(HBK_PATH, "8.3.27")
        assert len(chunks) > 50
        assert all(c.domain == "query_lang" for c in chunks)

    def test_html_readable(self) -> None:
        reader = HbkReader.from_path(HBK_PATH)
        with open_file_storage(reader.get_entity("FileStorage")) as zf:
            assert "root.html" in zf.namelist()


@pytest.mark.skipif(not HBK_PATH.is_file(), reason="shquery_root.hbk not present")
def test_ingest_pipeline(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    stats = ingest_hbk_dir(HBK_PATH.parent, "8.3.27", export_dir)
    assert stats.get("merged_total", 0) > 0
    assert (export_dir / "all_chunks.jsonl").is_file()
