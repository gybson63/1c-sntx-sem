"""Tests for HBK parsing and ingest."""

from __future__ import annotations

from pathlib import Path

import pytest

from sntx_sem.hbk.container import HbkReader, inflate_pack_block, open_file_storage
from sntx_sem.hbk.extractor import extract_hbk, ingest_hbk_dir
from sntx_sem.hbk.toc_parser import _extract_meta, flatten_toc, parse_toc

HBK_PATH = Path(__file__).resolve().parents[1] / "hbk" / "shquery_root.hbk"
SHCNTX_RU_PATH = Path(__file__).resolve().parents[1] / "hbk" / "shcntx_ru.hbk"


def test_extract_meta_locale_tagged_titles() -> None:
    meta = [
        [
            1,
            1,
            [
                1,
                2,
                ["ru", "СтрРазделить"],
                ["en", "StrSplit"],
            ],
            "/objects/Global context/methods/catalog4838/StrSplit4532.html",
        ]
    ]
    title_ru, title_en, html_path = _extract_meta(meta)
    assert title_ru == "СтрРазделить"
    assert title_en == "StrSplit"
    assert html_path.endswith("StrSplit4532.html")


def test_extract_meta_legacy_bilingual_pair() -> None:
    meta = [[1, 2, ["Выборка", "Selection"]], "/root.html"]
    title_ru, title_en, html_path = _extract_meta(meta)
    assert title_ru == "Выборка"
    assert title_en == "Selection"
    assert html_path == "/root.html"


@pytest.mark.skipif(not SHCNTX_RU_PATH.is_file(), reason="shcntx_ru.hbk not present")
def test_shcntx_strsplit_title_from_hbk() -> None:
    chunks = extract_hbk(SHCNTX_RU_PATH, "8.3.27")
    chunk = next(c for c in chunks if "StrSplit4532" in (c.html_path or c.id))
    assert chunk.title_ru == "СтрРазделить"
    assert chunk.title_en == "StrSplit"


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
