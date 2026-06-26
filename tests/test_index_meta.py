"""Tests for index build metadata."""

from __future__ import annotations

from sntx_sem.index.meta import INDEX_META_FILE, load_index_meta, save_index_meta


def test_save_and_load_index_meta(tmp_path) -> None:
    index_dir = tmp_path / "index"
    save_index_meta(
        index_dir,
        indexed_count=42,
        platform_version="8.3.27",
        embedding_provider="openai_compatible",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
    )
    meta = load_index_meta(index_dir)
    assert meta["indexed_count"] == 42
    assert meta["embedding_model"] == "text-embedding-3-small"
    assert (index_dir / INDEX_META_FILE).is_file()
