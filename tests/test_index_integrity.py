"""Tests for LanceDB index integrity helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sntx_sem.index.integrity import (
    drop_lance_table,
    lance_table_dir,
    lance_table_present,
    path_exists_safe,
)


def test_lance_table_present_detects_lance_suffix_dir(tmp_path: Path) -> None:
    lance_table_dir(tmp_path, "help_chunks").mkdir()
    assert lance_table_present(tmp_path, "help_chunks")


def test_lance_table_present_false_when_missing(tmp_path: Path) -> None:
    assert not lance_table_present(tmp_path, "help_chunks")


def test_path_exists_safe_treats_oserror_as_present(monkeypatch: pytest.MonkeyPatch) -> None:
    path = MagicMock(spec=Path)

    def raise_oserror() -> bool:
        raise OSError(1920, "access denied")

    path.exists.side_effect = raise_oserror
    assert path_exists_safe(path)


def test_drop_lance_table_removes_lance_dir(tmp_path: Path) -> None:
    table_dir = lance_table_dir(tmp_path, "help_chunks")
    table_dir.mkdir()
    (table_dir / "data.txt").write_text("x", encoding="utf-8")

    db = MagicMock()
    drop_lance_table(db, tmp_path, "help_chunks")

    db.drop_table.assert_called_once_with("help_chunks")
    assert not table_dir.exists()


def test_drop_lance_table_removes_legacy_path(tmp_path: Path) -> None:
    legacy = tmp_path / "help_chunks"
    legacy.mkdir()
    (legacy / "old.txt").write_text("x", encoding="utf-8")

    drop_lance_table(MagicMock(), tmp_path, "help_chunks")
    assert not legacy.exists()
