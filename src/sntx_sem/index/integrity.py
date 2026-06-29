"""LanceDB index integrity checks and filesystem helpers."""

from __future__ import annotations

import contextlib
import shutil
from pathlib import Path
from typing import Any

import lancedb

_LANCE_TABLE = "help_chunks"


def lance_table_dir(index_dir: Path, table_name: str) -> Path:
    return index_dir / f"{table_name}.lance"


def path_exists_safe(path: Path) -> bool:
    """Return True if path exists; treat broken reparse points as present (needs cleanup)."""
    try:
        return path.exists()
    except OSError:
        return True


def lance_table_present(index_dir: Path, table_name: str) -> bool:
    """True when Lance data dir or a legacy table path artifact is present."""
    if lance_table_dir(index_dir, table_name).is_dir():
        return True
    legacy = index_dir / table_name
    if legacy == lance_table_dir(index_dir, table_name):
        return False
    return path_exists_safe(legacy)


def _remove_path_safe(path: Path) -> None:
    import subprocess
    import sys

    if not path_exists_safe(path):
        return
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
        return
    except OSError:
        pass

    if sys.platform == "win32":
        subprocess.run(
            ["cmd", "/c", "rmdir", str(path)],
            check=False,
            capture_output=True,
        )
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
        return

    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)
    with contextlib.suppress(OSError):
        if path.is_dir():
            shutil.rmtree(path)


def drop_lance_table(db: Any, index_dir: Path, table_name: str) -> None:
    """Drop Lance table via API and remove on-disk artifacts (incl. broken Windows junctions)."""
    with contextlib.suppress(Exception):
        db.drop_table(table_name)
    _remove_path_safe(lance_table_dir(index_dir, table_name))
    legacy = index_dir / table_name
    if legacy != lance_table_dir(index_dir, table_name):
        _remove_path_safe(legacy)


def check_lance_index(index_dir: Path) -> tuple[bool, str | None]:
    """Verify Lance table opens and row count is readable."""
    lance_dir = lance_table_dir(index_dir, _LANCE_TABLE)
    if not lance_dir.is_dir():
        return False, None
    try:
        db = lancedb.connect(str(index_dir.resolve()))
        table = db.open_table(_LANCE_TABLE)
        table.count_rows()
        return True, None
    except Exception as exc:
        return False, str(exc).strip() or exc.__class__.__name__
