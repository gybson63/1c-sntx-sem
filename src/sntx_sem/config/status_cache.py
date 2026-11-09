"""Cached database status for API endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sntx_sem.config.diagnostics import bundled_database_status
from sntx_sem.config.models import AppConfig


def _path_fingerprint(path: Path) -> tuple[str, int | None, int | None]:
    if path.is_file() or path.is_dir():
        stat = path.stat()
        return (str(path), stat.st_mtime_ns, stat.st_size)
    return (str(path), None, None)


def status_cache_fingerprint(cfg: AppConfig) -> tuple[Any, ...]:
    """Inputs that affect bundled_database_status() output."""
    return (
        _path_fingerprint(cfg.export_dir / "all_chunks.jsonl"),
        _path_fingerprint(cfg.index_dir / "chunks_meta.json"),
        _path_fingerprint(cfg.index_dir / "build_meta.json"),
        _path_fingerprint(cfg.index_dir / "help_chunks.lance"),
        cfg.embedding.model,
        cfg.embedding.provider or "",
        str(cfg.bsp.path or ""),
        cfg.bsp.enabled,
    )


@dataclass
class DatabaseStatusCache:
    _status: dict[str, Any] | None = field(default=None, repr=False)
    _fingerprint: tuple[Any, ...] | None = field(default=None, repr=False)

    def get(self, cfg: AppConfig, *, force: bool = False) -> dict[str, Any]:
        fingerprint = status_cache_fingerprint(cfg)
        if not force and self._status is not None and self._fingerprint == fingerprint:
            return self._status
        status = bundled_database_status(cfg)
        self._status = status
        self._fingerprint = fingerprint
        return status

    def invalidate(self) -> None:
        self._status = None
        self._fingerprint = None
