"""Единый источник версии пакета (читается из pyproject.toml / metadata)."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _read_pyproject_version() -> str:
    root = Path(__file__).resolve().parent.parent.parent
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("version ="):
            return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    msg = "version not found in pyproject.toml"
    raise RuntimeError(msg)


def get_version() -> str:
    try:
        return version("sntx-sem")
    except PackageNotFoundError:
        return _read_pyproject_version()


__version__ = get_version()
