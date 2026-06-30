"""Index build metadata stored alongside LanceDB."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, cast

INDEX_META_FILE = "build_meta.json"


def load_index_meta(index_dir: Path) -> dict[str, Any]:
    path = index_dir / INDEX_META_FILE
    if not path.is_file():
        return {}
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def save_index_meta(
    index_dir: Path,
    *,
    indexed_count: int,
    platform_version: str,
    embedding_provider: str,
    embedding_model: str,
    embedding_dimensions: int | None = None,
    vector_index_built: bool | None = None,
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "indexed_count": indexed_count,
        "built_at": date.today().isoformat(),
        "platform_version": platform_version,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_dimensions": embedding_dimensions,
    }
    if vector_index_built is not None:
        meta["vector_index_built"] = vector_index_built
    path = index_dir / INDEX_META_FILE
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
