"""Database readiness diagnostics and index integrity checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sntx_sem.config.embedding import index_embedding_mismatch
from sntx_sem.config.models import REQUIRED_HBK, AppConfig
from sntx_sem.config.summary import config_summary


def scan_hbk_files(hbk_dir: Path) -> dict[str, Any]:
    """Report which required HBK files are present in hbk_dir."""
    found: list[str] = []
    missing: list[str] = []
    for name in REQUIRED_HBK:
        if (hbk_dir / name).is_file():
            found.append(name)
        else:
            missing.append(name)
    return {
        "found": found,
        "missing": missing,
        "found_count": len(found),
        "required_count": len(REQUIRED_HBK),
        "ready": not missing,
    }


def count_jsonl_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def count_domain_in_jsonl(path: Path, domain: str) -> int:
    if not path.is_file():
        return 0
    marker = f'"domain": "{domain}"'
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if marker in line:
                count += 1
    return count


def count_domain_in_meta(meta_file: Path, domain: str) -> int:
    if not meta_file.is_file():
        return 0
    marker = f'"domain": "{domain}"'
    count = 0
    with meta_file.open(encoding="utf-8") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            count += block.count(marker)
    return count


def estimate_indexed_chunks(index_dir: Path, index_meta: dict[str, Any]) -> int:
    """Estimate chunk count without loading large chunks_meta.json into memory."""
    built_count = index_meta.get("indexed_count")
    if built_count is not None:
        return int(built_count)

    meta_file = index_dir / "chunks_meta.json"
    if not meta_file.is_file():
        return 0

    count = 0
    with meta_file.open(encoding="utf-8") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            count += block.count('"id":')
    return count


def collect_database_issues(
    *,
    ready: bool,
    chunks_file: Path,
    meta_file: Path,
    lance_dir: Path,
    export_count: int,
    indexed_count: int,
    embedding_mismatch: bool,
    partial_index: bool,
    config_model: str,
    index_model: str | None,
    build_vector_index: bool = True,
    vector_index_built: bool | None = None,
    lance_corrupted: bool = False,
    bsp_expected: bool = False,
    bsp_indexed_count: int = 0,
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []

    if not chunks_file.is_file():
        issues.append(
            {
                "severity": "error",
                "code": "export_missing",
                "message": (
                    "Нет export/all_chunks.jsonl — выполните «Загрузить HBK + индекс» в /admin."
                ),
            }
        )

    if export_count > 0 and not lance_dir.is_dir():
        issues.append(
            {
                "severity": "error",
                "code": "index_missing",
                "message": (
                    "Векторный индекс отсутствует или rebuild прервался. "
                    "Выполните «Пересобрать индекс» в /admin."
                ),
            }
        )

    if meta_file.is_file() and not lance_dir.is_dir():
        issues.append(
            {
                "severity": "error",
                "code": "index_broken",
                "message": (
                    "Метаданные индекса есть, но LanceDB (help_chunks.lance) удалён. "
                    "Поиск недоступен — нужен «Пересобрать индекс» в /admin."
                ),
            }
        )

    if lance_corrupted:
        issues.append(
            {
                "severity": "error",
                "code": "index_broken",
                "message": (
                    "Индекс LanceDB повреждён (неполный rebuild или отсутствуют файлы данных). "
                    "Выполните «Пересобрать индекс» в /admin. Для Docker ~4 ГБ RAM: "
                    "search.build_vector_index: false в config.yaml."
                ),
            }
        )

    if partial_index:
        issues.append(
            {
                "severity": "warning",
                "code": "partial_index",
                "message": (
                    f"Индекс неполный: {indexed_count} из {export_count} фрагментов. "
                    "Повторите «Пересобрать индекс»."
                ),
            }
        )

    if embedding_mismatch:
        issues.append(
            {
                "severity": "warning",
                "code": "embedding_mismatch",
                "message": (
                    f"Модель в config ({config_model}) не совпадает с индексом "
                    f"({index_model or 'неизвестно'}). Выполните «Пересобрать индекс»."
                ),
            }
        )

    if (
        ready
        and build_vector_index
        and indexed_count >= 256
        and lance_dir.is_dir()
        and vector_index_built is False
    ):
        issues.append(
            {
                "severity": "warning",
                "code": "vector_index_missing",
                "message": (
                    "IVF-индекс LanceDB не построен — поиск работает, но первый запрос может быть "
                    "медленнее. Увеличьте RAM и «Пересобрать индекс» или задайте "
                    "search.build_vector_index: false в config."
                ),
            }
        )

    if ready and bsp_expected and bsp_indexed_count == 0:
        issues.append(
            {
                "severity": "warning",
                "code": "bsp_not_indexed",
                "message": (
                    "BSP не проиндексирован — в индексе нет фрагментов домена bsp. "
                    "Выполните «Загрузить BSP + индекс» в /admin."
                ),
            }
        )

    if not ready and not any(
        item["code"] in {"export_missing", "index_missing", "index_broken"} for item in issues
    ):
        issues.append(
            {
                "severity": "error",
                "code": "not_ready",
                "message": (
                    "База не готова к поиску. Проверьте загрузку HBK и сборку индекса в /admin."
                ),
            }
        )

    return issues


def resolve_embedding_mismatch(
    cfg: AppConfig,
    index_meta: dict[str, Any],
    *,
    indexed_count: int,
) -> bool:
    """Shared mismatch calculation for status and settings views."""
    if indexed_count <= 0:
        return False
    if not index_meta:
        return True
    return index_embedding_mismatch(cfg, index_meta, indexed_count=indexed_count)


def is_search_ready(cfg: AppConfig) -> bool:
    """Cheap readiness gate for POST /search (no full export/meta scan)."""
    from sntx_sem.index.meta import load_index_meta

    chunks_file = cfg.export_dir / "all_chunks.jsonl"
    meta_file = cfg.index_dir / "chunks_meta.json"
    lance_dir = cfg.index_dir / "help_chunks.lance"
    if not (chunks_file.is_file() and meta_file.is_file() and lance_dir.is_dir()):
        return False

    index_meta = load_index_meta(cfg.index_dir)
    return estimate_indexed_chunks(cfg.index_dir, index_meta) > 0


def index_embedding_info(cfg: AppConfig) -> dict[str, Any]:
    """Index embedding metadata for settings views (without full DB status)."""
    from sntx_sem.index.meta import load_index_meta

    index_meta = load_index_meta(cfg.index_dir)
    indexed_count = estimate_indexed_chunks(cfg.index_dir, index_meta)
    embedding_mismatch = resolve_embedding_mismatch(cfg, index_meta, indexed_count=indexed_count)

    return {
        "embedding_mismatch": embedding_mismatch,
        "embedding_model": index_meta.get("embedding_model") if index_meta else None,
        "embedding_provider": index_meta.get("embedding_provider") if index_meta else None,
    }


def bundled_database_status(cfg: AppConfig) -> dict[str, Any]:
    """Check whether the local help database is ready."""
    from sntx_sem.index.integrity import check_lance_index
    from sntx_sem.index.meta import load_index_meta

    index_meta = load_index_meta(cfg.index_dir)
    chunks_file = cfg.export_dir / "all_chunks.jsonl"
    meta_file = cfg.index_dir / "chunks_meta.json"
    lance_dir = cfg.index_dir / "help_chunks.lance"

    lance_ok = True
    if lance_dir.is_dir():
        lance_ok, _lance_error = check_lance_index(cfg.index_dir)
    lance_corrupted = lance_dir.is_dir() and not lance_ok

    indexed_count = estimate_indexed_chunks(cfg.index_dir, index_meta)
    export_count = count_jsonl_lines(chunks_file)

    ready = (
        chunks_file.is_file()
        and meta_file.is_file()
        and lance_dir.is_dir()
        and lance_ok
        and indexed_count > 0
        and (not export_count or indexed_count >= export_count * 0.95)
    )

    embedding_mismatch = resolve_embedding_mismatch(cfg, index_meta, indexed_count=indexed_count)

    partial_index = indexed_count > 0 and export_count > 0 and indexed_count < export_count * 0.95
    config_model = cfg.embedding.model
    index_model = index_meta.get("embedding_model") if index_meta else None
    vector_index_built = index_meta.get("vector_index_built") if index_meta else None
    if vector_index_built is not None:
        vector_index_built = bool(vector_index_built)
    bsp_expected = bool(cfg.bsp.path) or cfg.bsp.enabled
    bsp_indexed_count = count_domain_in_meta(meta_file, "bsp") if meta_file.is_file() else 0
    issues = collect_database_issues(
        ready=ready,
        chunks_file=chunks_file,
        meta_file=meta_file,
        lance_dir=lance_dir,
        export_count=export_count,
        indexed_count=indexed_count,
        embedding_mismatch=embedding_mismatch,
        partial_index=partial_index,
        config_model=config_model,
        index_model=str(index_model) if index_model else None,
        build_vector_index=cfg.search.build_vector_index,
        vector_index_built=vector_index_built,
        lance_corrupted=lance_corrupted,
        bsp_expected=bsp_expected,
        bsp_indexed_count=bsp_indexed_count,
    )

    return {
        "ready": ready,
        "issues": issues,
        "config": config_summary(cfg),
        "index": {
            "platform_version": index_meta.get("platform_version", cfg.platform_version),
            "embedding_provider": index_meta.get("embedding_provider"),
            "embedding_model": index_model,
            "embedding_dimensions": index_meta.get("embedding_dimensions"),
            "embedding_mismatch": embedding_mismatch,
            "export_chunks": export_count,
            "indexed_chunks": indexed_count,
            "built_at": index_meta.get("built_at"),
            "partial_index": partial_index,
            "lance_present": lance_dir.is_dir(),
            "lance_ok": lance_ok,
            "vector_index_built": vector_index_built,
        },
        "embedding_in_sync": not embedding_mismatch,
        "paths": {
            "config_file": str(cfg.config_path) if cfg.config_path else None,
            "hbk_dir": str(cfg.hbk_dir),
            "data_dir": str(cfg.data_dir),
            "export_dir": str(cfg.export_dir),
            "index_dir": str(cfg.index_dir),
            "chunks_file": str(chunks_file),
            "build_meta": str(cfg.index_dir / "build_meta.json"),
        },
        "hbk_files": scan_hbk_files(cfg.hbk_dir),
    }
