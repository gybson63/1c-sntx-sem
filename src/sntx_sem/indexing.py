"""Shared ingest and index operations for CLI and API jobs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from sntx_sem.bsp.extractor import ingest_bsp
from sntx_sem.config import AppConfig, detect_platform_path, resolve_embedding_provider
from sntx_sem.embeddings import create_embedding_backend
from sntx_sem.hbk.extractor import ingest_hbk_dir
from sntx_sem.hbk.java_bridge import merge_java_export, run_java_exporter
from sntx_sem.index.meta import save_index_meta
from sntx_sem.index.store import HelpIndex

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int], None]


def _make_log_fn(log: list[str] | LogFn | None) -> LogFn:
    if log is None:
        return lambda _text: None
    if isinstance(log, list):

        def _append(text: str) -> None:
            log.append(text)

        return _append
    return log


def _hbk_not_found_message(hbk_path: Path) -> str:
    msg = f"No HBK files in {hbk_path}"
    docker_hbk = Path("/hbk")
    if hbk_path != docker_hbk and docker_hbk.is_dir() and list(docker_hbk.glob("*.hbk")):
        msg += (
            f". Files are mounted at {docker_hbk} — use hbk_dir: /hbk in config "
            "(config.docker.example.yaml) or docker-compose with /config/hbk mount"
        )
    elif not hbk_path.is_dir() or not list(hbk_path.glob("*.hbk")):
        msg += (
            ". Copy shcntx_*.hbk, shlang_*.hbk, shquery_*.hbk from "
            "1C platform bin/ into the hbk directory on the host"
        )
    return msg


def run_ingest_hbk(
    cfg: AppConfig,
    hbk_path: Path,
    platform_version: str,
    *,
    platform_path: str | Path | None = None,
    log: list[str] | LogFn | None = None,
) -> dict[str, int]:
    """Copy HBK if needed, export chunks from hbk_path."""

    _msg = _make_log_fn(log)

    if platform_path:
        src = Path(platform_path)
        hbk_path.mkdir(parents=True, exist_ok=True)
        for name in [
            "shcntx_ru.hbk",
            "shcntx_root.hbk",
            "shlang_ru.hbk",
            "shlang_root.hbk",
            "shquery_ru.hbk",
            "shquery_root.hbk",
        ]:
            src_file = src / "bin" / name if (src / "bin").is_dir() else src / name
            if src_file.is_file():
                (hbk_path / name).write_bytes(src_file.read_bytes())
                _msg(f"Copied {name}")

    if not hbk_path.is_dir() or not list(hbk_path.glob("*.hbk")):
        auto = detect_platform_path()
        if auto:
            _msg(f"Auto-detected platform: {auto}")
            return run_ingest_hbk(cfg, hbk_path, platform_version, platform_path=auto, log=log)
        msg = _hbk_not_found_message(hbk_path)
        raise FileNotFoundError(msg)

    cfg.export_dir.mkdir(parents=True, exist_ok=True)
    stats = ingest_hbk_dir(hbk_path, platform_version, cfg.export_dir)
    _msg(f"Python ingest: {stats}")

    if cfg.java_exporter.enabled:
        jar = Path(cfg.java_exporter.jar_path)
        java_chunks = run_java_exporter(jar, hbk_path, cfg.export_dir, platform_version)
        if java_chunks:
            merge_java_export(java_chunks, cfg.export_dir)
            _msg(f"Java exporter: {len(java_chunks)} chunks")
        else:
            _msg("Java exporter skipped (JAR missing or shcntx not found)")

    return stats


def run_ingest_bsp(
    cfg: AppConfig, bsp_dir: Path, log: list[str] | LogFn | None = None
) -> dict[str, Any]:
    _msg = _make_log_fn(log)

    stats = ingest_bsp(bsp_dir, cfg.export_dir)
    _msg(
        f"BSP ingest: {stats['methods']} methods from {stats['modules']} modules "
        f"(version {stats.get('bsp_version', '?')})"
    )
    return stats


def build_index(
    cfg: AppConfig,
    *,
    rebuild: bool = True,
    chunks: Path | None = None,
    domain: str | None = None,
    log: list[str] | LogFn | None = None,
    on_progress: ProgressFn | None = None,
) -> int:
    _msg = _make_log_fn(log)

    jsonl = chunks or cfg.export_dir / "all_chunks.jsonl"
    if not jsonl.is_file():
        msg = f"Chunks not found: {jsonl}. Run ingest first."
        raise FileNotFoundError(msg)

    _msg(f"Загрузка чанков из {jsonl.name}")
    backend = create_embedding_backend(cfg.embedding)
    index = HelpIndex(cfg.index_dir, backend, cfg.search)
    raw_chunks = index.load_chunks_from_jsonl(jsonl)
    if domain:
        raw_chunks = [c for c in raw_chunks if c.get("domain") == domain]
        _msg(f"Filtered to domain={domain}: {len(raw_chunks)} chunks")

    _msg(f"Индексация {len(raw_chunks)} чанков (модель: {backend.model_id})")
    count, dimensions = index.build(raw_chunks, rebuild=rebuild, on_progress=on_progress)
    provider = resolve_embedding_provider(cfg.embedding)
    save_index_meta(
        cfg.index_dir,
        indexed_count=count,
        platform_version=cfg.platform_version,
        embedding_provider=provider,
        embedding_model=backend.model_id,
        embedding_dimensions=dimensions,
    )
    _msg(f"Indexed {count} chunks -> {cfg.index_dir}")
    return count
