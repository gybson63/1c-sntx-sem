"""CLI entry point."""

from __future__ import annotations

import json
import os
from pathlib import Path

import click

from sntx_sem.bsp.extractor import ingest_bsp
from sntx_sem.config import (
    LOCAL_EMBEDDING_PROVIDERS,
    AppConfig,
    align_embedding_with_index,
    load_config,
)
from sntx_sem.embeddings import create_embedding_backend
from sntx_sem.examples.linker import link_examples_batch
from sntx_sem.examples.scanner import (
    export_examples_jsonl,
    keyword_link_examples,
    scan_config_path,
)
from sntx_sem.examples.store import ExamplesStore
from sntx_sem.index.store import HelpIndex
from sntx_sem.indexing import build_index, run_ingest_hbk


@click.group()
def main() -> None:
    """1C syntax help semantic search toolkit."""


def _echo_logs(logs: list[str]) -> None:
    for line in logs:
        click.echo(line)


def _build_index(
    cfg: AppConfig,
    *,
    rebuild: bool = True,
    chunks: Path | None = None,
    domain: str | None = None,
) -> int:
    logs: list[str] = []
    try:
        count = build_index(cfg, rebuild=rebuild, chunks=chunks, domain=domain, log=logs)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    _echo_logs(logs)
    return count


def _run_ingest(
    hbk_path: Path,
    version: str,
    platform_path: str | None,
    *,
    build_index_flag: bool = True,
) -> None:
    cfg = load_config()
    logs: list[str] = []
    try:
        run_ingest_hbk(
            cfg,
            hbk_path,
            version,
            platform_path=platform_path,
            log=logs,
        )
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    _echo_logs(logs)
    if build_index_flag:
        _build_index(cfg)


@main.command("ingest")
@click.option("--hbk-dir", type=click.Path(exists=True), default=None)
@click.option("--platform-version", default=None)
@click.option("--platform-path", type=click.Path(exists=True), default=None)
@click.option("--no-index", is_flag=True, help="Skip vector index build after export")
def ingest_cmd(
    hbk_dir: str | None,
    platform_version: str | None,
    platform_path: str | None,
    no_index: bool,
) -> None:
    """Extract help chunks from HBK files and build vector index."""
    cfg = load_config()
    hbk_path = Path(hbk_dir) if hbk_dir else cfg.hbk_dir
    version = platform_version or cfg.platform_version
    _run_ingest(hbk_path, version, platform_path, build_index_flag=not no_index)


@main.command("ingest-bsp")
@click.option("--bsp-dir", type=click.Path(exists=True), default=None)
@click.option("--no-index", is_flag=True, help="Skip vector index build after export")
def ingest_bsp_cmd(bsp_dir: str | None, no_index: bool) -> None:
    """Extract BSP public API from CommonModules export and rebuild index."""
    cfg = load_config()
    if bsp_dir:
        path = Path(bsp_dir)
    elif cfg.bsp.path:
        path = Path(cfg.bsp.path)
    else:
        raise click.ClickException(
            "BSP path not configured. Set bsp.path in config.yaml or pass --bsp-dir."
        )

    stats = ingest_bsp(path, cfg.export_dir)
    click.echo(
        f"BSP ingest: {stats['methods']} methods from {stats['modules']} modules "
        f"(version {stats.get('bsp_version', '?')}, "
        f"{stats['without_comment']} without comment)"
    )
    click.echo(f"Saved -> {cfg.export_dir / 'bsp_api.jsonl'}")
    click.echo(f"Merged {stats['merged_total']} chunks -> {cfg.export_dir / 'all_chunks.jsonl'}")

    if no_index:
        click.echo("Skipped index build (--no-index). Run: python -m sntx_sem index --rebuild")
    else:
        _build_index(cfg)


@main.command("index")
@click.option("--rebuild/--no-rebuild", default=True)
@click.option("--chunks", type=click.Path(exists=True), default=None)
@click.option("--domain", default=None, help="Index only chunks from domain (e.g. query_lang)")
def index_cmd(rebuild: bool, chunks: str | None, domain: str | None) -> None:
    """Build vector index from exported chunks."""
    cfg = load_config()
    chunks_path = Path(chunks) if chunks else None
    _build_index(cfg, rebuild=rebuild, chunks=chunks_path, domain=domain)


@main.command("status")
def status_cmd() -> None:
    """Show local database readiness."""
    from sntx_sem.config import bundled_database_status

    cfg = load_config()
    status = bundled_database_status(cfg)
    click.echo(json.dumps(status, ensure_ascii=False, indent=2))
    click.echo("")
    click.echo("config — настройки из config.yaml (используются при search и index)")
    click.echo("index  — параметры последней сборки (data/index/build_meta.json)")

    if status["ready"]:
        click.echo("Database is ready — connect MCP and search.")
    elif status.get("index", {}).get("partial_index"):
        idx = status["index"]
        click.echo(
            f"Partial index ({idx['indexed_chunks']}/{idx['export_chunks']}). "
            "Run: python -m sntx_sem index --rebuild"
        )
    else:
        click.echo(
            "Database not ready. Build locally from your 1C platform install:\n"
            '  python -m sntx_sem ingest --platform-path "C:/Program Files/1cv8/.../bin"'
        )

    index_info = status.get("index", {})
    if index_info.get("embedding_mismatch"):
        cfg_emb = status.get("config", {}).get("embedding", {})
        idx_model = index_info.get("embedding_model")
        cfg_model = cfg_emb.get("model") or "?"
        if idx_model:
            click.echo(
                f"Warning: index built with embedding model '{idx_model}', "
                f"but config.yaml has '{cfg_model}'. "
                "Run: python -m sntx_sem index --rebuild"
            )
        else:
            click.echo(
                "Warning: index exists but build_meta.json is missing or outdated. "
                "Run: python -m sntx_sem index --rebuild"
            )

    embedding = status.get("config", {}).get("embedding", {})
    provider = embedding.get("provider") or ""
    if provider not in LOCAL_EMBEDDING_PROVIDERS and not embedding.get("api_key_set"):
        click.echo(
            "Warning: embedding API key is not set in config.yaml (api_key) "
            f"or environment ({cfg.embedding.api_key_env})."
        )


@main.command("serve")
@click.option("--host", default=None, help="Bind host (default from config.api.host)")
@click.option("--port", default=None, type=int, help="Bind port (default from config.api.port)")
def serve_cmd(host: str | None, port: int | None) -> None:
    """Start HTTP API and Web UI."""
    import uvicorn

    from sntx_sem.api.app import create_app

    cfg = load_config()
    bind_host = host or cfg.api.host
    bind_port = port or cfg.api.port
    api_app = create_app(cfg)
    uvicorn.run(api_app, host=bind_host, port=bind_port)


@main.command("mcp")
@click.option(
    "--api-url",
    default=None,
    help="HTTP API base URL (default: SNTX_SEM_API_URL or in-process mode)",
)
def mcp_cmd(api_url: str | None) -> None:
    """Start MCP stdio server (thin HTTP client or in-process)."""
    if api_url:
        os.environ["SNTX_SEM_API_URL"] = api_url

    if os.environ.get("SNTX_SEM_API_URL", "").strip():
        from sntx_sem.mcp.stdio_server import run as run_thin

        run_thin()
        return

    from sntx_sem.mcp_server import run as run_local

    run_local()


_backend_cache = None
_index_cache = None


def _get_search_index(cfg: AppConfig) -> HelpIndex:
    global _backend_cache, _index_cache
    if _index_cache is None:
        emb_cfg = align_embedding_with_index(cfg)
        _backend_cache = create_embedding_backend(emb_cfg)
        _index_cache = HelpIndex(cfg.index_dir, _backend_cache, cfg.search)
    return _index_cache


@main.command("search")
@click.argument("query")
@click.option("--domain", default="all")
@click.option("--limit", default=5, type=int)
def search_cmd(query: str, domain: str, limit: int) -> None:
    """Search help index from CLI."""
    cfg = load_config()
    index = _get_search_index(cfg)
    results = index.search(query, domain=domain, limit=limit)
    for i, r in enumerate(results, 1):
        click.echo(f"{i}. [{r.domain}] {r.title} (score={r.score:.4f})")
        click.echo(f"   id={r.id}")


@main.command("scan-examples")
@click.option("--link/--no-link", default=False, help="Use LLM linking (requires API key)")
def scan_examples_cmd(link: bool) -> None:
    """Scan local configs for code examples."""
    cfg = load_config()
    cache_dir = cfg.data_dir / ".sntx_sem_cache"
    all_examples = []
    for lc in cfg.local_configs:
        path = Path(lc.path)
        if not path.is_dir():
            click.echo(f"Skip missing config: {path}")
            continue
        found = scan_config_path(path, lc.label, cache_dir)
        click.echo(f"{lc.label}: {len(found)} examples")
        all_examples.extend(found)

    jsonl = cfg.export_dir / "all_chunks.jsonl"
    topics = []
    if jsonl.is_file():
        with jsonl.open(encoding="utf-8") as f:
            topics = [json.loads(line) for line in f if line.strip()]

    if link and all_examples and topics:
        all_examples = link_examples_batch(all_examples, topics)
        click.echo(f"LLM linked: {len(all_examples)} examples")
    elif all_examples and topics:
        all_examples = keyword_link_examples(all_examples, topics)
        click.echo(f"Keyword linked: {len(all_examples)} examples")

    out = cfg.data_dir / "examples.jsonl"
    export_examples_jsonl(all_examples, out)
    ExamplesStore(out).save(all_examples)
    click.echo(f"Saved {len(all_examples)} examples -> {out}")


if __name__ == "__main__":
    main()
